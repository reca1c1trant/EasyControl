import os
import json
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
import argparse
from pathlib import Path
import logging
from datetime import datetime
from torchvision.transforms.functional import pad

from src.pipeline import FluxPipeline
from src.transformer_flux import FluxTransformer2DModel
from src.lora_helper import set_single_lora

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LAIONFaceDataset(Dataset):
    """
    LAIONFace数据集加载器
    数据结构假设：
    data_root/
    ├── images/
    │   ├── 000000.jpg
    │   ├── 000001.jpg
    │   └── ...
    └── metadata.json  # 包含图片路径列表
    """
    def __init__(self, data_root: str, subset_size: Optional[int] = None):
        self.data_root = Path(data_root)
        self.images_dir = self.data_root / "images"
        
        # 如果有metadata文件就读取，否则扫描images目录
        metadata_path = self.data_root / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                self.image_paths = json.load(f)
        else:
            # 扫描images目录，支持常见图片格式
            extensions = ['.jpg', '.jpeg', '.png', '.bmp']
            self.image_paths = []
            for ext in extensions:
                self.image_paths.extend(list(self.images_dir.glob(f"*{ext}")))
                self.image_paths.extend(list(self.images_dir.glob(f"*{ext.upper()}")))
            self.image_paths = [str(p) for p in self.image_paths]
        
        # 如果指定了子集大小，就随机采样
        if subset_size and subset_size < len(self.image_paths):
            np.random.seed(42)  # 固定种子确保可复现
            indices = np.random.choice(len(self.image_paths), subset_size, replace=False)
            self.image_paths = [self.image_paths[i] for i in indices]
        
        logger.info(f"Loaded {len(self.image_paths)} images from {data_root}")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        
        # 如果路径是相对路径，就加上data_root
        if not os.path.isabs(image_path):
            image_path = self.images_dir / image_path
        
        try:
            image = Image.open(image_path).convert("RGB")
            return {
                'image': image,
                'image_path': str(image_path),
                'index': idx
            }
        except Exception as e:
            logger.warning(f"Failed to load image {image_path}: {e}")
            # 返回一个空白图片作为fallback
            dummy_image = Image.new('RGB', (512, 512), color='white')
            return {
                'image': dummy_image,
                'image_path': str(image_path),
                'index': idx
            }

class OptimizedAdversarialGenerator:
    def __init__(self, 
                 base_path: str = "black-forest-labs/FLUX.1-dev",
                 subject_lora_path: str = "./checkpoints/models/subject.safetensors",
                 device: str = "cuda",
                 output_dir: str = "./adversarial_results"):
        """
        优化版大规模对抗样本生成器
        
        Args:
            base_path: FLUX基础模型路径
            subject_lora_path: subject control LoRA模型路径
            device: 计算设备
            output_dir: 输出目录
        """
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建子目录
        self.clean_dir = self.output_dir / "clean_images"
        self.adversarial_dir = self.output_dir / "adversarial_images"
        self.logs_dir = self.output_dir / "logs"
        
        for dir_path in [self.clean_dir, self.adversarial_dir, self.logs_dir]:
            dir_path.mkdir(exist_ok=True)
        
        logger.info("Initializing EasyControl pipeline...")
        self._init_pipeline(base_path, subject_lora_path)
        
        # 使用单个攻击prompt
        self.attack_prompt = "A SKS on the beach"
        
        logger.info(f"Using attack prompt: '{self.attack_prompt}'")
    
    def _init_pipeline(self, base_path: str, subject_lora_path: str):
        """初始化EasyControl pipeline"""
        
        # 加载pipeline
        self.pipe = FluxPipeline.from_pretrained(
            base_path, 
            torch_dtype=torch.bfloat16, 
            device=self.device
        )
        
        # 加载transformer
        transformer = FluxTransformer2DModel.from_pretrained(
            base_path, 
            subfolder="transformer",
            torch_dtype=torch.bfloat16, 
            device=self.device
        )
        self.pipe.transformer = transformer
        self.pipe.to(self.device)
        
        # 加载subject control LoRA
        set_single_lora(self.pipe.transformer, subject_lora_path, lora_weights=[1], cond_size=512)
        
        # 确保模型参数需要梯度（用于对抗攻击）
        self.pipe.transformer.requires_grad_(True)
        
        logger.info("EasyControl pipeline initialized successfully!")
    
    def clear_cache(transformer):
        for name, attn_processor in transformer.attn_processors.items():
            attn_processor.bank_kv.clear()    
    
    def preprocess_subject_image(self, image: Image.Image, cond_size: int = 512) -> torch.Tensor:
        """
        使用Pipeline的预处理逻辑处理subject图片
        复用Pipeline中prepare_latents的预处理部分
        """
        w, h = image.size[:2]
        scale = cond_size / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        
        # 使用pipeline自带的图像处理器
        subject_image = self.pipe.image_processor.preprocess(image, height=new_h, width=new_w)
        subject_image = subject_image.to(dtype=torch.float32)
        
        # 填充逻辑 - 复用pipeline中的pad方法
        pad_h = cond_size - subject_image.shape[-2]
        pad_w = cond_size - subject_image.shape[-1]
        subject_image = pad(
            subject_image,
            padding=(int(pad_w / 2), int(pad_h / 2), int(pad_w / 2), int(pad_h / 2)),
            fill=0
        )
        
        return subject_image.to(device=self.device)
    
    def tensor_to_pil(self, tensor: torch.Tensor) -> Image.Image:
        """将tensor转换为PIL图片 - 使用pipeline的后处理逻辑"""
        if len(tensor.shape) == 4:
            tensor = tensor.squeeze(0)
        
        # 使用pipeline的后处理方法
        tensor = tensor.cpu()
        tensor = torch.clamp(tensor, 0, 1)
        
        # [C, H, W] -> [H, W, C]
        if len(tensor.shape) == 3 and tensor.shape[0] == 3:
            tensor = tensor.permute(1, 2, 0)
        
        image_np = (tensor.numpy() * 255).astype(np.uint8)
        return Image.fromarray(image_np)
    
    def generate_with_subject(self, prompt: str, subject_image: Image.Image, 
                            height: int = 1024, width: int = 1024, 
                            num_inference_steps: int = 20,
                            enable_grad: bool = True) -> Image.Image:
        """
        使用subject control生成图片
        """
        generation_func = self.pipe if not enable_grad else self.pipe
        
        if enable_grad:
            # 启用梯度计算
            image = generation_func(
                prompt,
                height=height,
                width=width,
                guidance_scale=3.5,
                num_inference_steps=num_inference_steps,
                max_sequence_length=512,
                generator=torch.Generator("cpu").manual_seed(42),
                subject_images=[subject_image],
                cond_size=512,
            ).images[0]
        else:
            # 不需要梯度
            with torch.no_grad():
                image = generation_func(
                    prompt,
                    height=height,
                    width=width,
                    guidance_scale=3.5,
                    num_inference_steps=num_inference_steps,
                    max_sequence_length=512,
                    generator=torch.Generator("cpu").manual_seed(42),
                    subject_images=[subject_image],
                    cond_size=512,
                ).images[0]
        
        # 智能缓存清理
        self.clear_cache()
        return image
    
    def compute_single_prompt_mse(self, original_img: Image.Image, 
                                 adversarial_img: Image.Image) -> torch.Tensor:
        """
        计算单个prompt下的MSE损失 - 优化版本
        """
        try:
            # 生成原始图片结果
            generated_original = self.generate_with_subject(
                self.attack_prompt, original_img, enable_grad=True
            )
            
            # 生成对抗图片结果
            generated_adversarial = self.generate_with_subject(
                self.attack_prompt, adversarial_img, enable_grad=True
            )
            
            # 使用pipeline的图像处理器进行预处理以保持一致性
            orig_tensor = self.pipe.image_processor.preprocess(generated_original)
            adv_tensor = self.pipe.image_processor.preprocess(generated_adversarial)
            
            # 确保tensor在同一设备上
            orig_tensor = orig_tensor.to(self.device)
            adv_tensor = adv_tensor.to(self.device)
            
            # 计算MSE
            mse = F.mse_loss(orig_tensor, adv_tensor)
            return mse
            
        except Exception as e:
            logger.warning(f"Failed to process prompt '{self.attack_prompt}': {e}")
            return torch.tensor(0.0, device=self.device)
    
    def pgd_attack_single_image(self, 
                               original_image: Image.Image,
                               epsilon: float = 8/255,
                               alpha: float = 2/255,
                               num_iterations: int = 50,
                               lambda_reg: float = 0.1) -> Tuple[torch.Tensor, Dict]:
        """
        对单张图片进行PGD攻击 - 优化版本
        """
        # 使用pipeline的预处理方法
        original_tensor = self.preprocess_subject_image(original_image, cond_size=512)
        original_tensor.requires_grad_(False)  # 原始图片不需要梯度
        
        # 初始化随机噪声
        delta = torch.zeros_like(original_tensor, requires_grad=True)
        delta.data = (torch.rand_like(original_tensor) - 0.5) * 2 * epsilon
        
        attack_info = {
            'loss_history': [],
            'mse_history': [],
            'epsilon': epsilon,
            'alpha': alpha,
            'num_iterations': num_iterations,
            'lambda_reg': lambda_reg,
            'attack_prompt': self.attack_prompt
        }
        
        logger.info(f"Using attack prompt: '{self.attack_prompt}'")
        
        for i in range(num_iterations):
            delta.requires_grad_(True)
            
            # 生成对抗图片
            adversarial_tensor = torch.clamp(original_tensor + delta, 0, 1)
            adversarial_image = self.tensor_to_pil(adversarial_tensor)
            
            # 计算单prompt损失
            mse_loss = self.compute_single_prompt_mse(original_image, adversarial_image)
            
            # 计算正则化项 (L∞范数)
            reg_loss = torch.max(torch.abs(delta))
            
            # 总损失：最大化MSE，最小化噪声
            total_loss = -mse_loss + lambda_reg * reg_loss
            
            # 记录历史
            attack_info['loss_history'].append(total_loss.item())
            attack_info['mse_history'].append(mse_loss.item())
            
            logger.info(f"Iter {i+1}/{num_iterations}: MSE={mse_loss.item():.6f}, "
                       f"Reg={reg_loss.item():.6f}, Total={total_loss.item():.6f}")
            
            # 反向传播
            total_loss.backward()
            
            # PGD更新
            with torch.no_grad():
                delta.data = delta.data + alpha * delta.grad.sign()
                
                # 投影到epsilon约束范围
                delta.data = torch.clamp(delta.data, -epsilon, epsilon)
                
                # 确保对抗图片在[0,1]范围内
                delta.data = torch.clamp(original_tensor + delta.data, 0, 1) - original_tensor
            
            # 清零梯度
            delta.grad = None
        return delta.detach(), attack_info
    
    def process_dataset(self, 
                       dataset: LAIONFaceDataset,
                       batch_size: int = 1,
                       epsilon: float = 8/255,
                       alpha: float = 2/255,
                       num_iterations: int = 50,
                       lambda_reg: float = 0.1,
                       save_frequency: int = 100,
                       resume_from: Optional[int] = None) -> None:
        """
        处理整个数据集
        """
        
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        
        # 创建进度条
        total_samples = len(dataset)
        start_idx = resume_from or 0
        
        # 用于统计
        success_count = 0
        total_mse_improvement = 0.0
        
        # 结果日志
        results_log = []
        
        logger.info(f"Starting adversarial generation for {total_samples} images")
        logger.info(f"Parameters: epsilon={epsilon}, alpha={alpha}, iterations={num_iterations}")
        logger.info(f"Attack prompt: '{self.attack_prompt}'")
        
        with tqdm(dataloader, desc="Processing images") as pbar:
            for batch_idx, batch in enumerate(pbar):
                # 如果需要恢复，跳过已处理的
                if batch_idx < start_idx:
                    continue
                
                try:
                    # 获取图片（当前batch_size=1）
                    original_image = batch['image'][0]
                    image_path = batch['image_path'][0]
                    image_idx = batch['index'][0].item()
                    
                    # 检查图片质量
                    if original_image.size[0] < 256 or original_image.size[1] < 256:
                        logger.warning(f"Skipping small image {image_path}")
                        continue
                    
                    # 执行PGD攻击
                    adversarial_noise, attack_info = self.pgd_attack_single_image(
                        original_image=original_image,
                        epsilon=epsilon,
                        alpha=alpha,
                        num_iterations=num_iterations,
                        lambda_reg=lambda_reg
                    )
                    
                    # 生成对抗图片
                    original_tensor = self.preprocess_subject_image(original_image, cond_size=512)
                    adversarial_tensor = torch.clamp(original_tensor + adversarial_noise, 0, 1)
                    adversarial_image = self.tensor_to_pil(adversarial_tensor)
                    
                    # 计算最终的MSE提升
                    final_mse = attack_info['mse_history'][-1] if attack_info['mse_history'] else 0
                    
                    # 保存结果
                    clean_path = self.clean_dir / f"{image_idx:06d}_clean.png"
                    adversarial_path = self.adversarial_dir / f"{image_idx:06d}_adversarial.png"
                    
                    original_image.save(clean_path)
                    adversarial_image.save(adversarial_path)
                    
                    # 记录结果
                    result_entry = {
                        'image_idx': image_idx,
                        'original_path': str(image_path),
                        'clean_path': str(clean_path),
                        'adversarial_path': str(adversarial_path),
                        'final_mse': final_mse,
                        'attack_info': attack_info,
                        'timestamp': datetime.now().isoformat()
                    }
                    results_log.append(result_entry)
                    
                    # 统计
                    if final_mse > 0.01:  # 认为成功的阈值
                        success_count += 1
                    total_mse_improvement += final_mse
                    
                    # 更新进度条
                    pbar.set_postfix({
                        'Success': f"{success_count}/{batch_idx+1}",
                        'Avg_MSE': f"{total_mse_improvement/(batch_idx+1):.4f}",
                        'Current_MSE': f"{final_mse:.4f}"
                    })
                    
                    # 定期保存日志
                    if (batch_idx + 1) % save_frequency == 0:
                        self._save_progress_log(results_log, batch_idx + 1)
                        logger.info(f"Saved progress at sample {batch_idx + 1}")
                    
                except Exception as e:
                    logger.error(f"Failed to process image {batch_idx}: {e}")
                    continue
                
                # 内存清理
                if batch_idx % 10 == 0:
                    torch.cuda.empty_cache()
        
        # 保存最终结果
        self._save_final_results(results_log, success_count, total_samples)
        logger.info(f"Completed! Success rate: {success_count}/{total_samples} ({100*success_count/total_samples:.1f}%)")
    
    def _save_progress_log(self, results_log: List[Dict], current_idx: int):
        """保存进度日志"""
        log_path = self.logs_dir / f"progress_{current_idx:06d}.json"
        with open(log_path, 'w') as f:
            json.dump(results_log, f, indent=2)
    
    def _save_final_results(self, results_log: List[Dict], success_count: int, total_samples: int):
        """保存最终结果"""
        
        # 保存详细日志
        final_log_path = self.logs_dir / "final_results.json"
        with open(final_log_path, 'w') as f:
            json.dump(results_log, f, indent=2)
        
        # 保存统计摘要
        summary = {
            'total_samples': total_samples,
            'successful_attacks': success_count,
            'success_rate': success_count / total_samples if total_samples > 0 else 0,
            'average_mse': sum(r['final_mse'] for r in results_log) / len(results_log) if results_log else 0,
            'completion_time': datetime.now().isoformat()
        }
        
        summary_path = self.logs_dir / "summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Results saved to {self.output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Optimized large-scale adversarial sample generation for EasyControl")
    
    # 数据集参数
    parser.add_argument("--data_root", type=str, required=True,
                       help="Root directory of LAIONFace dataset")
    parser.add_argument("--subset_size", type=int, default=None,
                       help="Use only a subset of the dataset")
    
    # 模型参数
    parser.add_argument("--base_model", type=str, default="black-forest-labs/FLUX.1-dev",
                       help="Base FLUX model path")
    parser.add_argument("--subject_lora", type=str, default="./checkpoints/models/subject.safetensors",
                       help="Subject LoRA model path")
    
    # 攻击参数
    parser.add_argument("--epsilon", type=float, default=8/255,
                       help="Maximum perturbation magnitude")
    parser.add_argument("--alpha", type=float, default=2/255,
                       help="PGD step size")
    parser.add_argument("--num_iterations", type=int, default=50,
                       help="Number of PGD iterations")
    parser.add_argument("--lambda_reg", type=float, default=0.1,
                       help="Regularization coefficient")
    
    # 系统参数
    parser.add_argument("--output_dir", type=str, default="./adversarial_results",
                       help="Output directory")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Computing device")
    parser.add_argument("--batch_size", type=int, default=1,
                       help="Batch size (only 1 supported currently)")
    parser.add_argument("--save_frequency", type=int, default=100,
                       help="Save progress every N samples")
    parser.add_argument("--resume_from", type=int, default=None,
                       help="Resume from specific sample index")
    
    args = parser.parse_args()
    
    # 创建数据集
    logger.info(f"Loading dataset from {args.data_root}")
    dataset = LAIONFaceDataset(args.data_root, args.subset_size)
    
    # 创建生成器
    logger.info("Initializing optimized adversarial generator")
    generator = OptimizedAdversarialGenerator(
        base_path=args.base_model,
        subject_lora_path=args.subject_lora,
        device=args.device,
        output_dir=args.output_dir
    )
    
    # 开始处理
    logger.info("Starting optimized large-scale adversarial generation")
    generator.process_dataset(
        dataset=dataset,
        batch_size=args.batch_size,
        epsilon=args.epsilon,
        alpha=args.alpha,
        num_iterations=args.num_iterations,
        lambda_reg=args.lambda_reg,
        save_frequency=args.save_frequency,
        resume_from=args.resume_from
    )
    
    logger.info("Generation completed!")

if __name__ == "__main__":
    main()

# 使用示例脚本
"""
# 基本使用
python optimized_adversarial_generator.py \
    --data_root /path/to/laionface \
    --subset_size 1000 \
    --output_dir ./results \
    --epsilon 0.03137 \
    --num_iterations 20

# 高质量攻击（更多迭代）
python optimized_adversarial_generator.py \
    --data_root /path/to/laionface \
    --epsilon 0.03137 \
    --alpha 0.00784 \
    --num_iterations 30 \
    --output_dir ./high_quality_results

# 从中断处恢复
python optimized_adversarial_generator.py \
    --data_root /path/to/laionface \
    --resume_from 500 \
    --output_dir ./results
"""