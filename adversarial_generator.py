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
        self.images_dir = self.data_root
        
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
                 base_path: str = "/openbayes/input/input0",
                 subject_lora_path: str = "/openbayes/input/input0/subject.safetensors",
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
    def investigate_pipeline_processor(self):
        """
        调查你的pipeline image processor到底做了什么
        """
        print("=" * 60)
        print("INVESTIGATING PIPELINE IMAGE PROCESSOR")
        print("=" * 60)

        # 检查processor类型和属性
        processor = self.pipe.image_processor
        print(f"Processor type: {type(processor)}")
        print(f"Processor dir: {[attr for attr in dir(processor) if not attr.startswith('_')]}")

        # 检查是否有normalize参数
        if hasattr(processor, 'do_normalize'):
            print(f"do_normalize: {processor.do_normalize}")
        if hasattr(processor, 'image_mean'):
            print(f"image_mean: {processor.image_mean}")
        if hasattr(processor, 'image_std'):
            print(f"image_std: {processor.image_std}")
        if hasattr(processor, 'config'):
            print(f"config: {processor.config}")

        # 测试一个简单的红色图像
        test_img = Image.new('RGB', (256, 256), color=(255, 0, 0))  # 纯红色
        processed = processor.preprocess(test_img, height=256, width=256)

        print(f"Test image (pure red) processed result:")
        print(f"  Shape: {processed.shape}")
        print(f"  Dtype: {processed.dtype}")
        print(f"  Range: [{processed.min():.6f}, {processed.max():.6f}]")
        print(f"  Mean per channel: {processed.mean(dim=[2,3])}")

        # 如果是normalize的话，红色通道应该是(1-mean)/std的值

    def quick_fix_preprocess_subject_image(self, image: Image.Image, cond_size: int = 512) -> torch.Tensor:
        """
        快速修复版本的预处理，避免使用pipeline的processor
        """
        # 直接使用torchvision的操作，不依赖pipeline processor
        from torchvision import transforms

        w, h = image.size
        scale = cond_size / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)

        # 使用torchvision的transforms
        transform = transforms.Compose([
            transforms.Resize((new_h, new_w)),
            transforms.ToTensor(),  # 自动归一化到[0,1]
        ])

        tensor = transform(image).unsqueeze(0)  # 添加batch维度

        # 手动padding到目标尺寸
        pad_h = cond_size - new_h
        pad_w = cond_size - new_w

        tensor = F.pad(tensor, 
                        (pad_w//2, pad_w-pad_w//2, pad_h//2, pad_h-pad_h//2), 
                        mode='constant', value=0)

        return tensor.to(device=self.device, dtype=torch.float32)

    def test_conversion_round_trip(self, original_image: Image.Image):
        """
        测试转换的往返一致性
        """
        print("=" * 60)
        print("TESTING CONVERSION ROUND TRIP")
        print("=" * 60)

        # 测试1：原始方法
        print("Test 1: Original pipeline method")
        try:
            tensor1 = self.preprocess_subject_image(original_image)
            pil1 = self.tensor_to_pil(tensor1)
            tensor1_back = self.preprocess_subject_image(pil1)
            diff1 = torch.abs(tensor1 - tensor1_back).max()
            print(f"  Round trip difference: {diff1.item():.6f}")
        except Exception as e:
            print(f"  Error: {e}")
            diff1 = float('inf')

        # 测试2：快速修复方法
        print("Test 2: Quick fix method")
        try:
            tensor2 = self.quick_fix_preprocess_subject_image(original_image)
            pil2 = self.tensor_to_pil(tensor2)
            tensor2_back = self.quick_fix_preprocess_subject_image(pil2)
            diff2 = torch.abs(tensor2 - tensor2_back).max()
            print(f"  Round trip difference: {diff2.item():.6f}")
        except Exception as e:
            print(f"  Error: {e}")
            diff2 = float('inf')

        # 测试3：纯tensor方法（无PIL转换）
        print("Test 3: Pure tensor method (no PIL conversion)")
        try:
            tensor3 = self.quick_fix_preprocess_subject_image(original_image)
            # 添加小扰动
            epsilon = 8/255
            delta = torch.randn_like(tensor3) * epsilon * 0.1
            adversarial_tensor = torch.clamp(tensor3 + delta, 0, 1)
            
            # 直接计算差异，不经过PIL
            diff3 = torch.abs(tensor3 - adversarial_tensor).max()
            expected_diff = torch.abs(delta).max()
            print(f"  Expected difference: {expected_diff.item():.6f}")
            print(f"  Actual difference: {diff3.item():.6f}")
            print(f"  Ratio: {(diff3/expected_diff).item():.3f}")
        except Exception as e:
            print(f"  Error: {e}")

        return diff1.item() if diff1 != float('inf') else None, \
                diff2.item() if diff2 != float('inf') else None

    def emergency_gradient_test(self, original_image: Image.Image):
        """
        紧急梯度测试 - 最简单的版本
        """
        print("=" * 60)
        print("EMERGENCY GRADIENT TEST")
        print("=" * 60)

        # 使用最简单的tensor操作
        tensor = self.quick_fix_preprocess_subject_image(original_image)

        # 创建需要梯度的扰动
        delta = torch.zeros_like(tensor, requires_grad=True, device=self.device)

        print(f"Delta requires_grad: {delta.requires_grad}")

        # 最简单的损失函数
        adversarial = tensor + delta
        simple_loss = adversarial.sum()  # 最简单的损失

        print(f"Simple loss: {simple_loss.item():.6f}")
        print(f"Simple loss requires_grad: {simple_loss.requires_grad}")

        # 反向传播
        try:
            simple_loss.backward()
            if delta.grad is not None:
                print(f"✅ Gradient computed! Norm: {delta.grad.norm().item():.6f}")
                return True
            else:
                print("❌ Gradient is None!")
                return False
        except Exception as e:
            print(f"❌ Backward failed: {e}")
            return False

        # 在你的主要攻击方法之前，先运行这些诊断
    def run_emergency_diagnosis(self, original_image: Image.Image):
        """
        运行紧急诊断
        """
        print("🔍 Running emergency diagnosis...")

        # 1. 调查processor
        self.investigate_pipeline_processor()

        # 2. 测试转换一致性
        diff1, diff2 = self.test_conversion_round_trip(original_image)

        # 3. 测试梯度
        gradient_ok = self.emergency_gradient_test(original_image)

        # 4. 给出建议
        print("\n" + "=" * 60)
        print("EMERGENCY DIAGNOSIS RESULTS")
        print("=" * 60)

        if diff1 is not None and diff1 > 0.01:
            print("❌ Pipeline processor method has consistency issues")
            print("💡 Recommendation: Use quick_fix_preprocess_subject_image instead")

        if diff2 is not None and diff2 < 0.01:
            print("✅ Quick fix method shows good consistency")

        if gradient_ok:
            print("✅ Basic gradient flow is working")
        else:
            print("❌ Gradient flow is broken")

        return {
            'pipeline_diff': diff1,
            'quickfix_diff': diff2,
            'gradient_ok': gradient_ok
        }
    
    def _init_pipeline(self, base_path: str, subject_lora_path: str):
        """初始化EasyControl pipeline"""
        print("-" * 50)
        print(base_path)
        print("-" * 50)
        # 加载pipeline
        self.pipe = FluxPipeline.from_pretrained(
            base_path, 
            torch_dtype=torch.bfloat16, 
            device=self.device,
            local_files_only=True
        )
        
        # 加载transformer
        transformer = FluxTransformer2DModel.from_pretrained(
            base_path, 
            subfolder="transformer",
            torch_dtype=torch.bfloat16, 
            device=self.device,
            local_files_only=True
        )
        self.pipe.transformer = transformer
        self.pipe.to(self.device)
        
        # 加载subject control LoRA
        set_single_lora(self.pipe.transformer, subject_lora_path, lora_weights=[1], cond_size=512)
        
        # 确保模型参数需要梯度（用于对抗攻击）
        self.pipe.transformer.requires_grad_(True)
        
        logger.info("EasyControl pipeline initialized successfully!")
    
    def clear_cache(self, transformer):
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
        
        image_np = (tensor.detach().numpy() * 255).astype(np.uint8)
        return Image.fromarray(image_np)
    
    def generate_with_subject(self, prompt: str, subject_image: Image.Image, 
                            height: int = 1024, width: int = 1024, 
                            num_inference_steps: int = 20,
                            enable_grad: bool = True) -> Image.Image:
        """
        使用subject control生成图片
        """
        generation_func = self.pipe
        
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
            self.clear_cache(self.pipe.transformer)
            # 生成对抗图片结果
            generated_adversarial = self.generate_with_subject(
                self.attack_prompt, adversarial_img, enable_grad=True
            )
            self.clear_cache(self.pipe.transformer)
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
        # 🔍 首先运行诊断
        print("🔍 Running emergency diagnosis...")
        diagnosis_results = self.run_emergency_diagnosis(original_image)
        
        # 根据诊断结果选择预处理方法
        if diagnosis_results['pipeline_diff'] is None or diagnosis_results['pipeline_diff'] > 0.01:
            print("⚠️  Using quick fix preprocessing due to pipeline issues")
            preprocess_func = self.quick_fix_preprocess_subject_image
        else:
            print("✅ Using original pipeline preprocessing")
            preprocess_func = self.preprocess_subject_image
            
        # 使用选定的预处理方法
        original_tensor = preprocess_func(original_image, cond_size=512)
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
            'attack_prompt': self.attack_prompt,
            'used_quick_fix': diagnosis_results['pipeline_diff'] is None or diagnosis_results['pipeline_diff'] > 0.01
        }
        
        logger.info(f"Using attack prompt: '{self.attack_prompt}'")
        
        for i in range(num_iterations):
            delta.requires_grad_(True)
            
            # 生成对抗图片
            adversarial_tensor = torch.clamp(original_tensor + delta, 0, 1)
            adversarial_image = self.tensor_to_pil(adversarial_tensor)
            
            # 🔧 修复：使用选定的预处理方法进行测试，而不是固定使用原始方法
            reconstructed_tensor = preprocess_func(adversarial_image)
            actual_diff = torch.abs(original_tensor - reconstructed_tensor).max()
            
            print(f"Iter {i+1}: Expected diff={torch.abs(delta).max().item():.6f}, "
                f"Actual diff={actual_diff.item():.6f}")
            
            if actual_diff.item() < 1e-6:
                print("⚠️  WARNING: 扰动在转换过程中丢失了！")

            # return mse损失
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
        def custom_collate_fn(batch):
            return batch 
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=custom_collate_fn)
        
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
                    original_image = batch[0]['image']
                    image_path = batch[0]['image_path']
                    image_idx = batch[0]['index']
                    
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
    parser.add_argument("--base_model", type=str, default="/openbayes/input/input0",
                       help="Base FLUX model path")
    parser.add_argument("--subject_lora", type=str, default="/openbayes/input/input0/subject.safetensors",
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
python adversarial_generator.py \
    --data_root /openbayes/input/input0/sample_faces \
    --output_dir ./results \
    --epsilon 0.03137 \
    --num_iterations 50

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