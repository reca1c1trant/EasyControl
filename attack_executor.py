import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from typing import Dict, Optional
from pathlib import Path
import argparse

# 导入生成器
from image_space_adversarial_generator import ImageSpaceAdversarialGenerator

class LAIONFaceDataset(Dataset):
    """简化的数据集加载器"""
    def __init__(self, data_root: str, subset_size: Optional[int] = None):
        self.data_root = Path(data_root)
        
        # 加载图片路径
        extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        self.image_paths = []
        for ext in extensions:
            self.image_paths.extend(list(self.data_root.glob(f"*{ext}")))
            self.image_paths.extend(list(self.data_root.glob(f"*{ext.upper()}")))
        self.image_paths = [str(p) for p in self.image_paths]
        
        if subset_size and subset_size < len(self.image_paths):
            np.random.seed(42)
            indices = np.random.choice(len(self.image_paths), subset_size, replace=False)
            self.image_paths = [self.image_paths[i] for i in indices]
        
        print(f"Loaded {len(self.image_paths)} images")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        try:
            image = Image.open(image_path).convert("RGB")
            return {'image': image, 'image_path': str(image_path), 'index': idx}
        except Exception:
            dummy_image = Image.new('RGB', (512, 512), color='white')
            return {'image': dummy_image, 'image_path': str(image_path), 'index': idx}

class MinimalAttackExecutor:
    """最小化的攻击执行器"""
    
    def __init__(self, base_path: str, subject_lora_path: str, device: str = "cuda", output_dir: str = "./results"):
        self.generator = ImageSpaceAdversarialGenerator(
            base_path=base_path,
            subject_lora_path=subject_lora_path,
            device=device
        )
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
    def process_single_image(self, image: Image.Image, image_idx: int, image_path: str,
                           epsilon: float = 0.03, alpha: float = 0.01,
                           num_iterations: int = 50, lambda_reg: float = 0.1) -> Dict:
        """处理单张图像"""
        
        try:
            print(f"Processing image {image_idx}")
            
            # 执行攻击
            delta, attack_info = self.generator.pgd_attack(
                original_image=image,
                epsilon=epsilon,
                alpha=alpha,
                num_iterations=num_iterations,
                lambda_reg=lambda_reg
            )
            
            # 生成结果
            clean_final_latents, clean_main_latents, clean_subject_packed, pipeline_components = self.generator.compute_clean_path(image)
            clean_generated, adversarial_generated, noisy_original = self.generator.generate_final_results(
                image, delta, clean_final_latents, clean_main_latents, pipeline_components
            )
            
            # 计算最终指标
            with torch.no_grad():
                final_noisy_tensor = self.generator.pil_to_tensor_512(image) + delta
                final_noisy_tensor = torch.clamp(final_noisy_tensor, 0, 1)
                noisy_pipeline = self.generator.preprocess_subject(final_noisy_tensor)
                noisy_packed_final = self.generator.encode_subject_with_vae(noisy_pipeline, enable_grad=False)
                
                prompt_embeds, pooled_prompt_embeds, text_ids, latent_image_ids = pipeline_components
                adversarial_final_latents = self.generator.denoise_latents(
                    clean_main_latents, noisy_packed_final,
                    prompt_embeds, pooled_prompt_embeds, text_ids, latent_image_ids,
                    enable_grad=False
                )
                final_mse = F.mse_loss(clean_final_latents, adversarial_final_latents).item()
            
            final_linf_norm = torch.norm(delta, p=float('inf')).item()
            
            # 保存结果
            image.save(self.output_dir / f"{image_idx:06d}_original.png")
            clean_generated.save(self.output_dir / f"{image_idx:06d}_clean.png")
            adversarial_generated.save(self.output_dir / f"{image_idx:06d}_adversarial.png")
            noisy_original.save(self.output_dir / f"{image_idx:06d}_noisy.png")
            
            result = {
                'image_idx': image_idx,
                'final_mse': final_mse,
                'final_linf_norm': final_linf_norm,
                'constraint_satisfied': final_linf_norm <= epsilon,
                'attack_successful': final_mse > 0.01,
                'status': 'SUCCESS'
            }
            
            print(f"✓ Image {image_idx}: MSE={final_mse:.6f}, L∞={final_linf_norm:.6f}")
            return result
            
        except Exception as e:
            print(f"✗ Image {image_idx} failed: {e}")
            return {
                'image_idx': image_idx,
                'error': str(e),
                'status': 'FAILED'
            }
    
    def process_dataset(self, dataset, epsilon: float = 0.03, alpha: float = 0.01,
                       num_iterations: int = 50, lambda_reg: float = 0.1,
                       resume_from: Optional[int] = None):
        """处理数据集"""
        
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, 
                              collate_fn=lambda batch: batch)
        
        total_samples = len(dataset)
        start_idx = resume_from or 0
        success_count = 0
        total_mse = 0.0
        results = []
        
        print(f"Starting processing: {total_samples} images")
        print(f"Parameters: ε={epsilon}, α={alpha}, iterations={num_iterations}")
        
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx < start_idx:
                continue
            
            original_image = batch[0]['image']
            image_path = batch[0]['image_path']
            image_idx = batch[0]['index']
            
            # 跳过小图片
            if original_image.size[0] < 256 or original_image.size[1] < 256:
                print(f"Skipping small image {image_path}")
                continue
            
            # 处理图像
            result = self.process_single_image(
                image=original_image,
                image_idx=image_idx,
                image_path=image_path,
                epsilon=epsilon,
                alpha=alpha,
                num_iterations=num_iterations,
                lambda_reg=lambda_reg
            )
            
            results.append(result)
            
            # 统计
            if result['status'] == 'SUCCESS':
                if result['attack_successful']:
                    success_count += 1
                total_mse += result['final_mse']
            
            # 内存清理
            if batch_idx % 5 == 0:
                torch.cuda.empty_cache()
        
        # 最终统计
        processed_count = len([r for r in results if r['status'] == 'SUCCESS'])
        success_rate = success_count / total_samples if total_samples > 0 else 0
        avg_mse = total_mse / max(processed_count, 1)
        
        print("\n" + "="*60)
        print("FINAL RESULTS:")
        print(f"Total samples: {total_samples}")
        print(f"Successfully processed: {processed_count}")
        print(f"Attack success: {success_count} ({success_rate*100:.1f}%)")
        print(f"Average MSE: {avg_mse:.6f}")
        print(f"Results saved to: {self.output_dir}")
        print("="*60)

def main():
    parser = argparse.ArgumentParser(description="Minimal image space adversarial attack")
    
    # 必要参数
    parser.add_argument("--data_root", type=str, required=True, help="Dataset root directory")
    parser.add_argument("--base_model", type=str, default="/openbayes/input/input0", help="Base FLUX model path")
    parser.add_argument("--subject_lora", type=str, default="/openbayes/input/input0/subject.safetensors", help="Subject LoRA path")
    
    # 攻击参数
    parser.add_argument("--epsilon", type=float, default=0.03, help="L∞ constraint")
    parser.add_argument("--alpha", type=float, default=0.01, help="PGD step size")
    parser.add_argument("--num_iterations", type=int, default=50, help="PGD iterations")
    parser.add_argument("--lambda_reg", type=float, default=0.1, help="Regularization coefficient")
    
    # 系统参数
    parser.add_argument("--subset_size", type=int, default=None, help="Dataset subset size")
    parser.add_argument("--output_dir", type=str, default="./results", help="Output directory")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument("--resume_from", type=int, default=None, help="Resume from index")
    
    args = parser.parse_args()
    
    # 检查CUDA
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = "cpu"
    
    # 参数验证
    if args.epsilon <= 0 or args.epsilon > 1:
        print(f"Invalid epsilon: {args.epsilon}")
        return
    
    if args.alpha <= 0 or args.alpha > args.epsilon:
        print(f"Invalid alpha: {args.alpha}")
        return
    
    # 创建数据集
    dataset = LAIONFaceDataset(args.data_root, args.subset_size)
    if len(dataset) == 0:
        print("Dataset is empty!")
        return
    
    # 创建执行器
    executor = MinimalAttackExecutor(
        base_path=args.base_model,
        subject_lora_path=args.subject_lora,
        device=args.device,
        output_dir=args.output_dir
    )
    
    # 开始处理
    print("="*60)
    print("IMAGE SPACE ADVERSARIAL ATTACK")
    print("="*60)
    print("Algorithm: Real image space perturbation [512,512,3]")
    print(f"Constraint: L∞ ≤ {args.epsilon}")
    print(f"Parameters: α={args.alpha}, iterations={args.num_iterations}")
    print("="*60)
    
    executor.process_dataset(
        dataset=dataset,
        epsilon=args.epsilon,
        alpha=args.alpha,
        num_iterations=args.num_iterations,
        lambda_reg=args.lambda_reg,
        resume_from=args.resume_from
    )

if __name__ == "__main__":
    main()

"""
最小化的图像空间对抗攻击执行器

## 特点:
- 移除所有不必要的日志和调试代码
- 只保留核心算法功能
- 最小化内存消耗
- 简化的结果保存

## 使用:
python attack_executor.py \
    --data_root /openbayes/input/input0/sample_faces \
    --base_model /openbayes/input/input0/ \
    --subject_lora /openbayes/input/input0/subject.safetensors \
    --epsilon 0.03 \
    --alpha 0.01 \
    --num_iterations 50 \
    --subset_size 10

"""