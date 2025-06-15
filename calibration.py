import os
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import argparse
import logging
from pathlib import Path
from torchvision.transforms.functional import pad

from src.pipeline import FluxPipeline
from src.transformer_flux import FluxTransformer2DModel
from src.lora_helper import set_single_lora

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EpsilonCalibrator:
    def __init__(self, 
                 base_path: str = "/openbayes/input/input0",
                 subject_lora_path: str = "/openbayes/input/input0/subject.safetensors",
                 device: str = "cuda"):
        """
        Epsilon校准器 - 将图像空间的epsilon映射到token空间
        """
        self.device = device
        logger.info("Initializing EasyControl pipeline for calibration...")
        self._init_pipeline(base_path, subject_lora_path)
    
    def _init_pipeline(self, base_path: str, subject_lora_path: str):
        """初始化pipeline"""
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
        
        logger.info("Pipeline initialized successfully!")
    
    def preprocess_subject_image(self, image: Image.Image, cond_size: int = 512) -> torch.Tensor:
        """预处理subject图片到 [1, 3, 512, 512]"""
        w, h = image.size[:2]
        scale = cond_size / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        
        # 使用pipeline自带的图像处理器
        subject_image = self.pipe.image_processor.preprocess(image, height=new_h, width=new_w)
        subject_image = subject_image.to(dtype=torch.float32)
        
        # 填充逻辑
        pad_h = cond_size - subject_image.shape[-2]
        pad_w = cond_size - subject_image.shape[-1]
        subject_image = pad(
            subject_image,
            padding=(int(pad_w / 2), int(pad_h / 2), int(pad_w / 2), int(pad_h / 2)),
            fill=0
        )
        
        return subject_image.to(device=self.device)
    
    def image_to_tokens(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """
        将图像tensor转换为token space
        [1, 3, 512, 512] → [1, 4096, 64]
        """
        # Step 1: VAE编码
        with torch.no_grad():
            # 使用pipeline的VAE编码方法
            image_latents = self.pipe._encode_vae_image(image=image_tensor, generator=None)
            # 结果: [1, 16, 64, 64]
        
        # Step 2: Pack latents
        batch_size = image_tensor.shape[0]
        num_channels_latents = 16
        height_cond = 512  # cond_size
        width_cond = 512
        
        # Pack到token序列
        packed_latents = self.pipe._pack_latents(
            image_latents, batch_size, num_channels_latents, height_cond, width_cond
        )
        # 结果: [1, 4096, 64]
        
        return packed_latents
    
    def calibrate_epsilon_mapping(self, test_images: list, epsilon_image_values: list):
        """
        校准图像空间epsilon到token空间的映射
        
        Args:
            test_images: 测试图像列表 (PIL Images)
            epsilon_image_values: 图像空间epsilon值列表
        
        Returns:
            dict: 映射关系和统计信息
        """
        logger.info("开始epsilon校准...")
        
        results = {
            'mappings': [],
            'statistics': {}
        }
        
        all_ratios = []
        
        for epsilon_image in epsilon_image_values:
            logger.info(f"校准 epsilon_image = {epsilon_image:.6f}")
            
            image_ratios = []
            
            for idx, test_image in enumerate(test_images):
                # 预处理图像
                clean_tensor = self.preprocess_subject_image(test_image)  # [1, 3, 512, 512]
                
                # 添加图像空间扰动
                perturbed_tensor = torch.clamp(clean_tensor + epsilon_image, 0, 1)
                
                # 转换到token space
                clean_tokens = self.image_to_tokens(clean_tensor)      # [1, 4096, 64]
                perturbed_tokens = self.image_to_tokens(perturbed_tensor)  # [1, 4096, 64]
                
                # 计算token space差异
                token_diff = torch.abs(perturbed_tokens - clean_tokens)
                epsilon_token = token_diff.max().item()
                
                # 计算映射比例
                ratio = epsilon_token / epsilon_image if epsilon_image > 0 else 0
                image_ratios.append(ratio)
                
                logger.info(f"  Image {idx+1}: epsilon_token = {epsilon_token:.8f}, ratio = {ratio:.2f}")
            
            # 统计当前epsilon的结果
            avg_ratio = np.mean(image_ratios)
            std_ratio = np.std(image_ratios)
            avg_epsilon_token = epsilon_image * avg_ratio
            
            mapping_info = {
                'epsilon_image': epsilon_image,
                'epsilon_token_avg': avg_epsilon_token,
                'ratio_avg': avg_ratio,
                'ratio_std': std_ratio,
                'ratios': image_ratios
            }
            
            results['mappings'].append(mapping_info)
            all_ratios.extend(image_ratios)
            
            logger.info(f"  Average ratio: {avg_ratio:.4f} ± {std_ratio:.4f}")
            logger.info(f"  Recommended epsilon_token: {avg_epsilon_token:.8f}")
        
        # 全局统计
        results['statistics'] = {
            'overall_ratio_mean': np.mean(all_ratios),
            'overall_ratio_std': np.std(all_ratios),
            'overall_ratio_min': np.min(all_ratios),
            'overall_ratio_max': np.max(all_ratios),
        }
        
        logger.info("=== 校准完成 ===")
        logger.info(f"总体映射比例: {results['statistics']['overall_ratio_mean']:.4f} ± {results['statistics']['overall_ratio_std']:.4f}")
        logger.info(f"比例范围: [{results['statistics']['overall_ratio_min']:.4f}, {results['statistics']['overall_ratio_max']:.4f}]")
        
        return results
    
    def test_reconstruction_precision(self, test_image: Image.Image):
        """测试重建精度"""
        logger.info("测试重建精度...")
        
        # 预处理
        original_tensor = self.preprocess_subject_image(test_image)
        
        # 转换到token space再转回来
        tokens = self.image_to_tokens(original_tensor)
        
        # 这里我们无法直接转回，但可以添加小扰动测试敏感性
        small_delta = torch.randn_like(tokens) * 1e-6
        perturbed_tokens = tokens + small_delta
        
        # 计算token差异
        token_diff = F.mse_loss(tokens, perturbed_tokens)
        
        logger.info(f"Token space sensitivity: {token_diff.item():.10f}")
        
        return token_diff.item()

def create_test_images():
    """创建测试图像"""
    test_images = []
    
    # 创建不同类型的测试图像
    patterns = [
        ("uniform_gray", lambda: np.ones((512, 512, 3)) * 0.5),  # 中性灰
        ("random_noise", lambda: np.random.rand(512, 512, 3)),   # 随机噪声
        ("gradient", lambda: np.tile(np.linspace(0, 1, 512).reshape(1, -1, 1), (512, 1, 3))),  # 梯度
        ("checkerboard", lambda: np.tile(((np.arange(512)[:, None] + np.arange(512)) % 64 < 32).astype(float)[:, :, None], (1, 1, 3))),  # 棋盘
    ]
    
    for name, pattern_func in patterns:
        image_array = pattern_func()
        image_array = (image_array * 255).astype(np.uint8)
        pil_image = Image.fromarray(image_array, 'RGB')
        test_images.append(pil_image)
        logger.info(f"Created test image: {name}")
    
    return test_images

def main():
    parser = argparse.ArgumentParser(description="Epsilon校准工具")
    parser.add_argument("--base_model", type=str, default="/openbayes/input/input0",
                       help="Base FLUX model path")
    parser.add_argument("--subject_lora", type=str, default="/openbayes/input/input0/subject.safetensors",
                       help="Subject LoRA model path")
    parser.add_argument("--device", type=str, default="cuda", help="Computing device")
    parser.add_argument("--output_file", type=str, default="./epsilon_calibration_results.json",
                       help="输出校准结果文件")
    
    args = parser.parse_args()
    
    # 创建校准器
    calibrator = EpsilonCalibrator(
        base_path=args.base_model,
        subject_lora_path=args.subject_lora,
        device=args.device
    )
    
    # 创建测试图像
    test_images = create_test_images()
    
    # 定义要测试的epsilon值
    epsilon_values = [
        1/255,    # 很小的扰动
        4/255,    # 小扰动  
        8/255,    # 标准扰动
        16/255,   # 大扰动
        32/255,   # 很大的扰动
    ]
    
    # 执行校准
    results = calibrator.calibrate_epsilon_mapping(test_images, epsilon_values)
    
    # 保存结果
    import json
    with open(args.output_file, 'w') as f:
        # 转换numpy类型为Python原生类型以便JSON序列化
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            return obj
        
        # 递归转换
        def recursive_convert(data):
            if isinstance(data, dict):
                return {k: recursive_convert(v) for k, v in data.items()}
            elif isinstance(data, list):
                return [recursive_convert(item) for item in data]
            else:
                return convert_numpy(data)
        
        json.dump(recursive_convert(results), f, indent=2)
    
    logger.info(f"校准结果已保存到: {args.output_file}")
    
    # 测试重建精度
    calibrator.test_reconstruction_precision(test_images[0])
    
    # 推荐设置
    overall_ratio = results['statistics']['overall_ratio_mean']
    logger.info("\n=== 推荐设置 ===")
    for epsilon_img in [4/255, 8/255, 16/255]:
        epsilon_token = epsilon_img * overall_ratio
        alpha_token = epsilon_token / 10
        logger.info(f"epsilon_image={epsilon_img:.4f} → epsilon_token={epsilon_token:.8f}, alpha_token={alpha_token:.8f}")

if __name__ == "__main__":
    main()