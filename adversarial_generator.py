import os
import sys
import json
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
from tqdm import tqdm
import argparse
from pathlib import Path
import logging
from datetime import datetime
from torchvision.transforms.functional import pad
from torchvision import transforms
from src.pipeline import FluxPipeline
from src.transformer_flux import FluxTransformer2DModel
from src.lora_helper import set_single_lora

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
 
class LAIONFaceDataset(Dataset):
    """LAIONFace数据集加载器"""
    def __init__(self, data_root: str, subset_size: Optional[int] = None):
        self.data_root = Path(data_root)
        self.images_dir = self.data_root
        
        # 加载图片路径
        metadata_path = self.data_root / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                self.image_paths = json.load(f)
        else:
            extensions = ['.jpg', '.jpeg', '.png', '.bmp']
            self.image_paths = []
            for ext in extensions:
                self.image_paths.extend(list(self.images_dir.glob(f"*{ext}")))
                self.image_paths.extend(list(self.images_dir.glob(f"*{ext.upper()}")))
            self.image_paths = [str(p) for p in self.image_paths]
        
        if subset_size and subset_size < len(self.image_paths):
            np.random.seed(42)
            indices = np.random.choice(len(self.image_paths), subset_size, replace=False)
            self.image_paths = [self.image_paths[i] for i in indices]
        
        logger.info(f"Loaded {len(self.image_paths)} images from {data_root}")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        if not os.path.isabs(image_path):
            image_path = self.images_dir / image_path
        
        try:
            image = Image.open(image_path).convert("RGB")
            return {'image': image, 'image_path': str(image_path), 'index': idx}
        except Exception as e:
            logger.warning(f"Failed to load image {image_path}: {e}")
            dummy_image = Image.new('RGB', (512, 512), color='white')
            return {'image': dummy_image, 'image_path': str(image_path), 'index': idx}

class GradientDiagnosticsMixin:
    """梯度诊断混合类"""
    
    def diagnose_gradient_flow(self, tensor: torch.Tensor, name: str = "tensor") -> bool:
        """诊断梯度流动情况"""
        if tensor.grad is None:
            logger.warning(f"{name}: No gradient")
            return False
        
        grad_norm = tensor.grad.norm().item()
        grad_max = tensor.grad.abs().max().item()
        grad_min = tensor.grad.abs().min().item()
        
        logger.debug(f"{name} gradient - norm: {grad_norm:.6f}, max: {grad_max:.6f}, min: {grad_min:.6f}")
        
        if grad_norm < 1e-8:
            logger.warning(f"{name}: Gradient too small (vanishing): {grad_norm:.2e}")
            return False
        elif grad_norm > 1e6:
            logger.warning(f"{name}: Gradient too large (exploding): {grad_norm:.2e}")
            return False
        
        return True
    
    def check_model_gradients(self, model):
        """检查模型各部分是否启用梯度"""
        components = {
            'transformer': model.transformer,
            'vae': model.vae,
            'text_encoder': model.text_encoder if hasattr(model, 'text_encoder') else None,
        }
        
        for name, component in components.items():
            if component is not None and hasattr(component, 'parameters'):
                param_count = sum(1 for p in component.parameters())
                grad_enabled_count = sum(1 for p in component.parameters() if p.requires_grad)
                logger.info(f"{name}: {grad_enabled_count}/{param_count} parameters have gradients enabled")
            else:
                logger.info(f"{name}: No parameters or component is None")

class FixedModifiedFluxPipeline(FluxPipeline, GradientDiagnosticsMixin):
    """修复梯度问题的FluxPipeline"""
    
    def preprocess_subject_tensor(self, subject_tensor: torch.Tensor, cond_size: int = 512) -> torch.Tensor:
        """
        直接处理已经预处理好的subject tensor
        复用原有的padding逻辑，但跳过image_processor.preprocess
        """
        # subject_tensor已经是[1, 3, H, W]格式
        pad_h = cond_size - subject_tensor.shape[-2]
        pad_w = cond_size - subject_tensor.shape[-1]
        
        subject_tensor = pad(
            subject_tensor,
            padding=(int(pad_w / 2), int(pad_h / 2), int(pad_w / 2), int(pad_h / 2)),
            fill=0
        )
        
        return subject_tensor.to(dtype=torch.float32)
    
    @torch.enable_grad()  #  强制启用梯度
    def __call__( 
            self,
            prompt: str,
            subject_tensors: Optional[List[torch.Tensor]] = None,  # 新增：支持tensor输入
            subject_images: Optional[List[Image.Image]] = None,    # 保持兼容性
            height: Optional[int] = None,
            width: Optional[int] = None,
            num_inference_steps: int = 28,
            guidance_scale: float = 3.5,
            num_images_per_prompt: Optional[int] = 1,
            generator: Optional[torch.Generator] = None,
            cond_size: int = 512,
            output_type: str = "pil",
            return_latents: bool = False,  # 新增：可选择返回latents用于loss计算
            **kwargs
    ):
        """
        修改后的调用方法，支持tensor输入，修复梯度问题
        """
        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor
        self.cond_size = cond_size
        
        # 处理subject输入 - 支持tensor或PIL
        sub_number = 0
        subject_image = None
        
        if subject_tensors is not None:
            sub_number = len(subject_tensors)
            subject_tensor_list = []
            for subject_tensor in subject_tensors:
                #  确保输入tensor保持梯度
                if not subject_tensor.requires_grad:
                    subject_tensor = subject_tensor.requires_grad_(True)
                
                # 直接处理tensor，跳过PIL转换
                processed_tensor = self.preprocess_subject_tensor(subject_tensor, cond_size)
                subject_tensor_list.append(processed_tensor)
            subject_image = torch.concat(subject_tensor_list, dim=-2)
            
        elif subject_images is not None:
            # 保持原有的PIL处理逻辑
            sub_number = len(subject_images)
            subject_image_ls = []
            for subject_img in subject_images:
                w, h = subject_img.size[:2]
                scale = self.cond_size / max(h, w)
                new_h, new_w = int(h * scale), int(w * scale)
                subject_tensor = self.image_processor.preprocess(subject_img, height=new_h, width=new_w)
                subject_tensor = subject_tensor.to(dtype=torch.float32)
                processed_tensor = self.preprocess_subject_tensor(subject_tensor, cond_size)
                subject_image_ls.append(processed_tensor)
            subject_image = torch.concat(subject_image_ls, dim=-2)
        
        # 空间条件（当前为空）
        condition_image = None
        cond_number = 0
        
        # 其余处理保持不变
        batch_size = 1
        device = self._execution_device
        
        #  在梯度上下文中执行所有操作
        with torch.set_grad_enabled(True):
            # 编码prompt
            prompt_embeds, pooled_prompt_embeds, text_ids = self.encode_prompt(
                prompt=prompt,
                prompt_2=prompt,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=512,
            )
            
            # 准备latents
            num_channels_latents = self.transformer.config.in_channels // 4
            cond_latents, latent_image_ids, noise_latents = self.prepare_latents(
                batch_size * num_images_per_prompt,
                num_channels_latents,
                height,
                width,
                prompt_embeds.dtype,
                device,
                generator,
                subject_image,
                condition_image,
                None,
                cond_number,
                sub_number
            )
            
            latents = noise_latents
            
            # 准备timesteps
            sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
            
            # 修复相对导入问题 - 直接在这里定义函数或使用绝对导入
            def calculate_shift( 
                    image_seq_len,
                    base_seq_len: int = 256,
                    max_seq_len: int = 4096,
                    base_shift: float = 0.5,
                    max_shift: float = 1.16,
            ):
                m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
                b = base_shift - m * base_seq_len
                mu = image_seq_len * m + b
                return mu
     
            def retrieve_timesteps( 
                    scheduler,
                    num_inference_steps: Optional[int] = None,
                    device: Optional[Union[str, torch.device]] = None,
                    timesteps: Optional[List[int]] = None,
                    sigmas: Optional[List[float]] = None,
                    **kwargs,
            ):
                import inspect
                if timesteps is not None and sigmas is not None:
                    raise ValueError("Only one of `timesteps` or `sigmas` can be passed.")
                if timesteps is not None:
                    accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
                    if not accepts_timesteps:
                        raise ValueError(f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom timestep schedules.")
                    scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
                    timesteps = scheduler.timesteps
                    num_inference_steps = len(timesteps)
                elif sigmas is not None:
                    accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
                    if not accept_sigmas:
                        raise ValueError(f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom sigmas schedules.")
                    scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
                    timesteps = scheduler.timesteps
                    num_inference_steps = len(timesteps)
                else:
                    scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
                    timesteps = scheduler.timesteps
                return timesteps, num_inference_steps
            
            image_seq_len = latents.shape[1]
            mu = calculate_shift(
                image_seq_len,
                self.scheduler.config.base_image_seq_len,
                self.scheduler.config.max_image_seq_len,
                self.scheduler.config.base_shift,
                self.scheduler.config.max_shift,
            )
            timesteps, num_inference_steps = retrieve_timesteps(
                self.scheduler,
                num_inference_steps,
                device,
                None,
                sigmas,
                mu=mu,
            )
            
            # guidance
            if self.transformer.config.guidance_embeds:
                guidance = torch.full([1], guidance_scale, device=device, dtype=torch.float32)
                guidance = guidance.expand(latents.shape[0])
            else:
                guidance = None
            
            # 清除和缓存条件
            for name, attn_processor in self.transformer.attn_processors.items():
                if hasattr(attn_processor, 'bank_kv'):
                    attn_processor.bank_kv.clear()
            
            # warmup缓存
            start_idx = latents.shape[1] - 32
            warmup_latents = latents[:, start_idx:, :]
            warmup_latent_ids = latent_image_ids[start_idx:, :]
            t = torch.tensor([timesteps[0]], device=device)
            timestep = t.expand(warmup_latents.shape[0]).to(latents.dtype)
            _ = self.transformer(
                hidden_states=warmup_latents,
                cond_hidden_states=cond_latents,
                timestep=timestep / 1000,
                guidance=guidance,
                pooled_projections=pooled_prompt_embeds,
                encoder_hidden_states=prompt_embeds,
                txt_ids=text_ids,
                img_ids=warmup_latent_ids,
                joint_attention_kwargs=None,
                return_dict=False,
            )[0]
            
            # 去噪循环
            with self.progress_bar(total=num_inference_steps) as progress_bar:
                for i, t in enumerate(timesteps):
                    timestep = t.expand(latents.shape[0]).to(latents.dtype)
                    noise_pred = self.transformer(
                        hidden_states=latents,
                        cond_hidden_states=cond_latents,
                        timestep=timestep / 1000,
                        guidance=guidance,
                        pooled_projections=pooled_prompt_embeds,
                        encoder_hidden_states=prompt_embeds,
                        txt_ids=text_ids,
                        img_ids=latent_image_ids,
                        joint_attention_kwargs=None,
                        return_dict=False,
                    )[0]
                    
                    latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
                    progress_bar.update()
            
            # 解码和后处理
            if return_latents:
                # 返回解码后的tensor用于loss计算
                latents = self._unpack_latents(latents, height, width, self.vae_scale_factor)
                latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
                decoded_tensor = self.vae.decode(latents.to(dtype=self.vae.dtype), return_dict=False)[0]
                return decoded_tensor
            else:
                # 正常返回PIL图像
                latents = self._unpack_latents(latents, height, width, self.vae_scale_factor)
                latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
                image = self.vae.decode(latents.to(dtype=self.vae.dtype), return_dict=False)[0]
                image = self.image_processor.postprocess(image, output_type=output_type)
                return image

class FixedOptimizedTensorSpaceAdversarialGenerator(GradientDiagnosticsMixin):
    """修复梯度问题的对抗样本生成器"""
    
    def __init__(self, 
                 base_path: str = "/openbayes/input/input0",
                 subject_lora_path: str = "/openbayes/input/input0/subject.safetensors",
                 device: str = "cuda",
                 output_dir: str = "./fixed_adversarial_results"):
        
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建子目录
        self.clean_dir = self.output_dir / "clean_images"
        self.adversarial_dir = self.output_dir / "adversarial_images"
        self.logs_dir = self.output_dir / "logs"
        
        for dir_path in [self.clean_dir, self.adversarial_dir, self.logs_dir]:
            dir_path.mkdir(exist_ok=True)
        
        logger.info("Initializing FIXED EasyControl pipeline...")
        self._init_pipeline(base_path, subject_lora_path)
        
        # 攻击prompt
        self.attack_prompt = "A SKS on the beach"
        logger.info(f"Using attack prompt: '{self.attack_prompt}'")
        
        #  运行基础梯度测试
        self._test_basic_gradients()
    
    def _init_pipeline(self, base_path: str, subject_lora_path: str):
        """初始化修复版本的pipeline"""
        self.pipe = FixedModifiedFluxPipeline.from_pretrained(
            base_path, 
            torch_dtype=torch.bfloat16, 
            device=self.device,
            local_files_only=True
        )
        
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
        
        #  关键修复：确保所有必要组件启用梯度
        self.pipe.transformer.requires_grad_(True)
        self.pipe.vae.requires_grad_(True)  # 关键！
        
        #  确保VAE解码器也启用梯度
        if hasattr(self.pipe.vae, 'decoder'):
            self.pipe.vae.decoder.requires_grad_(True)
        
        #  禁用可能阻断梯度的优化
        if hasattr(self.pipe.vae, 'disable_slicing'):
            self.pipe.vae.disable_slicing()
        if hasattr(self.pipe.vae, 'disable_tiling'):
            self.pipe.vae.disable_tiling()
        
        logger.info("FIXED pipeline initialized with gradient support!")
        
        # 诊断梯度设置
        self.check_model_gradients(self.pipe)
    
    def _test_basic_gradients(self) -> bool:
        """测试基础梯度功能"""
        logger.info("Testing basic gradient functionality...")
        
        try:
            # 创建简单测试张量
            test_tensor = torch.randn(1, 3, 64, 64, device=self.device, requires_grad=True)
            
            # 测试简单操作的梯度
            result = test_tensor * 2 + 1
            loss = result.mean()
            loss.backward()
            
            if test_tensor.grad is not None:
                grad_norm = test_tensor.grad.norm().item()
                logger.info(f"✓ Basic gradient test passed, grad norm: {grad_norm:.6f}")
                return True
            else:
                logger.error("✗ Basic gradient test failed: No gradient")
                return False
                
        except Exception as e:
            logger.error(f"✗ Basic gradient test failed with error: {e}")
            return False
    
    def clear_cache(self):
        """清除attention cache"""
        for name, attn_processor in self.pipe.transformer.attn_processors.items():
            if hasattr(attn_processor, 'bank_kv'):
                attn_processor.bank_kv.clear()
    
    def preprocess_to_tensor(self, image: Image.Image, cond_size: int = 512) -> torch.Tensor:
        """
        将PIL图像预处理为tensor - 复用官方预处理逻辑
        所有操作都是线性的，满足可微分要求
        """
        w, h = image.size
        scale = cond_size / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        
        # 使用官方的image_processor.preprocess
        tensor = self.pipe.image_processor.preprocess(image, height=new_h, width=new_w)
        tensor = tensor.to(dtype=torch.float32)
        
        return tensor.to(device=self.device)
    
    def tensor_to_pil_official(self, tensor: torch.Tensor) -> Image.Image:
        """使用官方方法将tensor转换为PIL图片"""
        if len(tensor.shape) == 4:
            tensor = tensor.squeeze(0)
        
        # 使用官方的postprocess方法
        tensor = tensor.unsqueeze(0)  # 添加batch维度
        image = self.pipe.image_processor.postprocess(tensor, output_type="pil")[0]
        
        return image
    
    @torch.enable_grad()  #  强制启用梯度
    def generate_with_tensor_subject(self, prompt: str, subject_tensor: torch.Tensor,
                                   height: int = 1024, width: int = 1024, 
                                   num_inference_steps: int = 20,
                                   return_latents: bool = False) -> torch.Tensor:
        """
        使用tensor作为subject输入进行生成
        关键：整个过程保持在tensor空间，梯度连续
        """
        #  确保输入tensor启用梯度
        if not subject_tensor.requires_grad:
            subject_tensor = subject_tensor.requires_grad_(True)
        
        #  在梯度上下文中执行
        with torch.set_grad_enabled(True):
            result = self.pipe(
                prompt=prompt,
                subject_tensors=[subject_tensor],  # 直接传入tensor
                height=height,
                width=width,
                guidance_scale=3.5,
                num_inference_steps=num_inference_steps,
                generator=torch.Generator("cpu").manual_seed(42),
                cond_size=512,
                return_latents=return_latents,  # 控制返回类型
            )
        
        return result
    
    def compute_mse_loss_fixed(self, clean_decoded: torch.Tensor, 
                              adversarial_tensor: torch.Tensor) -> torch.Tensor:
        """
        修复版本：带梯度诊断的MSE计算
        """
        try:
            #  确保adversarial_tensor启用梯度
            if not adversarial_tensor.requires_grad:
                adversarial_tensor = adversarial_tensor.requires_grad_(True)
            
            logger.debug(f"Input tensor requires_grad: {adversarial_tensor.requires_grad}")
            
            #  显式启用梯度计算
            with torch.set_grad_enabled(True):
                adversarial_decoded = self.generate_with_tensor_subject(
                    self.attack_prompt, adversarial_tensor, return_latents=True
                )
            
            self.clear_cache()
            
            #  检查解码结果是否有梯度
            if not adversarial_decoded.requires_grad:
                logger.warning("Adversarial decoded tensor has no gradient!")
                # 尝试重新启用梯度（虽然这通常不会成功）
                adversarial_decoded = adversarial_decoded.requires_grad_(True)
            
            logger.debug(f"Decoded tensor requires_grad: {adversarial_decoded.requires_grad}")
            
            # 直接使用预计算的clean_decoded计算MSE
            mse_loss = F.mse_loss(clean_decoded, adversarial_decoded)
            
            #  检查loss是否有梯度
            logger.debug(f"MSE loss requires_grad: {mse_loss.requires_grad}")
            
            if not mse_loss.requires_grad:
                logger.error("MSE loss has no gradient! This will cause optimization failure.")
            
            return mse_loss
            
        except Exception as e:
            logger.error(f"Failed to compute fixed MSE loss: {e}")
            # 返回一个需要梯度的零张量
            return torch.tensor(0.0, device=self.device, requires_grad=True)
    
    def pgd_attack_tensor_space_fixed(self, 
                                    original_image: Image.Image,
                                    epsilon: float = 8/255,
                                    alpha: float = 2/255,
                                    num_iterations: int = 50,
                                    lambda_reg: float = 0.1) -> Tuple[torch.Tensor, Dict]:
        """
        修复版本：带详细梯度诊断的PGD攻击
        """
        
        # 预处理原始图像为tensor
        clean_tensor = self.preprocess_to_tensor(original_image, cond_size=512)
        clean_tensor.requires_grad_(False)
        
        # 预计算clean_decoded
        logger.info("Pre-computing clean decoded tensor...")
        with torch.no_grad():
            clean_decoded = self.generate_with_tensor_subject(
                self.attack_prompt, clean_tensor, return_latents=True
            )
            self.clear_cache()
            clean_decoded = clean_decoded.detach().to(device=self.device, dtype=torch.float32)
            clean_decoded.requires_grad_(False)
        
        logger.info(f"Clean decoded tensor shape: {clean_decoded.shape}, device: {clean_decoded.device}")
        
        #  改进噪声初始化 - 从更小的值开始
        noise_tensor = torch.zeros_like(clean_tensor, requires_grad=True, device=self.device)
        #  从10%的epsilon开始，避免立即被裁剪
        noise_tensor.data = (torch.rand_like(clean_tensor) - 0.5) * 2 * (epsilon * 0.1)
        
        attack_info = {
            'loss_history': [],
            'mse_history': [],
            'gradient_norms': [],
            'gradient_status': [],
            'epsilon': epsilon,
            'alpha': alpha,
            'num_iterations': num_iterations,
            'lambda_reg': lambda_reg,
            'attack_prompt': self.attack_prompt,
            'optimization': 'gradient_fixed'
        }
        
        logger.info(f"Starting FIXED tensor-space PGD attack with {num_iterations} iterations")
        logger.info(" Using gradient-fixed version with enhanced diagnostics")
        
        consecutive_failures = 0
        max_consecutive_failures = 5
        
        for i in range(num_iterations):
            #  确保噪声张量启用梯度
            noise_tensor.requires_grad_(True)
            
            # 在tensor空间直接组合
            adversarial_tensor = torch.clamp(clean_tensor + noise_tensor, 0, 1)
            
            # 验证扰动保持情况
            actual_perturbation = torch.abs(adversarial_tensor - clean_tensor).max()
            
            #  使用修复版本计算MSE
            mse_loss = self.compute_mse_loss_fixed(clean_decoded, adversarial_tensor)
            
            #  如果MSE没有梯度，尝试恢复
            if not mse_loss.requires_grad:
                logger.warning(f"Iteration {i+1}: MSE has no gradient, attempting recovery...")
                consecutive_failures += 1
                
                if consecutive_failures >= max_consecutive_failures:
                    logger.error(f"Too many consecutive gradient failures ({consecutive_failures}), stopping attack")
                    break
                
                # 尝试重新计算
                torch.cuda.empty_cache()
                continue
            
            consecutive_failures = 0  # 重置失败计数
            
            # 计算正则化项
            reg_loss = torch.max(torch.abs(noise_tensor))
            
            # 总损失：最大化MSE，最小化噪声幅度
            total_loss = -mse_loss + lambda_reg * reg_loss
            
            # 记录历史
            attack_info['loss_history'].append(total_loss.item())
            attack_info['mse_history'].append(mse_loss.item())
            
            logger.info(f"Iter {i+1}/{num_iterations}: MSE={mse_loss.item():.6f}, "
                       f"Reg={reg_loss.item():.6f}, Total={total_loss.item():.6f}, "
                       f"Perturbation={actual_perturbation.item():.6f}")
            
            # 反向传播
            total_loss.backward()
            
            #  详细梯度诊断
            if noise_tensor.grad is None:
                logger.error(f"Iteration {i+1}: Gradient is None! Breaking...")
                attack_info['gradient_status'].append('None')
                break
            
            grad_diagnosis = self.diagnose_gradient_flow(noise_tensor, "noise_tensor")
            grad_norm = noise_tensor.grad.norm().item()
            attack_info['gradient_norms'].append(grad_norm)
            attack_info['gradient_status'].append('OK' if grad_diagnosis else 'Poor')
            
            #  自适应步长调整
            if grad_norm < 1e-8:
                logger.warning(f"Iteration {i+1}: Gradient too small ({grad_norm:.2e}), using larger step")
                alpha_adjusted = min(alpha * 10, epsilon)
            elif grad_norm > 1e3:
                logger.warning(f"Iteration {i+1}: Gradient too large ({grad_norm:.2e}), using smaller step")
                alpha_adjusted = alpha * 0.1
            else:
                alpha_adjusted = alpha
            
            #  改进的PGD更新
            with torch.no_grad():
                # 使用调整后的步长
                grad_sign = noise_tensor.grad.sign()
                noise_tensor.data = noise_tensor.data + alpha_adjusted * grad_sign
                
                # 投影到epsilon约束
                noise_tensor.data = torch.clamp(noise_tensor.data, -epsilon, epsilon)
                
                # 确保adversarial tensor在[0,1]范围
                temp_adversarial = clean_tensor + noise_tensor.data
                noise_tensor.data = torch.clamp(temp_adversarial, 0, 1) - clean_tensor
            
            # 清零梯度
            noise_tensor.grad = None
            
            #  早期停止条件
            if i > 10 and len(attack_info['mse_history']) > 10:
                recent_mse = attack_info['mse_history'][-10:]
                if max(recent_mse) - min(recent_mse) < 1e-6:
                    logger.warning(f"MSE converged early at iteration {i+1}")
                    break
            
            # 内存清理
            if i % 10 == 0:
                torch.cuda.empty_cache()
        
        logger.info("FIXED PGD attack completed!")
        logger.info(f"Final gradient status: {attack_info['gradient_status'][-5:] if attack_info['gradient_status'] else 'No gradients'}")
        
        return noise_tensor.detach(), attack_info
    
    def process_dataset(self, 
                       dataset: LAIONFaceDataset,
                       batch_size: int = 1,
                       epsilon: float = 8/255,
                       alpha: float = 2/255,
                       num_iterations: int = 50,
                       lambda_reg: float = 0.1,
                       save_frequency: int = 100,
                       resume_from: Optional[int] = None) -> None:
        """处理整个数据集"""
        
        def custom_collate_fn(batch):
            return batch 
            
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, 
                              num_workers=0, collate_fn=custom_collate_fn)
        
        total_samples = len(dataset)
        start_idx = resume_from or 0
        success_count = 0
        total_mse_improvement = 0.0
        results_log = []
        
        logger.info(f"Starting FIXED tensor-space adversarial generation for {total_samples} images")
        logger.info(f"Parameters: epsilon={epsilon}, alpha={alpha}, iterations={num_iterations}")
        logger.info(" Using FIXED algorithm with gradient diagnostics")
        
        with tqdm(dataloader, desc="Processing images (FIXED)") as pbar:
            for batch_idx, batch in enumerate(pbar):
                if batch_idx < start_idx:
                    continue
                
                try:
                    original_image = batch[0]['image']
                    image_path = batch[0]['image_path']
                    image_idx = batch[0]['index']
                    
                    # 检查图片质量
                    if original_image.size[0] < 256 or original_image.size[1] < 256:
                        logger.warning(f"Skipping small image {image_path}")
                        continue
                    
                    #  执行修复版本的tensor空间PGD攻击
                    noise_tensor, attack_info = self.pgd_attack_tensor_space_fixed(
                        original_image=original_image,
                        epsilon=epsilon,
                        alpha=alpha,
                        num_iterations=num_iterations,
                        lambda_reg=lambda_reg
                    )
                    
                    # 生成最终的对抗样本
                    clean_tensor = self.preprocess_to_tensor(original_image, cond_size=512)
                    adversarial_tensor = torch.clamp(clean_tensor + noise_tensor, 0, 1)
                    
                    # 使用官方方法转换为PIL图像保存
                    adversarial_image = self.tensor_to_pil_official(adversarial_tensor)
                    
                    # 计算最终MSE
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
                        'timestamp': datetime.now().isoformat(),
                        'optimization_used': 'FIXED_GRADIENT',
                        'gradient_success': len([s for s in attack_info.get('gradient_status', []) if s == 'OK'])
                    }
                    results_log.append(result_entry)
                    
                    # 统计
                    if final_mse > 0.01:
                        success_count += 1
                    total_mse_improvement += final_mse
                    
                    # 更新进度条
                    gradient_success_rate = len([s for s in attack_info.get('gradient_status', []) if s == 'OK']) / max(len(attack_info.get('gradient_status', [])), 1)
                    pbar.set_postfix({
                        'Success': f"{success_count}/{batch_idx+1}",
                        'Avg_MSE': f"{total_mse_improvement/(batch_idx+1):.4f}",
                        'Current_MSE': f"{final_mse:.4f}",
                        'Grad_Success': f"{gradient_success_rate:.1%}",
                        'FIXED': '✓'
                    })
                    
                    # 定期保存
                    if (batch_idx + 1) % save_frequency == 0:
                        self._save_progress_log(results_log, batch_idx + 1)
                
                except Exception as e:
                    logger.error(f"Failed to process image {batch_idx}: {e}")
                    continue
                
                # 内存清理
                if batch_idx % 10 == 0:
                    torch.cuda.empty_cache()
        
        # 保存最终结果
        self._save_final_results(results_log, success_count, total_samples)
        logger.info(f"FIXED generation completed! Success rate: {success_count}/{total_samples}")
    
    def _save_progress_log(self, results_log: List[Dict], current_idx: int):
        """保存进度日志"""
        log_path = self.logs_dir / f"fixed_progress_{current_idx}.json"
        with open(log_path, 'w') as f:
            json.dump(results_log, f, indent=2)
    
    def _save_final_results(self, results_log: List[Dict], success_count: int, total_samples: int):
        """保存最终结果"""
        final_log_path = self.logs_dir / "fixed_final_results.json"
        
        # 计算梯度统计
        total_gradient_checks = sum(len(r.get('attack_info', {}).get('gradient_status', [])) for r in results_log)
        successful_gradients = sum(len([s for s in r.get('attack_info', {}).get('gradient_status', []) if s == 'OK']) for r in results_log)
        gradient_success_rate = successful_gradients / max(total_gradient_checks, 1)
        
        summary = {
            'total_samples': total_samples,
            'success_count': success_count,
            'success_rate': success_count / total_samples if total_samples > 0 else 0,
            'gradient_statistics': {
                'total_gradient_checks': total_gradient_checks,
                'successful_gradients': successful_gradients,
                'gradient_success_rate': gradient_success_rate
            },
            'optimization_info': {
                'version': 'FIXED_GRADIENT_v1.0',
                'fixes_applied': [
                    'VAE gradient enabled',
                    'Pipeline gradient context',
                    'Improved noise initialization',
                    'Adaptive step size',
                    'Gradient diagnostics',
                    'Early stopping conditions'
                ]
            },
            'results': results_log,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(final_log_path, 'w') as f:
            json.dump(summary, f, indent=2)
 
def main():
    parser = argparse.ArgumentParser(description="FIXED tensor-space adversarial generation for EasyControl")
    
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
    parser.add_argument("--output_dir", type=str, default="./fixed_adversarial_results",
                       help="Output directory")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Computing device")
    parser.add_argument("--save_frequency", type=int, default=100,
                       help="Save progress every N samples")
    parser.add_argument("--resume_from", type=int, default=None,
                       help="Resume from specific sample index")
    
    args = parser.parse_args()
    
    # 检查CUDA可用性
    if args.device == "cuda" and not torch.cuda.is_available():
        logger.error("CUDA requested but not available! Falling back to CPU.")
        args.device = "cpu"
    
    # 创建数据集
    logger.info(f"Loading dataset from {args.data_root}")
    dataset = LAIONFaceDataset(args.data_root, args.subset_size)
    
    # 创建修复版本的tensor空间生成器
    logger.info("Initializing FIXED tensor-space adversarial generator")
    generator = FixedOptimizedTensorSpaceAdversarialGenerator(
        base_path=args.base_model,
        subject_lora_path=args.subject_lora,
        device=args.device,
        output_dir=args.output_dir
    )
    
    # 开始处理
    logger.info("Starting FIXED tensor-space adversarial generation")
    generator.process_dataset(
        dataset=dataset,
        epsilon=args.epsilon,
        alpha=args.alpha,
        num_iterations=args.num_iterations,
        lambda_reg=args.lambda_reg,
        save_frequency=args.save_frequency,
        resume_from=args.resume_from
    )
    
    logger.info("FIXED generation completed!")
 
if __name__ == "__main__":
    main()
 
"""
修复版本使用命令:
python adversarial_generator.py \
    --data_root /openbayes/input/input0/sample_faces \
    --base_model /openbayes/input/input0 \
    --subject_lora /openbayes/input/input0/subject.safetensors \
    --epsilon 0.03137 \
    --alpha 0.00784 \
    --num_iterations 50 \
    --lambda_reg 0.1 \
    --output_dir ./fixed_adversarial_results \
    --device cuda 



"""