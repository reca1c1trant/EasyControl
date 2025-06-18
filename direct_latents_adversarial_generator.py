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
 
# 修复导入路径问题
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, script_dir)
sys.path.insert(0, parent_dir)
 
# 尝试多种导入路径

from src.pipeline import FluxPipeline
from src.transformer_flux import FluxTransformer2DModel
from src.lora_helper import set_single_lora

 
# 设置环境变量优化内存
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["PYTORCH_NO_CUDA_MEMORY_CACHING"] = "1"
 
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
 
    def log_memory_usage(self, step_name: str):
        """记录内存使用情况"""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            free = total - reserved
            logger.info(f"{step_name}: GPU Memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB, Free: {free:.2f}GB")
 
class MemoryOptimizedFluxPipeline(FluxPipeline, GradientDiagnosticsMixin):
    """内存优化的FluxPipeline"""
    
    def preprocess_subject_tensor(self, subject_tensor: torch.Tensor, cond_size: int = 512) -> torch.Tensor:
        """预处理subject tensor"""
        pad_h = cond_size - subject_tensor.shape[-2]
        pad_w = cond_size - subject_tensor.shape[-1]
        
        subject_tensor = pad(
            subject_tensor,
            padding=(int(pad_w / 2), int(pad_h / 2), int(pad_w / 2), int(pad_h / 2)),
            fill=0
        )
        
        return subject_tensor.to(dtype=torch.bfloat16)

# 从pipeline.py复制的辅助函数
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
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps

def retrieve_latents(encoder_output: torch.Tensor, generator: Optional[torch.Generator] = None, sample_mode: str = "sample"):
    """获取VAE编码结果"""
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        return encoder_output.latent_dist.sample(generator)
    elif hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        return encoder_output.latent_dist.mode()
    elif hasattr(encoder_output, "latents"):
        return encoder_output.latents
    else:
        raise AttributeError("Could not access latents of provided encoder_output")
 
class DirectLatentsAdversarialGenerator(GradientDiagnosticsMixin):
    """直接在主要latents上加扰动的对抗样本生成器"""
    
    def __init__(self, 
                 base_path: str = "/openbayes/input/input0",
                 subject_lora_path: str = "/openbayes/input/input0/subject.safetensors",
                 device: str = "cuda",
                 output_dir: str = "./direct_latents_adversarial_results"):
        
        self.device = torch.device(device)  # 修复：转换为torch.device对象
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建子目录
        self.clean_dir = self.output_dir / "clean_images"
        self.adversarial_dir = self.output_dir / "adversarial_images"
        self.logs_dir = self.output_dir / "logs"
        
        for dir_path in [self.clean_dir, self.adversarial_dir, self.logs_dir]:
            dir_path.mkdir(exist_ok=True)
        
        logger.info("Initializing DIRECT-LATENTS adversarial generator...")
        self.log_memory_usage("Before initialization")
        
        self._init_pipeline(base_path, subject_lora_path)
        
        self.attack_prompt = "A SKS on the beach"
        logger.info(f"Using attack prompt: '{self.attack_prompt}'")
        
        self.log_memory_usage("After initialization")
        
        # 测试基础功能
        self._test_basic_gradients()
        
    def _init_pipeline(self, base_path: str, subject_lora_path: str):
        """初始化pipeline - 只启用Transformer梯度"""
        logger.info("Loading base model...")
        self.pipe = MemoryOptimizedFluxPipeline.from_pretrained(
            base_path, 
            torch_dtype=torch.bfloat16, 
            local_files_only=True
        )
        
        logger.info("Loading transformer...")
        transformer = FluxTransformer2DModel.from_pretrained(
            base_path, 
            subfolder="transformer",
            torch_dtype=torch.bfloat16, 
            local_files_only=True
        )
        self.pipe.transformer = transformer
        self.pipe.to(self.device)
        
        logger.info("Loading LoRA...")
        # 加载subject control LoRA
        set_single_lora(self.pipe.transformer, subject_lora_path, lora_weights=[1], cond_size=512)
        
        # 关键：只启用Transformer梯度，VAE完全frozen
        logger.info("Setting up gradients...")
        self.pipe.transformer.requires_grad_(True)
        self.pipe.vae.requires_grad_(False)  # 显式禁用VAE梯度
        self.pipe.transformer.gradient_checkpointing = True
        # 禁用VAE的优化功能
        if hasattr(self.pipe.vae, 'disable_slicing'):
            self.pipe.vae.disable_slicing()
        if hasattr(self.pipe.vae, 'disable_tiling'):
            self.pipe.vae.disable_tiling()
        
        logger.info("DIRECT-LATENTS pipeline initialized!")
        logger.info("VAE gradients DISABLED, MSE comparison in denoised latent space")
        logger.info(f"VAE dtype: {self.pipe.vae.dtype}")
        
        # 诊断梯度设置
        self.check_model_gradients(self.pipe)
    
    def encode_image_to_latents(self, image: Image.Image) -> torch.Tensor:
        """将图像编码为latents（无梯度）"""
        with torch.no_grad():  # VAE编码无梯度
            # 预处理图像
            tensor = self.preprocess_to_tensor(image, cond_size=512)
            
            # 修复：确保输入tensor与VAE模型数据类型一致
            tensor = tensor.to(dtype=self.pipe.vae.dtype, device=str(self.device))
            
            # 修复：generator设备类型与计算设备一致
            generator = torch.Generator(str(self.device)).manual_seed(42)
            latents = self._encode_vae_image(tensor, generator)
            
            return latents.to(device=str(self.device), dtype=torch.bfloat16)
    
    def _encode_vae_image(self, image: torch.Tensor, generator: torch.Generator):
        """编码VAE图像"""
        # 修复：确保输入与VAE模型数据类型一致
        image = image.to(dtype=self.pipe.vae.dtype, device=str(self.device))
        
        if isinstance(generator, list):
            image_latents = [
                retrieve_latents(self.pipe.vae.encode(image[i: i + 1]), generator=generator[i])
                for i in range(image.shape[0])
            ]
            image_latents = torch.cat(image_latents, dim=0)
        else:
            image_latents = retrieve_latents(self.pipe.vae.encode(image), generator=generator)
 
        image_latents = (image_latents - self.pipe.vae.config.shift_factor) * self.pipe.vae.config.scaling_factor
        return image_latents
    
    def decode_latents_to_tensor(self, latents: torch.Tensor) -> torch.Tensor:
        """将latents解码为图像tensor（用于最终显示）"""
        with torch.no_grad():  # 最终解码不需要梯度
            # 修复：确保latents与VAE模型数据类型一致
            latents = (latents / self.pipe.vae.config.scaling_factor) + self.pipe.vae.config.shift_factor
            latents = latents.to(dtype=self.pipe.vae.dtype, device=str(self.device))
            decoded_tensor = self.pipe.vae.decode(latents, return_dict=False)[0]
            return decoded_tensor
    
    def preprocess_to_tensor(self, image: Image.Image, cond_size: int = 512) -> torch.Tensor:
        """预处理图像为tensor"""
        w, h = image.size
        scale = cond_size / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        
        tensor = self.pipe.image_processor.preprocess(image, height=new_h, width=new_w)
        # 修复：暂时保持bfloat16，在使用时再转换为VAE的数据类型
        tensor = tensor.to(dtype=torch.bfloat16)
        
        return tensor.to(device=str(self.device))
    
    def prepare_clean_latents_and_conditions(self, subject_image: Image.Image, 
                                           height: int = 1024, width: int = 1024) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        按照pipeline.py逻辑准备clean latents和条件信息
        返回：(clean_main_latents, subject_condition_latents, pipeline_components)
        """
        logger.debug("Preparing clean latents using EasyControl pipeline logic...")
        
        batch_size = 1
        device = self.device  # torch.device对象
        cond_size = 512
        
        with torch.no_grad():
            # 1. 编码prompt（按照pipeline.py逻辑）
            prompt_embeds, pooled_prompt_embeds, text_ids = self.pipe.encode_prompt(
                prompt=self.attack_prompt,
                prompt_2=self.attack_prompt,
                device=str(device),  # 修复：encode_prompt需要字符串设备名
                num_images_per_prompt=1,
                max_sequence_length=512,
            )
            logger.debug(f"Prompt embeds shape: {prompt_embeds.shape}")
            
            # 2. 准备subject图像（完全按照pipeline.py的预处理逻辑）
            w, h = subject_image.size
            scale = cond_size / max(h, w)
            new_h, new_w = int(h * scale), int(w * scale)
            
            subject_image_tensor = self.pipe.image_processor.preprocess(subject_image, height=new_h, width=new_w)
            subject_image_tensor = subject_image_tensor.to(dtype=torch.bfloat16)
            
            # Padding到cond_size
            pad_h = cond_size - subject_image_tensor.shape[-2]
            pad_w = cond_size - subject_image_tensor.shape[-1]
            subject_image_tensor = pad(
                subject_image_tensor,
                padding=(int(pad_w / 2), int(pad_h / 2), int(pad_w / 2), int(pad_h / 2)),
                fill=0
            )
            logger.debug(f"Subject image tensor shape: {subject_image_tensor.shape}")
            
            # 3. 调用原始pipeline的prepare_latents（但不使用spatial_images）
            self.pipe.cond_size = cond_size
            num_channels_latents = self.pipe.transformer.config.in_channels // 4  # 16
            
            generator = torch.Generator(str(device)).manual_seed(42)  # 固定种子
            
            # 直接调用原始pipeline的prepare_latents方法
            cond_latents, latent_image_ids, clean_main_latents = self.pipe.prepare_latents(
                batch_size,
                num_channels_latents,
                height,
                width,
                prompt_embeds.dtype,
                device,  # 修复：prepare_latents需要字符串设备名
                generator,
                subject_image=subject_image_tensor,  # subject图像
                condition_image=None,  # 不使用spatial条件
                latents=None,
                cond_number=0,  # 没有spatial条件
                sub_number=1    # 一个subject
            )
            
            logger.debug(f"Prepared latents - main: {clean_main_latents.shape}, cond: {cond_latents.shape}, ids: {latent_image_ids.shape}")
            
            # 4. 返回组件信息
            pipeline_components = {
                'prompt_embeds': prompt_embeds,
                'pooled_prompt_embeds': pooled_prompt_embeds,
                'text_ids': text_ids,
                'latent_image_ids': latent_image_ids,
                'height': height,
                'width': width,
                'num_inference_steps': 5,
                'guidance_scale': 3.5
            }
            
            return clean_main_latents, cond_latents, pipeline_components
    
    def denoise_latents_with_perturbation(self, main_latents: torch.Tensor, 
                                        cond_latents: torch.Tensor,
                                        pipeline_components: Dict,
                                        enable_grad: bool = True) -> torch.Tensor:
        """
        使用扰动后的主要latents进行去噪，返回去噪后的latents
        完全按照pipeline.py的去噪逻辑
        """
        if enable_grad:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.use_deterministic_algorithms(True, warn_only=True)
        
        logger.debug(f"Starting denoising with main_latents shape: {main_latents.shape}")
        
        # 提取pipeline组件
        prompt_embeds = pipeline_components['prompt_embeds']
        pooled_prompt_embeds = pipeline_components['pooled_prompt_embeds']
        text_ids = pipeline_components['text_ids']
        latent_image_ids = pipeline_components['latent_image_ids']
        height = pipeline_components['height']
        width = pipeline_components['width']
        num_inference_steps = pipeline_components['num_inference_steps']
        guidance_scale = pipeline_components['guidance_scale']
        
        device = self.device  # torch.device对象
        latents = main_latents.clone()  # 复制以避免修改原始tensor
        
        with torch.set_grad_enabled(enable_grad):
            # 准备timesteps（按照pipeline.py逻辑）
            sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
            image_seq_len = latents.shape[1]
            mu = calculate_shift(
                image_seq_len,
                self.pipe.scheduler.config.base_image_seq_len,
                self.pipe.scheduler.config.max_image_seq_len,
                self.pipe.scheduler.config.base_shift,
                self.pipe.scheduler.config.max_shift,
            )
            timesteps, num_inference_steps = retrieve_timesteps(
                self.pipe.scheduler,
                num_inference_steps,
                str(device),  # 修复：scheduler需要字符串设备名
                None,
                sigmas,
                mu=mu,
            )
            
            # guidance设置（按照pipeline.py逻辑）
            if self.pipe.transformer.config.guidance_embeds:
                guidance = torch.full([1], guidance_scale, device=str(device), dtype=torch.bfloat16)
                guidance = guidance.expand(latents.shape[0])
            else:
                guidance = None
            
            # 清除attention cache（按照pipeline.py逻辑）
            for name, attn_processor in self.pipe.transformer.attn_processors.items():
                if hasattr(attn_processor, 'bank_kv'):
                    attn_processor.bank_kv.clear()
                if hasattr(attn_processor, '_cache'):
                    attn_processor._cache = None
            original_training_mode = self.pipe.transformer.training
            if enable_grad:
                self.pipe.transformer.eval()
            # Warmup缓存（按照pipeline.py逻辑）
            start_idx = latents.shape[1] - 32
            warmup_latents = latents[:, start_idx:, :]
            warmup_latent_ids = latent_image_ids[start_idx:, :]
            t = torch.tensor([timesteps[0]], device=str(device))
            timestep = t.expand(warmup_latents.shape[0]).to(latents.dtype)
            
            _ = self.pipe.transformer(
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
            
            # 去噪循环（按照pipeline.py逻辑）
            try:
                for i, t in enumerate(timesteps):
                    torch.cuda.empty_cache()
                    timestep = t.expand(latents.shape[0]).to(latents.dtype)

                    # transformer推理
                    noise_pred = self.pipe.transformer(
                        hidden_states=latents,           # 主要latents（包含扰动）
                        cond_hidden_states=cond_latents, # 条件latents（固定）
                        timestep=timestep / 1000,
                        guidance=guidance,
                        pooled_projections=pooled_prompt_embeds,
                        encoder_hidden_states=prompt_embeds,
                        txt_ids=text_ids,
                        img_ids=latent_image_ids,
                        joint_attention_kwargs=None,
                        return_dict=False,
                    )[0]

                    # 调度器更新
                    latents = self.pipe.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                    if i % 5 == 0:
                        logger.debug(f"Denoising step {i+1}/{num_inference_steps}")
            finally:
                # 🔥 恢复原始状态
                self.pipe.transformer.train(original_training_mode)
                if enable_grad:
                    torch.backends.cudnn.deterministic = False
                    torch.backends.cudnn.benchmark = True
                    torch.use_deterministic_algorithms(False)

            
            logger.debug(f"Denoising completed, final latents shape: {latents.shape}")
            return latents
    
    def pgd_attack_direct_latents(self, 
                                original_image: Image.Image,
                                epsilon: float = 0.1,
                                alpha: float = 0.02,
                                num_iterations: int = 50,
                                lambda_reg: float = 0.1) -> Tuple[torch.Tensor, Dict]:
        """
        直接在主要latents上进行PGD攻击
        算法：
        1. 记录clean_main_latents和clean_denoised_latents  
        2. 在iteration中直接对main_latents加噪声
        3. 在denoised_latents比较MSE
        4. 梯度更新delta噪声
        """
        
        # 第一步：准备clean latents和条件
        logger.info("Preparing clean latents and conditions...")
        self.log_memory_usage("Before clean preparation")
        
        clean_main_latents, subject_condition_latents, pipeline_components = self.prepare_clean_latents_and_conditions(
            original_image
        )
        
        self.log_memory_usage("After clean preparation")
        logger.info(f"Clean main latents shape: {clean_main_latents.shape}")
        logger.info(f"Subject condition latents shape: {subject_condition_latents.shape}")
        
        # 第二步：计算clean的去噪结果
        logger.info("Computing clean denoised latents...")
        with torch.no_grad():
            clean_denoised_latents = self.denoise_latents_with_perturbation(
                clean_main_latents, subject_condition_latents, pipeline_components, enable_grad=False
            )
            clean_denoised_latents = clean_denoised_latents.detach().to(device=str(self.device), dtype=torch.bfloat16)
            clean_denoised_latents.requires_grad_(False)
        
        self.log_memory_usage("After clean denoising")
        logger.info(f"Clean denoised latents shape: {clean_denoised_latents.shape}")
        
        # 第三步：初始化可学习的扰动delta
        # 扰动的shape与main_latents相同: (1, 4096, 16)
        delta_perturbation = torch.zeros_like(clean_main_latents, requires_grad=True, device=str(self.device), dtype=torch.bfloat16)
        
        # 初始化为小的随机扰动
        with torch.no_grad():
            delta_perturbation.data = (torch.randn_like(clean_main_latents) * epsilon * 0.1).to(device=str(self.device), dtype=torch.bfloat16)
        
        attack_info = {
            'loss_history': [],
            'mse_history': [],
            'gradient_norms': [],
            'gradient_status': [],
            'perturbation_norms': [],
            'epsilon': epsilon,
            'alpha': alpha,
            'num_iterations': num_iterations,
            'lambda_reg': lambda_reg,
            'attack_prompt': self.attack_prompt,
            'space': 'direct_main_latents',
            'comparison': 'denoised_latents_mse'
        }
        
        logger.info(f"Starting DIRECT-LATENTS PGD attack with {num_iterations} iterations")
        logger.info(f"Attacking MAIN latents directly, shape: {clean_main_latents.shape}")
        logger.info(f"Delta epsilon: {epsilon}, alpha: {alpha}")
        
        consecutive_failures = 0
        max_consecutive_failures = 5
        
        for i in range(num_iterations):
            self.log_memory_usage(f"Iteration {i+1} start")
            
            delta_perturbation.requires_grad_(True)
            
            # 第四步：在主要latents上加扰动
            adversarial_main_latents = clean_main_latents + delta_perturbation
            
            # 计算扰动幅度
            perturbation_norm = delta_perturbation.norm().item()
            attack_info['perturbation_norms'].append(perturbation_norm)
            
            # 第五步：使用扰动后的latents进行去噪
            adversarial_denoised_latents = self.denoise_latents_with_perturbation(
                adversarial_main_latents, subject_condition_latents, pipeline_components, enable_grad=True
            )
            
            # 第六步：计算MSE损失
            mse_loss = F.mse_loss(clean_denoised_latents, adversarial_denoised_latents)
            
            if not mse_loss.requires_grad:
                logger.warning(f"Iteration {i+1}: MSE has no gradient, attempting recovery...")
                consecutive_failures += 1
                
                if consecutive_failures >= max_consecutive_failures:
                    logger.error(f"Too many consecutive gradient failures ({consecutive_failures}), stopping attack")
                    break
                
                torch.cuda.empty_cache()
                continue
            
            consecutive_failures = 0  # 重置失败计数
            
            # 正则化项（L2范数约束）
            reg_loss = delta_perturbation.norm()
            
            # 第七步：总损失（最大化MSE，最小化正则化）
            total_loss = -mse_loss + lambda_reg * reg_loss
            
            # 记录
            attack_info['loss_history'].append(total_loss.item())
            attack_info['mse_history'].append(mse_loss.item())
            
            logger.info(f"Iter {i+1}/{num_iterations}: MSE={mse_loss.item():.6f}, "
                       f"Reg={reg_loss.item():.6f}, Total={total_loss.item():.6f}, "
                       f"Delta_norm={perturbation_norm:.6f}")
            
            # 第八步：反向传播和梯度更新
            total_loss.backward()
            
            if delta_perturbation.grad is None:
                logger.error(f"Iteration {i+1}: Gradient is None!")
                attack_info['gradient_status'].append('None')
                break
            
            # 梯度诊断
            grad_diagnosis = self.diagnose_gradient_flow(delta_perturbation, "delta_perturbation")
            grad_norm = delta_perturbation.grad.norm().item()
            attack_info['gradient_norms'].append(grad_norm)
            attack_info['gradient_status'].append('OK' if grad_diagnosis else 'Poor')
            
            # 自适应步长调整
            if grad_norm < 1e-8:
                alpha_adjusted = min(alpha * 10, epsilon)
            elif grad_norm > 1e3:
                alpha_adjusted = alpha * 0.1
            else:
                alpha_adjusted = alpha
            
            # PGD更新
            with torch.no_grad():
                # 梯度上升（最大化MSE）
                delta_perturbation.data = delta_perturbation.data + alpha_adjusted * delta_perturbation.grad.sign()
                
                # L∞约束投影
                delta_perturbation.data = torch.clamp(delta_perturbation.data, -epsilon, epsilon)
            
            # 清零梯度
            delta_perturbation.grad = None
            
            # 早期停止
            if i > 10 and len(attack_info['mse_history']) > 10:
                recent_mse = attack_info['mse_history'][-10:]
                if max(recent_mse) - min(recent_mse) < 1e-6:
                    logger.warning(f"MSE converged early at iteration {i+1}")
                    break
            
            # 内存清理
            if i % 5 == 0:
                torch.cuda.empty_cache()
                self.log_memory_usage(f"Iteration {i+1} after cleanup")
        
        logger.info("DIRECT-LATENTS PGD attack completed!")
        logger.info(f"Final gradient status: {attack_info['gradient_status'][-5:] if attack_info['gradient_status'] else 'No gradients'}")
        logger.info(f"Final perturbation norm: {delta_perturbation.norm().item():.6f}")
        
        return delta_perturbation.detach(), attack_info
    
    def tensor_to_pil_official(self, tensor: torch.Tensor) -> Image.Image:
        """将tensor转换为PIL图片"""
        if len(tensor.shape) == 4:
            tensor = tensor.squeeze(0)
        
        tensor = tensor.unsqueeze(0)  # 添加batch维度
        image = self.pipe.image_processor.postprocess(tensor, output_type="pil")[0]
        return image
    
    def _test_basic_gradients(self):
        """测试基础梯度功能"""
        logger.info("Testing basic gradient functionality...")
        
        try:
            test_tensor = torch.randn(1, 4096, 16, device=str(self.device), requires_grad=True)  # main latents shape
            result = test_tensor * 2 + 1
            loss = result.mean()
            loss.backward()
            
            if test_tensor.grad is not None:
                grad_norm = test_tensor.grad.norm().item()
                logger.info(f"✓ Basic gradient test passed, grad norm: {grad_norm:.6f}")
                return True
            else:
                logger.error("✗ Basic gradient test failed")
                return False
        except Exception as e:
            logger.error(f"✗ Basic gradient test failed: {e}")
            return False
    
    def process_dataset(self, 
                       dataset,
                       epsilon: float = 0.1,  # latent space epsilon
                       alpha: float = 0.02,   # latent space alpha
                       num_iterations: int = 50,
                       lambda_reg: float = 0.1,
                       save_frequency: int = 100,
                       resume_from: Optional[int] = None):
        """处理数据集"""
        
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, 
                              collate_fn=lambda batch: batch)
        
        total_samples = len(dataset)
        start_idx = resume_from or 0
        success_count = 0
        total_mse_improvement = 0.0
        results_log = []
        
        logger.info(f"Starting DIRECT-LATENTS adversarial generation for {total_samples} images")
        logger.info(f"Attack target: main latents shape (1, 4096, 16)")
        logger.info(f"MSE comparison: denoised latents space")
        logger.info(f"Parameters: epsilon={epsilon}, alpha={alpha}")
        
        with tqdm(dataloader, desc="Processing (DIRECT-LATENTS)") as pbar:
            for batch_idx, batch in enumerate(pbar):
                if batch_idx < start_idx:
                    continue
                
                try:
                    original_image = batch[0]['image']
                    image_path = batch[0]['image_path']
                    image_idx = batch[0]['index']
                    
                    # 跳过小图片
                    if original_image.size[0] < 256 or original_image.size[1] < 256:
                        logger.warning(f"Skipping small image {image_path}")
                        continue
                    
                    self.log_memory_usage(f"Processing image {batch_idx+1}")
                    
                    # 执行直接latents PGD攻击
                    delta_perturbation, attack_info = self.pgd_attack_direct_latents(
                        original_image=original_image,
                        epsilon=epsilon,
                        alpha=alpha,
                        num_iterations=num_iterations,
                        lambda_reg=lambda_reg
                    )
                    
                    # 生成最终对抗样本
                    logger.info("Generating final adversarial sample...")
                    clean_main_latents, subject_condition_latents, pipeline_components = self.prepare_clean_latents_and_conditions(
                        original_image
                    )
                    
                    # 应用最终扰动
                    final_adversarial_latents = clean_main_latents + delta_perturbation
                    
                    # 最终去噪和VAE解码
                    with torch.no_grad():
                        adversarial_denoised = self.denoise_latents_with_perturbation(
                            final_adversarial_latents, subject_condition_latents, pipeline_components, enable_grad=False
                        )
                        
                        # Unpack和VAE解码
                        adversarial_unpacked = self.pipe._unpack_latents(
                            adversarial_denoised, 
                            pipeline_components['height'], 
                            pipeline_components['width'], 
                            self.pipe.vae_scale_factor
                        )
                        adversarial_decoded = self.decode_latents_to_tensor(adversarial_unpacked)
                        adversarial_image = self.tensor_to_pil_official(adversarial_decoded)
                    
                    # 保存结果
                    clean_path = self.clean_dir / f"{image_idx:06d}_clean.png"
                    adversarial_path = self.adversarial_dir / f"{image_idx:06d}_adversarial.png"
                    
                    original_image.save(clean_path)
                    adversarial_image.save(adversarial_path)
                    
                    # 记录结果
                    final_mse = attack_info['mse_history'][-1] if attack_info['mse_history'] else 0
                    final_perturbation_norm = attack_info['perturbation_norms'][-1] if attack_info['perturbation_norms'] else 0
                    
                    result_entry = {
                        'image_idx': image_idx,
                        'original_path': str(image_path),
                        'clean_path': str(clean_path),
                        'adversarial_path': str(adversarial_path),
                        'final_mse': final_mse,
                        'final_perturbation_norm': final_perturbation_norm,
                        'attack_info': attack_info,
                        'timestamp': datetime.now().isoformat(),
                        'method': 'DIRECT_LATENTS_ATTACK',
                        'gradient_success': len([s for s in attack_info.get('gradient_status', []) if s == 'OK'])
                    }
                    results_log.append(result_entry)
                    
                    if final_mse > 0.01:
                        success_count += 1
                    total_mse_improvement += final_mse
                    
                    # 计算梯度成功率
                    gradient_success_rate = len([s for s in attack_info.get('gradient_status', []) if s == 'OK']) / max(len(attack_info.get('gradient_status', [])), 1)
                    
                    pbar.set_postfix({
                        'Success': f"{success_count}/{batch_idx+1}",
                        'Avg_MSE': f"{total_mse_improvement/(batch_idx+1):.4f}",
                        'Current_MSE': f"{final_mse:.4f}",
                        'Delta_norm': f"{final_perturbation_norm:.4f}",
                        'Grad_Success': f"{gradient_success_rate:.1%}",
                        'Method': 'DIRECT-LATENTS'
                    })
                    
                    # 定期保存
                    if (batch_idx + 1) % save_frequency == 0:
                        log_path = self.logs_dir / f"direct_latents_progress_{batch_idx + 1}.json"
                        with open(log_path, 'w') as f:
                            json.dump(results_log, f, indent=2)
                
                except Exception as e:
                    logger.error(f"Failed to process image {batch_idx}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
                
                # 更频繁的内存清理
                if batch_idx % 2 == 0:
                    torch.cuda.empty_cache()
        
        # 保存最终结果
        final_log_path = self.logs_dir / "direct_latents_final_results.json"
        
        # 计算梯度统计
        total_gradient_checks = sum(len(r.get('attack_info', {}).get('gradient_status', [])) for r in results_log)
        successful_gradients = sum(len([s for s in r.get('attack_info', {}).get('gradient_status', []) if s == 'OK']) for r in results_log)
        gradient_success_rate = successful_gradients / max(total_gradient_checks, 1)
        
        summary = {
            'total_samples': total_samples,
            'success_count': success_count,
            'success_rate': success_count / total_samples if total_samples > 0 else 0,
            'average_mse': total_mse_improvement / max(success_count, 1),
            'gradient_statistics': {
                'total_gradient_checks': total_gradient_checks,
                'successful_gradients': successful_gradients,
                'gradient_success_rate': gradient_success_rate
            },
            'method_info': {
                'method': 'DIRECT_LATENTS_ATTACK',
                'attack_target': 'main_latents_shape_(1,4096,16)',
                'mse_comparison': 'denoised_latents_space',
                'advantages': [
                    'Direct perturbation on main latents',
                    'Stable gradient flow',
                    'Precise control over perturbation magnitude',
                    'Follows EasyControl pipeline exactly',
                    'Subject conditions remain fixed'
                ],
                'parameters': {
                    'epsilon': epsilon,
                    'alpha': alpha,
                    'num_iterations': num_iterations,
                    'lambda_reg': lambda_reg
                }
            },
            'results': results_log,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(final_log_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"DIRECT-LATENTS generation completed!")
        logger.info(f"Success rate: {success_count}/{total_samples} ({success_count/total_samples*100:.1f}%)")
        logger.info(f"Gradient success rate: {gradient_success_rate:.1%}")
        logger.info(f"Average MSE: {total_mse_improvement/max(success_count, 1):.4f}")
 
def main():
    parser = argparse.ArgumentParser(description="DIRECT-LATENTS adversarial generation")
    
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
    parser.add_argument("--epsilon", type=float, default=0.1, 
                       help="Maximum perturbation magnitude in latent space")
    parser.add_argument("--alpha", type=float, default=0.02, 
                       help="PGD step size in latent space")
    parser.add_argument("--num_iterations", type=int, default=50,
                       help="Number of PGD iterations")
    parser.add_argument("--lambda_reg", type=float, default=0.1,
                       help="Regularization coefficient")
    
    # 系统参数
    parser.add_argument("--output_dir", type=str, default="./direct_latents_adversarial_results",
                       help="Output directory")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Computing device")
    parser.add_argument("--save_frequency", type=int, default=50,
                       help="Save progress every N samples")
    parser.add_argument("--resume_from", type=int, default=None,
                       help="Resume from specific sample index")
    
    args = parser.parse_args()
    
    # 检查CUDA可用性
    if args.device == "cuda" and not torch.cuda.is_available():
        logger.error("CUDA requested but not available! Falling back to CPU.")
        args.device = "cpu"
    
    # 额外的内存优化设置
    if torch.cuda.is_available():
        torch.backends.cuda.max_split_size_mb = 512
        logger.info("Set CUDA max split size to 512MB")
    
    # 创建数据集
    logger.info(f"Loading dataset from {args.data_root}")
    dataset = LAIONFaceDataset(args.data_root, args.subset_size)
    
    # 使用直接latents攻击生成器
    logger.info("Initializing DIRECT-LATENTS adversarial generator")
    generator = DirectLatentsAdversarialGenerator(
        base_path=args.base_model,
        subject_lora_path=args.subject_lora,
        device=args.device,
        output_dir=args.output_dir
    )
    
    # 开始处理
    logger.info("Starting DIRECT-LATENTS adversarial generation")
    logger.info("ATTACK TARGET: main latents shape (1, 4096, 16)")
    logger.info("MSE comparison: denoised latents space")
    logger.info("METHOD: Direct delta perturbation on packed latents")
    logger.info(f"Parameters: epsilon={args.epsilon}, alpha={args.alpha}, iterations={args.num_iterations}")
    
    generator.process_dataset(
        dataset=dataset,
        epsilon=args.epsilon,
        alpha=args.alpha,
        num_iterations=args.num_iterations,
        lambda_reg=args.lambda_reg,
        save_frequency=args.save_frequency,
        resume_from=args.resume_from
    )
    
    logger.info("DIRECT-LATENTS generation completed!")
 
if __name__ == "__main__":
    main()
 
"""
直接在主要Latents上加扰动的对抗样本生成器

## 🎯 算法核心：
1. 记录clean_main_latents和clean_denoised_latents  
2. 在iteration中直接对main_latents加可学习的delta扰动
3. 在denoised_latents比较MSE
4. 梯度更新delta噪声

## 核心优势：
- ✅ 直接在packed latents (1, 4096, 16)上加扰动
- ✅ 完全按照pipeline.py的prepare_latents逻辑
- ✅ 稳定的梯度流（不依赖随机种子）
- ✅ 精确控制扰动幅度（L∞约束）
- ✅ Subject条件保持固定不变

## 使用命令：
python direct_latents_adversarial_generator.py \
    --data_root /openbayes/input/input0/sample_faces \
    --base_model /openbayes/input/input0 \
    --subject_lora /openbayes/input/input0/subject.safetensors \
    --epsilon 0.1 \
    --alpha 0.02 \
    --num_iterations 50 \
    --subset_size 1 \
    --device cuda

预期：更稳定的梯度，更直接的攻击效果
"""