import os
import sys
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
from typing import Tuple, Dict
from torchvision.transforms.functional import pad

# 导入路径
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, script_dir)
sys.path.insert(0, parent_dir)

from src.pipeline import FluxPipeline
from src.transformer_flux import FluxTransformer2DModel
from src.lora_helper import set_single_lora

# 设置环境变量
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["PYTORCH_NO_CUDA_MEMORY_CACHING"] = "1"

# 辅助函数
def calculate_shift(image_seq_len, base_seq_len=256, max_seq_len=4096, base_shift=0.5, max_shift=1.16):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu

def retrieve_timesteps(scheduler, num_inference_steps, device, timesteps=None, sigmas=None, **kwargs):
    import inspect
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom timestep schedules")
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom sigmas schedules")
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps

def retrieve_latents(encoder_output, generator=None, sample_mode="sample"):
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        return encoder_output.latent_dist.sample(generator)
    elif hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        return encoder_output.latent_dist.mode()
    elif hasattr(encoder_output, "latents"):
        return encoder_output.latents
    else:
        raise AttributeError("Could not access latents of provided encoder_output")

class ImageSpaceAdversarialGenerator:
    """图像空间对抗攻击生成器 - 严格按照算法要求实现"""
    
    def __init__(self, base_path: str, subject_lora_path: str, device: str = "cuda"):
        self.device = torch.device(device)
        self.attack_prompt = "A SKS on the beach"
        self.cond_size = 512
        self._init_pipeline(base_path, subject_lora_path)
        
    def _init_pipeline(self, base_path: str, subject_lora_path: str):
        """✅ 要求: VAE和transformer都切换到eval()"""
        self.pipe = FluxPipeline.from_pretrained(base_path, torch_dtype=torch.bfloat16, local_files_only=True)
        
        transformer = FluxTransformer2DModel.from_pretrained(
            base_path, subfolder="transformer", torch_dtype=torch.bfloat16, local_files_only=True
        )
        self.pipe.transformer = transformer
        self.pipe.to(self.device)
        
        set_single_lora(self.pipe.transformer, subject_lora_path, lora_weights=[1], cond_size=512)
        def disable_dropout_recursive(module):
            for name, child in module.named_modules():
                if isinstance(child, torch.nn.Dropout):
                    child.p = 0.0  # 强制设置dropout概率为0
                    print(f"Disabled dropout in: {name}")
        
        print("正在禁用transformer中的dropout...")
        disable_dropout_recursive(self.pipe.transformer)
        # ✅ 要求: VAE和transformer都切换到eval()，但保持梯度传播
        self.pipe.vae.eval()
        self.pipe.transformer.eval()
        self.pipe.transformer.gradient_checkpointing = True
        
        for param in self.pipe.vae.parameters():
            param.requires_grad = False
        for param in self.pipe.transformer.parameters():
            param.requires_grad = False
    
    def pil_to_tensor_512(self, image: Image.Image) -> torch.Tensor:
        """
        ✅ 修正: 按照原始pipeline预处理方式，保持宽高比然后padding
        PIL Image -> [512, 512, 3] tensor
        """
        w, h = image.size
        scale = self.cond_size / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        
        # 使用pipeline的image_processor预处理
        tensor = self.pipe.image_processor.preprocess(image, height=new_h, width=new_w)
        tensor = tensor.to(dtype=torch.bfloat16)
        
        # Padding到cond_size
        pad_h = self.cond_size - tensor.shape[-2]
        pad_w = self.cond_size - tensor.shape[-1]
        tensor = pad(
            tensor,
            padding=(int(pad_w / 2), int(pad_h / 2), int(pad_w / 2), int(pad_h / 2)),
            fill=0
        )
        
        # 转换为[512, 512, 3]格式
        tensor = tensor.squeeze(0).permute(1, 2, 0).to(self.device)
        return tensor
    
    def tensor_512_to_pil(self, tensor: torch.Tensor) -> Image.Image:
        """[512, 512, 3] tensor -> PIL Image"""
        tensor = torch.clamp(tensor, 0, 1)
        image_array = (tensor.cpu().detach().numpy() * 255).astype(np.uint8)
        return Image.fromarray(image_array)
    
    def preprocess_subject(self, image_tensor_512: torch.Tensor) -> torch.Tensor:
        """[512,512,3] -> [1,3,512,512] pipeline格式"""
        tensor = image_tensor_512.permute(2, 0, 1).unsqueeze(0).to(dtype=torch.bfloat16)
        return tensor
    
    def encode_subject_with_vae(self, subject_tensor: torch.Tensor, enable_grad: bool = False) -> torch.Tensor:
        """VAE编码: [1,3,512,512] -> [1,1024,64]"""
        subject_tensor = subject_tensor.to(dtype=self.pipe.vae.dtype, device=self.device)
        generator = torch.Generator(self.device).manual_seed(42)
        
        if enable_grad:
            image_latents = retrieve_latents(self.pipe.vae.encode(subject_tensor), generator=generator)
        else:
            with torch.no_grad():
                image_latents = retrieve_latents(self.pipe.vae.encode(subject_tensor), generator=generator)

        image_latents = (image_latents - self.pipe.vae.config.shift_factor) * self.pipe.vae.config.scaling_factor
        packed_latents = self.pipe._pack_latents(image_latents, 1, 16, 64, 64)
        return packed_latents
    
    def generate_main_latents(self, height: int = 1024, width: int = 1024, enable_grad: bool = False) -> torch.Tensor:
        """
        ✅ 要求: 生成随机噪声 [1,16,128,128] 需要梯度
        ✅ 要求: 打包主图 [1,4096,64] 需要梯度
        """
        height_main = 2 * (int(height) // self.pipe.vae_scale_factor)  # 128
        width_main = 2 * (int(width) // self.pipe.vae_scale_factor)   # 128
        
        shape = (1, 16, height_main, width_main)
        generator = torch.Generator(str(self.device)).manual_seed(42)  # 固定种子确保相同主图
        
        if enable_grad:
            noise_latents = torch.randn(shape, generator=generator, device=self.device, 
                                      dtype=torch.bfloat16, requires_grad=True)
        else:
            with torch.no_grad():
                noise_latents = torch.randn(shape, generator=generator, device=self.device, dtype=torch.bfloat16)
        
        main_latents = self.pipe._pack_latents(noise_latents, 1, 16, height_main, width_main)
        return main_latents
    
    def prepare_pipeline_components(self, height: int = 1024, width: int = 1024) -> Tuple:
        """准备pipeline组件"""
        with torch.no_grad():
            prompt_embeds, pooled_prompt_embeds, text_ids = self.pipe.encode_prompt(
                prompt=self.attack_prompt, prompt_2=self.attack_prompt,
                device=str(self.device), num_images_per_prompt=1, max_sequence_length=512,
            )
            
            # 构建latent_image_ids
            from src.pipeline import prepare_latent_subject_ids, resize_position_encoding
            
            height_cond = 2 * (self.cond_size // self.pipe.vae_scale_factor)  # 64
            width_cond = 2 * (self.cond_size // self.pipe.vae_scale_factor)   # 64
            height_main = 2 * (int(height) // self.pipe.vae_scale_factor)     # 128
            width_main = 2 * (int(width) // self.pipe.vae_scale_factor)       # 128
            
            noise_latent_image_ids, _ = resize_position_encoding(
                1, height_main, width_main, height_cond, width_cond, self.device, torch.bfloat16
            )
            
            latent_subject_ids = prepare_latent_subject_ids(height_cond, width_cond, self.device, torch.bfloat16)
            latent_subject_ids[:, 1] += 64  # 固定偏移
            
            latent_image_ids = torch.cat([noise_latent_image_ids, latent_subject_ids], dim=0)
            
            return prompt_embeds, pooled_prompt_embeds, text_ids, latent_image_ids
    
    def denoise_latents(self, main_latents: torch.Tensor, subject_packed: torch.Tensor,
                       prompt_embeds: torch.Tensor, pooled_prompt_embeds: torch.Tensor,
                       text_ids: torch.Tensor, latent_image_ids: torch.Tensor,
                       enable_grad: bool = False) -> torch.Tensor:
        """完整去噪过程"""
        device = self.device
        num_inference_steps = 5
        guidance_scale = 3.5
        
        latents = main_latents.clone()
        
        # 准备timesteps
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
        image_seq_len = latents.shape[1]
        mu = calculate_shift(image_seq_len, self.pipe.scheduler.config.base_image_seq_len,
                           self.pipe.scheduler.config.max_image_seq_len,
                           self.pipe.scheduler.config.base_shift, self.pipe.scheduler.config.max_shift)
        timesteps, num_inference_steps = retrieve_timesteps(
            self.pipe.scheduler, num_inference_steps, str(device), None, sigmas, mu=mu
        )
        
        # guidance设置
        if self.pipe.transformer.config.guidance_embeds:
            guidance = torch.full([1], guidance_scale, device=str(device), dtype=torch.bfloat16)
            guidance = guidance.expand(latents.shape[0])
        else:
            guidance = None
        
        # 清除cache
        for name, attn_processor in self.pipe.transformer.attn_processors.items():
            if hasattr(attn_processor, 'bank_kv'):
                attn_processor.bank_kv.clear()
        
        # Warmup
        start_idx = latents.shape[1] - 32
        warmup_latents = latents[:, start_idx:, :]
        warmup_latent_ids = latent_image_ids[start_idx:, :]
        t = torch.tensor([timesteps[0]], device=str(device))
        timestep = t.expand(warmup_latents.shape[0]).to(latents.dtype)
        
        if enable_grad:
            _ = self.pipe.transformer(
                hidden_states=warmup_latents, cond_hidden_states=subject_packed,
                timestep=timestep / 1000, guidance=guidance,
                pooled_projections=pooled_prompt_embeds, encoder_hidden_states=prompt_embeds,
                txt_ids=text_ids, img_ids=warmup_latent_ids,
                joint_attention_kwargs=None, return_dict=False,
            )[0]
        else:
            with torch.no_grad():
                _ = self.pipe.transformer(
                    hidden_states=warmup_latents, cond_hidden_states=subject_packed,
                    timestep=timestep / 1000, guidance=guidance,
                    pooled_projections=pooled_prompt_embeds, encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids, img_ids=warmup_latent_ids,
                    joint_attention_kwargs=None, return_dict=False,
                )[0]
        
        # 去噪循环
        context = torch.no_grad() if not enable_grad else torch.enable_grad()
        with context:
            for i, t in enumerate(timesteps):
                timestep = t.expand(latents.shape[0]).to(latents.dtype)
                
                noise_pred = self.pipe.transformer(
                    hidden_states=latents, cond_hidden_states=subject_packed,
                    timestep=timestep / 1000, guidance=guidance,
                    pooled_projections=pooled_prompt_embeds, encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids, img_ids=latent_image_ids,
                    joint_attention_kwargs=None, return_dict=False,
                )[0]
                
                latents = self.pipe.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

        return latents
    
    def compute_clean_path(self, subject_image: Image.Image, height: int = 1024, width: int = 1024) -> Tuple:
        """
        ✅ 要求1: 计算clean_image [512,512,3]一直到denoised_latent [1,4096,64]
        ✅ 要求1: 记住clean_image生成的以及他的主图[1,4096,64]和最终的final latents [1,4096,64]
        ✅ 要求1: 到目前为止都不要开启梯度
        """
        with torch.no_grad():
            # 准备组件
            prompt_embeds, pooled_prompt_embeds, text_ids, latent_image_ids = self.prepare_pipeline_components(height, width)
            
            # 处理subject图像: clean_image [512,512,3]
            clean_image_tensor = self.pil_to_tensor_512(subject_image)
            clean_subject_pipeline = self.preprocess_subject(clean_image_tensor)
            clean_subject_packed = self.encode_subject_with_vae(clean_subject_pipeline, enable_grad=False)
            
            # 生成主图: [1,4096,64]
            clean_main_latents = self.generate_main_latents(height, width, enable_grad=False)
            
            # 去噪生成clean final latents: [1,4096,64]
            clean_final_latents = self.denoise_latents(
                clean_main_latents, clean_subject_packed,
                prompt_embeds, pooled_prompt_embeds, text_ids, latent_image_ids,
                enable_grad=False
            )
            
            pipeline_components = (prompt_embeds, pooled_prompt_embeds, text_ids, latent_image_ids)
            
            return clean_final_latents, clean_main_latents, clean_subject_packed, pipeline_components, clean_image_tensor
    
    def pgd_attack(self, original_image: Image.Image, epsilon: float = 0.03, alpha: float = 0.01,
                  num_iterations: int = 50, lambda_reg: float = 0.1) -> Tuple[torch.Tensor, Dict]:
        """
        ✅ 要求2: 开启epoch，第一个epoch在clean_image [512,512,3]的基础上生成噪声，提前计算好无穷范数，然后保持梯度
        ✅ 要求3: 使用相同的主图，但是在这里将主图开启梯度，因为防止更新delta梯度断裂
        ✅ 要求4: 计算MSE，-MSE+无穷范数作为最终loss，然后反向传播，用PGD来更新delta
        """
        
        # Phase 1: 计算clean路径（无梯度）
        clean_final_latents, clean_main_latents, clean_subject_packed, pipeline_components, clean_image_tensor = self.compute_clean_path(original_image)
        prompt_embeds, pooled_prompt_embeds, text_ids, latent_image_ids = pipeline_components
        
        # Phase 2: 初始化delta在图像空间 [512,512,3]
        delta = torch.zeros_like(clean_image_tensor, requires_grad=True, device=self.device, dtype=torch.bfloat16)
        
        with torch.no_grad():
            delta.data = (torch.randn_like(clean_image_tensor) * epsilon * 0.1).to(device=self.device, dtype=torch.bfloat16)
        
        attack_info = {'loss_history': [], 'mse_history': []}
        consecutive_failures = 0
        max_consecutive_failures = 5
        
        for i in range(num_iterations):
            delta.requires_grad_(True)
            
            # ✅ 要求2: 第一个epoch在clean_image [512,512,3]的基础上生成噪声，提前在这里计算好无穷范数
            noisy_image_tensor = clean_image_tensor + delta
            noisy_image_tensor = torch.clamp(noisy_image_tensor, 0, 1)
            
            # 提前计算无穷范数
            linf_norm = torch.norm(delta, p=float('inf'))
            
            # 处理加噪subject并保持梯度
            noisy_subject_pipeline = self.preprocess_subject(noisy_image_tensor)
            noisy_subject_packed = self.encode_subject_with_vae(noisy_subject_pipeline, enable_grad=True)
            
            # ✅ 要求3: 使用相同的主图，但是在这里将主图开启梯度（修正：移除detach）
            main_latents_with_grad = clean_main_latents.clone().requires_grad_(True)
            
            # 去噪生成adversarial final latents
            adversarial_final_latents = self.denoise_latents(
                main_latents_with_grad, noisy_subject_packed,
                prompt_embeds, pooled_prompt_embeds, text_ids, latent_image_ids,
                enable_grad=True
            )
            
            # ✅ 要求4: 计算MSE，-MSE+无穷范数作为最终loss
            mse_loss = F.mse_loss(clean_final_latents, adversarial_final_latents)
            
            if not mse_loss.requires_grad:
                consecutive_failures += 1
                if consecutive_failures >= max_consecutive_failures:
                    break
                torch.cuda.empty_cache()
                continue
            
            consecutive_failures = 0
            
            # 总损失
            total_loss = -mse_loss + lambda_reg * linf_norm
            
            attack_info['loss_history'].append(total_loss.item())
            attack_info['mse_history'].append(mse_loss.item())
            

            print(f"Iter {i+1}: MSE={mse_loss.item():.6f}, L_inf={linf_norm.item():.6f}, Total_loss={total_loss.item():.6f}")
            
            # ✅ 要求4: 反向传播，用PGD来更新delta
            total_loss.backward()
            
            if delta.grad is None:
                break
            
            # PGD更新
            with torch.no_grad():
                delta.data = delta.data + alpha * delta.grad.sign()
                delta.data = torch.clamp(delta.data, -epsilon, epsilon)
                
                # 确保图像有效性
                noisy_check = clean_image_tensor + delta.data
                if torch.any(noisy_check < 0) or torch.any(noisy_check > 1):
                    delta.data = torch.clamp(noisy_check, 0, 1) - clean_image_tensor
                    delta.data = torch.clamp(delta.data, -epsilon, epsilon)
            
            delta.grad = None
            
            # 早期停止
            if i > 10 and len(attack_info['mse_history']) > 10:
                recent_mse = attack_info['mse_history'][-10:]
                if max(recent_mse) - min(recent_mse) < 1e-6:
                    break
            
            if i % 5 == 0:
                torch.cuda.empty_cache()
        
        return delta.detach(), attack_info
    
    def generate_final_results(self, delta: torch.Tensor,
                             clean_final_latents: torch.Tensor, clean_main_latents: torch.Tensor,
                             pipeline_components: Tuple, clean_image_tensor: torch.Tensor) -> Tuple[Image.Image, Image.Image, Image.Image]:
        """
        ✅ 要求5: 两个final_latents一个clean一个加噪，通过unpack和VAE decode生成出两个不一样的图片结果
        ✅ 要求5: 用最终的delta得到最初的加噪图片
        """
        
        prompt_embeds, pooled_prompt_embeds, text_ids, latent_image_ids = pipeline_components
        
        # 计算adversarial路径
        final_noisy_image_tensor = clean_image_tensor + delta
        final_noisy_image_tensor = torch.clamp(final_noisy_image_tensor, 0, 1)
        
        noisy_subject_pipeline = self.preprocess_subject(final_noisy_image_tensor)
        
        with torch.no_grad():
            noisy_subject_packed = self.encode_subject_with_vae(noisy_subject_pipeline, enable_grad=False)
            
            # 使用传入的clean_main_latents避免重复计算
            adversarial_final_latents = self.denoise_latents(
                clean_main_latents, noisy_subject_packed,
                prompt_embeds, pooled_prompt_embeds, text_ids, latent_image_ids,
                enable_grad=False
            )
            
            # ✅ 要求5: 通过unpack和VAE decode生成两个不一样的图片结果
            # Clean结果
            clean_unpacked = self.pipe._unpack_latents(clean_final_latents, 1024, 1024, self.pipe.vae_scale_factor)
            clean_decoded = self.decode_latents_to_tensor(clean_unpacked)
            clean_generated_image = self.tensor_to_pil_official(clean_decoded)
            
            # Adversarial结果
            adversarial_unpacked = self.pipe._unpack_latents(adversarial_final_latents, 1024, 1024, self.pipe.vae_scale_factor)
            adversarial_decoded = self.decode_latents_to_tensor(adversarial_unpacked)
            adversarial_generated_image = self.tensor_to_pil_official(adversarial_decoded)
            
            # ✅ 要求5: 用最终的delta得到最初的加噪图片
            noisy_original_image = self.tensor_512_to_pil(final_noisy_image_tensor)
        
        return clean_generated_image, adversarial_generated_image, noisy_original_image
    
    def decode_latents_to_tensor(self, latents: torch.Tensor) -> torch.Tensor:
        """解码latents"""
        with torch.no_grad():
            latents = (latents / self.pipe.vae.config.scaling_factor) + self.pipe.vae.config.shift_factor
            latents = latents.to(dtype=self.pipe.vae.dtype, device=self.device)
            decoded_tensor = self.pipe.vae.decode(latents, return_dict=False)[0]
            return decoded_tensor
    
    def tensor_to_pil_official(self, tensor: torch.Tensor) -> Image.Image:
        """tensor转PIL（官方方式）"""
        if len(tensor.shape) == 4:
            tensor = tensor.squeeze(0)
        tensor = tensor.unsqueeze(0)
        image = self.pipe.image_processor.postprocess(tensor, output_type="pil")[0]
        return image

def process_single_image(generator: ImageSpaceAdversarialGenerator, 
                        image: Image.Image,
                        epsilon: float = 0.03,
                        alpha: float = 0.01,
                        num_iterations: int = 50,
                        lambda_reg: float = 0.1) -> Dict:
    """处理单张图像（完全按照算法要求）"""
    
    try:
        print("Starting image space adversarial attack...")
        
        # 执行PGD攻击
        delta, attack_info = generator.pgd_attack(
            original_image=image,
            epsilon=epsilon,
            alpha=alpha,
            num_iterations=num_iterations,
            lambda_reg=lambda_reg
        )
        
        # 重新计算clean路径（只计算一次用于最终结果）
        clean_final_latents, clean_main_latents, clean_subject_packed, pipeline_components, clean_image_tensor = generator.compute_clean_path(image)
        
        # 生成最终三种结果
        clean_generated, adversarial_generated, noisy_original = generator.generate_final_results(
            delta, clean_final_latents, clean_main_latents, pipeline_components, clean_image_tensor
        )
        
        # 计算最终指标
        with torch.no_grad():
            final_noisy_tensor = generator.pil_to_tensor_512(image) + delta
            final_noisy_tensor = torch.clamp(final_noisy_tensor, 0, 1)
            noisy_pipeline = generator.preprocess_subject(final_noisy_tensor)
            noisy_packed_final = generator.encode_subject_with_vae(noisy_pipeline, enable_grad=False)
            
            prompt_embeds, pooled_prompt_embeds, text_ids, latent_image_ids = pipeline_components
            adversarial_final_latents = generator.denoise_latents(
                clean_main_latents, noisy_packed_final,
                prompt_embeds, pooled_prompt_embeds, text_ids, latent_image_ids,
                enable_grad=False
            )
            final_mse = F.mse_loss(clean_final_latents, adversarial_final_latents).item()
        
        final_linf_norm = torch.norm(delta, p=float('inf')).item()
        
        result = {
            'final_mse': final_mse,
            'final_linf_norm': final_linf_norm,
            'constraint_satisfied': final_linf_norm <= epsilon,
            'attack_successful': final_mse > 0.01,
            'images': {
                'clean_generated': clean_generated,
                'adversarial_generated': adversarial_generated,
                'noisy_original': noisy_original
            }
        }
        
        print(f"Attack completed: MSE={final_mse:.6f}, L∞={final_linf_norm:.6f}")
        return result
        
    except Exception as e:
        print(f"Attack failed: {e}")
        return {'error': str(e)}

# 使用示例
def main():
    """主函数示例"""
    
    # 创建测试图像
    test_image = Image.new('RGB', (512, 512), color=(128, 128, 128))
    
    # 初始化生成器
    generator = ImageSpaceAdversarialGenerator(
        base_path="/openbayes/input/input0",
        subject_lora_path="/openbayes/input/input0/subject.safetensors",
        device="cuda"
    )
    
    # 处理图像
    result = process_single_image(
        generator=generator,
        image=test_image,
        epsilon=0.03,
        alpha=0.01,
        num_iterations=5,
        lambda_reg=0.1
    )
    
    if 'error' not in result:
        print("Attack successful!")
        # 保存结果
        result['images']['clean_generated'].save("clean_result.png")
        result['images']['adversarial_generated'].save("adversarial_result.png")
        result['images']['noisy_original'].save("noisy_original.png")
    else:
        print(f"Attack failed: {result['error']}")

if __name__ == "__main__":
    main()