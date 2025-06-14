import os
import torch
from pathlib import Path
from huggingface_hub import snapshot_download, try_to_load_from_cache
from src.pipeline import FluxPipeline
from src.transformer_flux import FluxTransformer2DModel

def check_local_model_exists(model_path_or_id):
    """
    检查模型是否在本地存在
    
    Args:
        model_path_or_id: 本地路径或HuggingFace模型ID
    
    Returns:
        tuple: (is_local, actual_path)
    """
    # 检查是否是本地路径
    if os.path.exists(model_path_or_id):
        # 检查是否包含必要的模型文件
        required_files = [
            "model_index.json",
            "scheduler/scheduler_config.json", 
            "text_encoder/config.json",
            "text_encoder_2/config.json",
            "transformer/config.json",
            "vae/config.json"
        ]
        
        all_exist = all(os.path.exists(os.path.join(model_path_or_id, file)) 
                       for file in required_files)
        
        if all_exist:
            print(f"✅ 找到完整的本地模型: {model_path_or_id}")
            return True, model_path_or_id
        else:
            print(f"❌ 本地路径存在但模型文件不完整: {model_path_or_id}")
            return False, None
    
    # 检查HuggingFace缓存
    try:
        # 尝试从缓存加载
        cached_path = try_to_load_from_cache(
            repo_id=model_path_or_id,
            filename="model_index.json"
        )
        
        if cached_path is not None:
            # 获取缓存的根目录
            cache_dir = str(Path(cached_path).parent)
            print(f"✅ 找到HuggingFace缓存模型: {cache_dir}")
            return True, cache_dir
        else:
            print(f"❌ 模型不在本地缓存中: {model_path_or_id}")
            return False, None
            
    except Exception as e:
        print(f"❌ 检查缓存时出错: {e}")
        return False, None

def load_model_local_only(model_path_or_id, device="cuda", torch_dtype=torch.bfloat16):
    """
    仅从本地加载模型，不进行自动下载
    
    Args:
        model_path_or_id: 模型路径或ID
        device: 设备
        torch_dtype: 数据类型
    
    Returns:
        tuple: (pipeline, transformer) 或 (None, None)
    """
    print(f"🔍 检查模型是否在本地: {model_path_or_id}")
    
    # 检查本地是否存在
    is_local, actual_path = check_local_model_exists(model_path_or_id)
    
    if not is_local:
        print(f"❌ 模型不在本地，请先手动下载模型!")
        print(f"💡 下载提示:")
        print(f"   方法1: 使用 huggingface-cli download {model_path_or_id}")
        print(f"   方法2: 使用以下Python代码下载:")
        print(f"   from huggingface_hub import snapshot_download")
        print(f"   snapshot_download(repo_id='{model_path_or_id}', local_dir='./models/{model_path_or_id.replace('/', '-')}')")
        return None, None
    
    try:
        print(f"📂 从本地加载模型: {actual_path}")
        
        # 设置本地模式，防止意外下载
        os.environ['HF_HUB_OFFLINE'] = '1'
        
        # 加载pipeline
        print("🔄 加载 FluxPipeline...")
        pipe = FluxPipeline.from_pretrained(
            actual_path, 
            torch_dtype=torch_dtype, 
            device=device,
            local_files_only=True  # 强制仅使用本地文件
        )
        
        # 加载transformer
        print("🔄 加载 FluxTransformer2DModel...")
        transformer = FluxTransformer2DModel.from_pretrained(
            actual_path,
            subfolder="transformer",
            torch_dtype=torch_dtype,
            device=device,
            local_files_only=True  # 强制仅使用本地文件
        )
        
        pipe.transformer = transformer
        pipe.to(device)
        
        # 恢复在线模式
        if 'HF_HUB_OFFLINE' in os.environ:
            del os.environ['HF_HUB_OFFLINE']
        
        print("✅ 模型加载成功!")
        return pipe, transformer
        
    except Exception as e:
        # 恢复在线模式
        if 'HF_HUB_OFFLINE' in os.environ:
            del os.environ['HF_HUB_OFFLINE']
        
        print(f"❌ 加载模型时出错: {e}")
        return None, None

def download_model_if_needed(model_id, local_dir=None):
    """
    手动下载模型（可选功能）
    
    Args:
        model_id: HuggingFace模型ID
        local_dir: 本地存储目录
    """
    if local_dir is None:
        local_dir = f"./models/{model_id.replace('/', '-')}"
    
    print(f"📥 开始下载模型到: {local_dir}")
    
    try:
        snapshot_download(
            repo_id=model_id,
            local_dir=local_dir,
            local_dir_use_symlinks=False
        )
        print(f"✅ 模型下载完成: {local_dir}")
        return local_dir
    except Exception as e:
        print(f"❌ 下载失败: {e}")
        return None

# 使用示例
if __name__ == "__main__":
    # 你的模型路径或ID
    model_paths = [
        "./models/FLUX.1-dev",  # 本地路径
        "black-forest-labs/FLUX.1-dev",  # HF模型ID
        "/path/to/your/local/model"  # 另一个本地路径
    ]
    
    for model_path in model_paths:
        print(f"\n{'='*50}")
        print(f"测试路径: {model_path}")
        print(f"{'='*50}")
        
        # 尝试加载
        pipe, transformer = load_model_local_only(model_path)
        
        if pipe is not None and transformer is not None:
            print("🎉 可以开始使用模型了!")
            # 这里可以继续你的代码...
            break
        else:
            print("⏳ 尝试下一个路径...")
    
    # 如果都没找到，提供下载选项
    if pipe is None:
        print(f"\n❓ 是否要下载模型? (y/n)")
        choice = input().lower().strip()
        if choice == 'y':
            model_id = "black-forest-labs/FLUX.1-dev"
            downloaded_path = download_model_if_needed(model_id)
            if downloaded_path:
                pipe, transformer = load_model_local_only(downloaded_path)