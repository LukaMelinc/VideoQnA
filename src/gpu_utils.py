try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

def check_gpu_availability():
    """Check GPU availability and print system info."""
    if not TORCH_AVAILABLE:
        print("PyTorch is not available. GPU functionality disabled.")
        return False
        
    if torch.cuda.is_available():
        print(f"GPU is available!")
        print(f"GPU device count: {torch.cuda.device_count()}")
        print(f"Current GPU device: {torch.cuda.current_device()}")
        print(f"GPU device name: {torch.cuda.get_device_name()}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        return True
    else:
        print("GPU is not available. Using CPU.")
        return False

def get_gpu_memory_usage():
    """Get current GPU memory usage."""
    if not TORCH_AVAILABLE:
        return "PyTorch not available"
        
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        cached = torch.cuda.memory_reserved() / 1024**3
        return f"GPU Memory - Allocated: {allocated:.2f} GB, Cached: {cached:.2f} GB"
    return "GPU not available"

def clear_gpu_cache():
    """Clear GPU cache to free up memory."""
    if not TORCH_AVAILABLE:
        print("PyTorch not available")
        return
        
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("GPU cache cleared")
    else:
        print("GPU not available")