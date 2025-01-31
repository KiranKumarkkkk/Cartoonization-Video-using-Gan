
import torch

# Check if CUDA is available
cuda_available = torch.cuda.is_available()

# Get the name of the GPU
gpu_name = torch.cuda.get_device_name(0) if cuda_available else None

print(f"CUDA Available: {cuda_available}")
if cuda_available:
    print(f"GPU: {gpu_name}")
