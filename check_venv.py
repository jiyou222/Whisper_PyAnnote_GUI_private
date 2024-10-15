import torch

print("檢查虛擬環境中的 CUDA 可用性：")
print(f"PyTorch 版本: {torch.__version__}")

if torch.cuda.is_available():
    print("CUDA 可用。")
    print(f"可用 GPU 數量: {torch.cuda.device_count()}")
    print(f"GPU 型號: {torch.cuda.get_device_name(0)}")
else:
    print("CUDA 不可用，將使用 CPU。")
