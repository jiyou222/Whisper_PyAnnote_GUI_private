import os
import subprocess
import torch

# 檢查當前環境的 CUDA 可用性
def check_cuda_environment():
    try:
        print("檢查全局環境中的 CUDA 可用性：")
        print(f"PyTorch 版本: {torch.__version__}")

        if torch.cuda.is_available():
            print("CUDA 可用。")
            print(f"可用 GPU 數量: {torch.cuda.device_count()}")
            print(f"GPU 型號: {torch.cuda.get_device_name(0)}")
        else:
            print("CUDA 不可用，將使用 CPU。")
    except Exception as e:
        print(f"全局環境檢查失敗，錯誤訊息：{e}")
        print("跳過全局環境檢查，直接進入虛擬環境檢查。\n")

# 檢查虛擬環境的 CUDA 可用性
def check_virtual_env_cuda():
    try:
        print("檢查虛擬環境中的 CUDA 可用性：")

        # 激活虛擬環境並運行檢查腳本
        activate_virtual_env = 'whisvenv\\Scripts\\activate'
        check_cuda_command = 'python check_venv.py'

        if os.name == 'nt':
            subprocess.run(f'start cmd /K "{activate_virtual_env} && {check_cuda_command}"', shell=True)
        else:
            print("這個功能只適用於 Windows 系統。請手動激活虛擬環境並運行檢查命令。")
    except Exception as e:
        print(f"虛擬環境檢查失敗，錯誤訊息：{e}")

# 先檢查全局環境，如果出錯則跳過
check_cuda_environment()

# 檢查虛擬環境
check_virtual_env_cuda()

# 暫停程序，等待用戶操作
input("按任意鍵結束...")
