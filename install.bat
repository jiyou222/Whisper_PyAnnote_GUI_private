@echo off
chcp 65001 >nul
echo 建立虛擬環境

:: 檢查是否已經存在虛擬環境
if exist "whisvenv" (
    echo 虛擬環境已經存在
) else (
    echo 正在建立虛擬環境
    python -m venv whisvenv
)

:: 啟動虛擬環境
echo 啟動虛擬環境
call whisvenv\Scripts\activate

:: 安裝所需的 Python 依賴
echo 安裝python庫
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install pyannote.audio
pip install openai-whisper
pip install transformers
pip install pyqt5

:: 安裝完成後的提示信息
echo 安裝項目已經完成，環境已經建置完畢
pause
