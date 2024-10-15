@echo off
chcp 65001 >nul
call whisvenv\Scripts\activate
echo "啟動虛擬環境"
python scriptGUI.py
echo "scriptGUI.py finished execution"
pause