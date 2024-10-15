import sys
import os
import subprocess
import torch
from pyannote.audio import Pipeline
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QLabel, QLineEdit, QFileDialog, QComboBox, QProgressBar, QMessageBox, QHBoxLayout, QTextEdit, QListWidget, QCheckBox
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from datetime import timedelta, datetime
import whisper  # 確保使用 openai-whisper 套件

# 讀取 token 從 Token.txt
def load_token():
    token_path = os.path.join(os.path.dirname(__file__), 'Token.txt')
    with open(token_path, 'r') as token_file:
        token = token_file.read().strip()
    return token

# 檢查設備並回傳顯示的資訊
def detect_device():
    if torch.cuda.is_available():
        return torch.device("cuda")  # 返回 torch.device 對象
    else:
        return torch.device("cpu")  # 返回 torch.device 對象

# Whisper 和 PyAnnote 執行緒類，處理多個檔案的批量轉換
class TranscriptionThread(QThread):
    progress_signal = pyqtSignal(int, str)
    status_signal = pyqtSignal(str)  # 用於更新狀態訊號
    device_signal = pyqtSignal(str)  # 用於更新設備訊號
    
    def __init__(self, audio_files, model_size, language, output_dir, run_pyannote):
        super().__init__()
        self.audio_files = audio_files
        self.model_size = model_size
        self.language = language
        self.output_dir = output_dir
        self.run_pyannote = run_pyannote  # 新增參數，用於判斷是否執行 PyAnnote

    def run(self):
        try:
            # 檢查設備並發送設備訊息
            device = detect_device()
            self.device_signal.emit(f"使用設備: {device}")

            total_files = len(self.audio_files)

            # 檢查模型並下載
            self.status_signal.emit("檢查模型是否已安裝...")
            model_downloaded = check_and_download_model(self.model_size, self.progress_signal, self.status_signal)

            for idx, audio_file in enumerate(self.audio_files):
                # 每個檔案的處理進度顯示
                self.status_signal.emit(f"開始處理檔案 {idx + 1}/{total_files}: {os.path.basename(audio_file)}")
                progress = int((idx / total_files) * 100)
                self.progress_signal.emit(progress, f"正在處理 {idx + 1}/{total_files} 檔案...")

                # 開始Whisper轉錄
                start_time = datetime.now().strftime("%H:%M:%S")
                self.status_signal.emit(f"{start_time} 開始Whisper轉錄，使用設備: {device}")
                transcription = run_whisper(audio_file, self.model_size, self.language, device, self.output_dir)

                # Whisper轉錄完成，顯示時間戳
                whisper_end_time = datetime.now().strftime("%H:%M:%S")
                self.status_signal.emit(f"{whisper_end_time} Whisper轉錄完成")

                # 判斷是否執行 PyAnnote 講者分離
                if self.run_pyannote:
                    # 開始PyAnnote講者分離
                    start_time = datetime.now().strftime("%H:%M:%S")
                    self.status_signal.emit(f"{start_time} 開始PyAnnote講者分離，使用設備: {device}")
                    run_speaker_diarization(audio_file, self.output_dir, transcription)

                    # PyAnnote講者分離完成
                    diarization_end_time = datetime.now().strftime("%H:%M:%S")
                    self.status_signal.emit(f"{diarization_end_time} 講者分離完成")
                
                # 更新進度條
                progress = int(((idx + 1) / total_files) * 100)
                self.progress_signal.emit(progress, f"已處理 {idx + 1}/{total_files} 檔案")

            # 最後顯示完成時間戳
            end_time = datetime.now().strftime("%H:%M:%S")
            self.progress_signal.emit(100, "所有檔案已處理完成！")
            self.status_signal.emit(f"{end_time} 完成所有批量轉換！")
        except Exception as e:
            error_time = datetime.now().strftime("%H:%M:%S")
            self.status_signal.emit(f"{error_time} 錯誤：{str(e)}")
            self.progress_signal.emit(0, f"錯誤：{str(e)}")

# 檢查並下載模型
def check_and_download_model(model_size, progress_signal, status_signal):
    model_dir = os.path.join(os.getcwd(), 'models')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)  # 如果沒有 models 資料夾，則創建

    # 嘗試加載模型，如果模型不存在則自動下載
    try:
        model_path = os.path.join(model_dir, model_size)
        if not os.path.exists(model_path):
            # 模型下載開始，更新狀態和進度條
            status_signal.emit(f"模型 {model_size} 不存在，開始下載...")
            download_start_time = datetime.now().strftime("%H:%M:%S")
            status_signal.emit(f"{download_start_time} 模型下載開始")

            model = whisper.load_model(model_size, download_root=model_dir)

            download_end_time = datetime.now().strftime("%H:%M:%S")
            status_signal.emit(f"{download_end_time} 模型 {model_size} 已成功下載")
            progress_signal.emit(20, "模型下載完成")
            return True
        else:
            model = whisper.load_model(model_size, download_root=model_dir)
            status_signal.emit(f"模型 {model_size} 已加載。")
            return False
    except Exception as e:
        status_signal.emit(f"下載或加載模型 {model_size} 時發生錯誤：{str(e)}")
        return False

# Whisper轉錄
def run_whisper(audio_file, model_size, language, device, output_dir):
    # 將 torch.device 對象轉換為字符串
    device_str = str(device) if isinstance(device, torch.device) else device

    base_command = [
        "whisper", audio_file,
        "--model", model_size,
        "--language", language,
        "--device", device_str,  # 使用設備名稱的字符串
        "--output_dir", output_dir
    ]
    
    process = subprocess.Popen(base_command, shell=False)
    process.wait()

    transcription_file = os.path.join(output_dir, os.path.basename(audio_file).replace('.mp3', '.txt').replace('.wav', '.txt').replace('.mp4', '.txt').replace('.m4a', '.txt').replace('.aac', '.txt'))
    
    with open(transcription_file, 'r', encoding='utf-8') as f:
        transcription = f.readlines()
    
    return transcription

# PyAnnote講者分離
def run_speaker_diarization(audio_file, output_dir, transcription):
    token = load_token()

    # 檢查音頻文件格式，並在需要時進行轉換
    if audio_file.endswith('.mp4'):
        converted_audio_file = os.path.splitext(audio_file)[0] + '_pyannote.wav'
        try:
            # 使用 FFmpeg 進行轉換
            subprocess.run(['ffmpeg', '-i', audio_file, converted_audio_file], check=True)
            audio_file = converted_audio_file  # 更新路徑為轉換後的 WAV 文件
        except Exception as e:
            print(f"轉換 MP4 時發生錯誤：{str(e)}")
            return []

    # 進行講者分離
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization", use_auth_token=token)
    device = detect_device()
    pipeline.to(device)
    diarization = pipeline(audio_file)

    # 以"原檔名+SpeakerDiarization.srt"作為結果名稱
    base_name = os.path.splitext(os.path.basename(audio_file))[0]
    diarization_path = os.path.join(output_dir, f'{base_name}_SpeakerDiarization.rttm')

    with open(diarization_path, 'w') as f:
        diarization.write_rttm(f)

    srt_output_path = os.path.join(output_dir, f'{base_name}_SpeakerDiarization.srt')
    with open(srt_output_path, 'w', encoding='utf-8') as srt_file:
        for i, (turn, _, speaker) in enumerate(diarization.itertracks(yield_label=True)):
            start_time = seconds_to_srt_time(turn.start)
            end_time = seconds_to_srt_time(turn.end)
            srt_file.write(f"{i+1}\n")
            srt_file.write(f"{start_time} --> {end_time}\n")
            srt_file.write(f"Speaker {speaker}\n")
            if i < len(transcription):
                srt_file.write(f"{transcription[i].strip()}\n")
            srt_file.write("\n")


# 將時間轉換為SRT格式
def seconds_to_srt_time(seconds):
    td = timedelta(seconds=seconds)
    return str(td)[:-3].replace('.', ',')

# GUI 界面
class TranscriptionApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        main_layout = QHBoxLayout()  # 使用水平佈局分配左側和右側區域

        # 左側部分：文件選擇和進度條
        left_layout = QVBoxLayout()

        self.file_label = QLabel("選擇音檔：")
        self.file_input = QListWidget(self)  # 支援選擇多個檔案
        self.browse_button = QPushButton('瀏覽', self)
        self.browse_button.clicked.connect(self.browse_files)

        self.model_label = QLabel("Whisper模型大小：")
        self.model_select = QComboBox(self)
        # 新增 large v1, large v2, large v3
        self.model_select.addItems(["tiny", "base", "small", "medium", "large", "large-v1", "large-v2", "large-v3", "large-v3-turbo"])

        self.language_label = QLabel("轉錄語言：")
        self.language_select = QComboBox(self)
        self.language_select.addItems(["Chinese", "English", "Spanish", "French", "German", "Japanese", "Korean"])
        self.language_select.setCurrentText("Chinese")  # 預設選中 Chinese

        self.output_label = QLabel("輸出目錄：")
        self.output_input = QLineEdit(self)
        self.output_button = QPushButton('選擇目錄', self)
        self.output_button.clicked.connect(self.browse_output_dir)

        # 新增選擇框，讓使用者選擇是否執行 PyAnnote
        self.pyannote_checkbox = QCheckBox("是否執行 PyAnnote 講者分離")
        self.pyannote_checkbox.setChecked(True)  # 預設選中

        self.start_button = QPushButton('開始批量轉錄與講者分離', self)
        self.start_button.clicked.connect(self.confirm_start_transcription)  # 呼叫確認函數

        # 進度條顯示在左側最底部
        self.progress = QProgressBar(self)
        self.progress.setAlignment(Qt.AlignCenter)

        # 添加左側所有組件
        left_layout.addWidget(self.file_label)
        left_layout.addWidget(self.file_input)
        left_layout.addWidget(self.browse_button)
        left_layout.addWidget(self.model_label)
        left_layout.addWidget(self.model_select)
        left_layout.addWidget(self.language_label)
        left_layout.addWidget(self.language_select)
        left_layout.addWidget(self.output_label)
        left_layout.addWidget(self.output_input)
        left_layout.addWidget(self.output_button)
        left_layout.addWidget(self.pyannote_checkbox)  # 添加選擇框
        left_layout.addWidget(self.start_button)
        left_layout.addWidget(self.progress)

        # 右側部分：狀態顯示和設備資訊
        right_layout = QVBoxLayout()

        self.status_display = QTextEdit()
        self.status_display.setReadOnly(True)  # 設為只讀

        self.device_info = QLabel("當前設備：未知")  # 顯示目前設備資訊

        # 添加右側所有組件
        right_layout.addWidget(self.status_display)
        right_layout.addWidget(self.device_info)

        # 將左右兩側加入主佈局
        main_layout.addLayout(left_layout)
        main_layout.addLayout(right_layout)

        self.setLayout(main_layout)
        self.setWindowTitle('批量語音轉錄與講者分離工具')
        self.setGeometry(300, 300, 800, 400)

    def browse_files(self):
        # 更新支援的音檔格式，包括 mp4, m4a, aac
        files, _ = QFileDialog.getOpenFileNames(self, '選擇音檔', '', '音檔 (*.wav *.mp3 *.mp4 *.m4a *.aac)')
        if files:
            self.file_input.clear()  # 清空列表
            self.file_input.addItems(files)  # 將選中的檔案添加到列表中


    def browse_output_dir(self):
        output_dir = QFileDialog.getExistingDirectory(self, '選擇輸出目錄')
        if output_dir:
            self.output_input.setText(output_dir)

    # 新增確認視窗函數
    def confirm_start_transcription(self):
        reply = QMessageBox.question(self, '確認', '是否確定要開始批量轉錄與講者分離？', QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.start_transcription()

    def start_transcription(self):
        audio_files = [self.file_input.item(i).text() for i in range(self.file_input.count())]  # 取得所有選擇的檔案
        model_size = self.model_select.currentText()
        language = self.language_select.currentText()  # 使用語言選擇框的值
        output_dir = self.output_input.text()
        run_pyannote = self.pyannote_checkbox.isChecked()  # 根據選擇框判斷是否執行 PyAnnote

        if not all([audio_files, model_size, language, output_dir]):
            QMessageBox.warning(self, "錯誤", "請確保所有欄位都有填寫")
            return

        self.thread = TranscriptionThread(audio_files, model_size, language, output_dir, run_pyannote)
        self.thread.progress_signal.connect(self.update_progress)
        self.thread.status_signal.connect(self.update_status)  # 連接狀態訊號
        self.thread.device_signal.connect(self.update_device_info)  # 連接設備訊號
        self.thread.start()

    # 更新狀態顯示的函數，將訊息添加到狀態框
    def update_status(self, message):
        self.status_display.append(message)

    # 更新當前設備訊息
    def update_device_info(self, device_info):
        self.device_info.setText(device_info)

    # 更新進度條
    def update_progress(self, value, message):
        self.progress.setValue(value)
        self.progress.setFormat(message)

# 主函數
if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = TranscriptionApp()
    ex.show()
    sys.exit(app.exec_())
