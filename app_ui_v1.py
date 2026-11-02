import json
import os
import shutil
import sys
import base64
import subprocess
import time
from pathlib import Path

from PyQt6.QtCore import QObject, QThread, QUrl, pyqtSignal, pyqtSlot
from PyQt6.QtWebChannel import QWebChannel
from PyQt6.QtWebEngineWidgets import QWebEngineView
from PyQt6.QtWidgets import QApplication, QMainWindow

from backend.avatar.avatar_engine import generate_avatar_sequence
from backend.runtime.session import LiveSession
from backend.translation.ksl_translator import supported_vocabulary, translate_to_ksl


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
HTML_PATH = os.path.join(BASE_DIR, "app", "ui", "platform.html")
BLENDER_RECEIVER_PATH = Path(BASE_DIR) / "blender" / "aperture_signs_receiver.py"


def find_blender_executable() -> str | None:
    configured = os.environ.get("BLENDER_EXECUTABLE")
    if configured and Path(configured).is_file():
        return configured
    on_path = shutil.which("blender") or shutil.which("blender.exe")
    if on_path:
        return on_path
    install_root = Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "Blender Foundation"
    installed = sorted(install_root.glob("Blender */blender.exe"), reverse=True)
    return str(installed[0]) if installed else None


class CaptureWorker(QThread):
    frame_ready = pyqtSignal(str, object, object, object, object)
    camera_error = pyqtSignal(str)
    components_ready = pyqtSignal(object)

    def __init__(self, session: LiveSession, parent=None):
        super().__init__(parent)
        self.session = session

    def run(self):
        last_ui_update = 0.0
        try:
            error = self.session.start_camera()
            self.components_ready.emit(self.session.components())
            if error:
                self.camera_error.emit(error)
                return
            while not self.isInterruptionRequested() and self.session.running:
                payload = self.session.process_once()
                if payload is None:
                    if self.session.last_error:
                        self.camera_error.emit(self.session.last_error)
                    break
                now = time.perf_counter()
                if now - last_ui_update >= 0.05:
                    frame = payload["frame"]
                    frame_b64 = base64.b64encode(frame).decode("ascii") if frame is not None else ""
                    self.frame_ready.emit(
                        frame_b64,
                        payload["overlay"],
                        payload["motion"],
                        payload["metrics"],
                        self.session.components(),
                    )
                    last_ui_update = now
                self.msleep(5)
        except Exception as exc:
            self.session.last_error = f"Capture failed: {exc}"
            self.camera_error.emit(self.session.last_error)
        finally:
            try:
                self.session.stop_camera()
            except Exception as exc:
                self.camera_error.emit(f"Camera cleanup failed: {exc}")
            self.components_ready.emit(self.session.components())


class VoiceCaptureWorker(QThread):
    transcript_ready = pyqtSignal(str)
    voice_error = pyqtSignal(str)

    def run(self):
        pythoncom = None
        try:
            import pythoncom as python_com
            import win32com.client

            pythoncom = python_com
            pythoncom.CoInitialize()
            recognizer = win32com.client.Dispatch("SAPI.SpSharedRecognizer")
            context = recognizer.CreateRecoContext()
            grammar = context.CreateGrammar(1)
            grammar.DictationLoad("", 0)
            worker = self

            class RecognitionEvents:
                def OnRecognition(self, _stream_number, _stream_position, _result_type, result):
                    text = result.PhraseInfo.GetText().strip()
                    if text:
                        worker.transcript_ready.emit(text)
                        worker.requestInterruption()

            event_sink = win32com.client.WithEvents(context, RecognitionEvents)
            grammar.DictationSetState(1)
            deadline = time.monotonic() + 20
            while not self.isInterruptionRequested() and time.monotonic() < deadline:
                pythoncom.PumpWaitingMessages()
                self.msleep(40)
            grammar.DictationSetState(0)
            del event_sink
            if not self.isInterruptionRequested() and time.monotonic() >= deadline:
                self.voice_error.emit("Voice input timed out. Try again or type instead.")
        except Exception as exc:
            self.voice_error.emit(f"Windows voice input failed: {exc}")
        finally:
            if pythoncom is not None:
                pythoncom.CoUninitialize()


class PlatformApi(QObject):
    def __init__(self, window: "MainWindow"):
        super().__init__(window)
        self.window = window

    @pyqtSlot()
    def startCapture(self):
        self.window.start_capture()

    @pyqtSlot()
    def stopCapture(self):
        self.window.stop_capture()

    @pyqtSlot()
    def startBlender(self):
        self.window.start_blender()

    @pyqtSlot()
    def stopBlender(self):
        self.window.session.blender.stop()
        self.window.push_components()

    @pyqtSlot()
    def beginAvatarPlayback(self):
        self.window.session.avatar_playback_active = True

    @pyqtSlot(str)
    def sendAvatarFrame(self, payload: str):
        self.window.send_avatar_frame(payload)

    @pyqtSlot()
    def endAvatarPlayback(self):
        self.window.session.avatar_playback_active = False

    @pyqtSlot()
    def startVoiceCapture(self):
        self.window.start_voice_capture()

    @pyqtSlot()
    def stopVoiceCapture(self):
        self.window.stop_voice_capture()

    @pyqtSlot()
    def startRecording(self):
        path = self.window.session.start_recording()
        self.window.run_js(
            "document.getElementById('sessionLog').textContent = " + json.dumps("Recording " + path)
        )
        self.window.push_components()

    @pyqtSlot()
    def stopRecording(self):
        self.window.session.stop_recording()
        self.window.push_components()

    @pyqtSlot(str)
    def applySettings(self, payload: str):
        data = json.loads(payload)
        cfg = self.window.session.config
        cfg.camera_index = int(data.get("camera_index", cfg.camera_index))
        cfg.language = str(data.get("language", cfg.language))
        cfg.min_cutoff = float(data.get("min_cutoff", cfg.min_cutoff))
        cfg.beta = float(data.get("beta", cfg.beta))
        cfg.blender_host = str(data.get("blender_host", cfg.blender_host))
        cfg.blender_port = int(data.get("blender_port", cfg.blender_port))
        cfg.avatar_blend_path = str(data.get("avatar_blend_path", cfg.avatar_blend_path))
        self.window.session.motion.configure_smoothing(cfg.min_cutoff, cfg.beta)
        self.window.session.blender.configure(cfg.blender_host, cfg.blender_port)
        self.window.push_components()

    @pyqtSlot(str)
    def translateText(self, text: str):
        self.window.translate_text(text)

    @pyqtSlot(result="QString")
    def getSupportedVocabulary(self) -> str:
        return json.dumps(supported_vocabulary())


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Aperture Signs")
        screen = QApplication.primaryScreen()
        available = screen.availableGeometry() if screen is not None else None
        width = min(1440, available.width() - 32) if available is not None else 1280
        height = min(900, available.height() - 48) if available is not None else 800
        self.setMinimumSize(min(960, width), min(620, height))
        self.resize(width, height)
        if available is not None:
            self.move(
                available.x() + (available.width() - width) // 2,
                available.y() + (available.height() - height) // 2,
            )
        self.session = LiveSession()
        self.worker = None
        self.voice_worker = None
        self.blender_process = None

        self.browser = QWebEngineView(self)
        self.setCentralWidget(self.browser)
        self.channel = QWebChannel(self.browser.page())
        self.api = PlatformApi(self)
        self.channel.registerObject("platform", self.api)
        self.browser.page().setWebChannel(self.channel)
        self.browser.loadFinished.connect(self.start_ui)
        self.browser.load(QUrl.fromLocalFile(HTML_PATH))

    def run_js(self, script: str):
        self.browser.page().runJavaScript(script)

    def start_ui(self, loaded: bool):
        if not loaded:
            return
        self.run_js("window.showScreen('platform');")
        self.push_components()
        self.start_capture()

    def start_capture(self):
        if self.worker is not None and self.worker.isRunning():
            return
        self.worker = CaptureWorker(self.session, self)
        self.worker.frame_ready.connect(self.on_frame)
        self.worker.camera_error.connect(self.on_camera_error)
        self.worker.components_ready.connect(self.on_components)
        self.worker.start()

    def stop_capture(self):
        self.session.running = False
        if self.worker is not None and self.worker.isRunning():
            self.worker.requestInterruption()
            self.worker.wait(2000)
        self.worker = None
        self.push_components()
        self.run_js("window.setCameraStatus('Capture stopped.');")

    def start_voice_capture(self):
        if self.voice_worker is not None and self.voice_worker.isRunning():
            self.run_js("window.setVoiceStatus('Voice input is already listening.');")
            return
        self.voice_worker = VoiceCaptureWorker(self)
        self.voice_worker.transcript_ready.connect(self.on_voice_transcript)
        self.voice_worker.voice_error.connect(self.on_voice_error)
        self.voice_worker.start()
        self.run_js("window.setVoiceStatus('Listening with Windows speech recognition…');")

    def stop_voice_capture(self):
        if self.voice_worker is not None and self.voice_worker.isRunning():
            self.voice_worker.requestInterruption()
            self.voice_worker.wait(1000)
        self.run_js("window.setVoiceStatus('Voice input stopped.');")

    def on_voice_transcript(self, text: str):
        self.run_js("window.setVoiceTranscript(" + json.dumps(text) + ");")

    def on_voice_error(self, message: str):
        self.run_js("window.setVoiceStatus(" + json.dumps(message) + ");")

    def start_blender(self):
        blend_path = Path(self.session.config.avatar_blend_path)
        if not blend_path.is_file():
            self.session.blender.last_error = f"Avatar file not found: {blend_path}"
            self.push_components()
            return
        blender_executable = find_blender_executable()
        if not blender_executable:
            self.session.blender.last_error = "Blender was not found. Set BLENDER_EXECUTABLE or install Blender."
            self.push_components()
            return
        if not BLENDER_RECEIVER_PATH.is_file():
            self.session.blender.last_error = f"Blender receiver script not found: {BLENDER_RECEIVER_PATH}"
            self.push_components()
            return
        if self.blender_process is None or self.blender_process.poll() is not None:
            try:
                self.blender_process = subprocess.Popen(
                    [blender_executable, str(blend_path), "--python", str(BLENDER_RECEIVER_PATH)],
                    cwd=BASE_DIR,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            except OSError as exc:
                self.session.blender.last_error = f"Could not launch Blender: {exc}"
                self.push_components()
                return
        self.session.blender.start()
        self.push_components()

    def on_frame(self, frame, overlay, motion, metrics, components):
        script = (
            "window.setState("
            + json.dumps(frame)
            + ", "
            + json.dumps(overlay)
            + ", "
            + json.dumps(motion)
            + ", "
            + json.dumps(metrics)
            + ", "
            + json.dumps(components)
            + ");"
        )
        self.run_js(script)

    def on_camera_error(self, message: str):
        self.run_js("window.setCameraStatus(" + json.dumps(message) + ");")
        self.push_components()

    def on_components(self, components):
        self.run_js("window.applyComponents(" + json.dumps(components) + ");")

    def push_components(self):
        self.on_components(self.session.components())

    def translate_text(self, text: str):
        translation = translate_to_ksl(text)
        motion_sequences = translation.pop("_motion_sequences")
        avatar = generate_avatar_sequence(translation["sign_sequence"], motion_sequences)
        self.run_js(
            "window.setTranslationResult("
            + json.dumps(translation)
            + ", "
            + json.dumps(avatar)
            + ");"
        )

    def send_avatar_frame(self, payload: str):
        try:
            frame = json.loads(payload)
        except json.JSONDecodeError as exc:
            self.session.blender.last_error = f"Invalid avatar frame: {exc}"
            self.push_components()
            return
        if not isinstance(frame, dict):
            return
        self.session.blender.send(
            {
                "v": 1,
                "source": "aperture-signs-sign-playback",
                "joints": frame.get("joints") or {},
                "left_hand": frame.get("left_hand") or [],
                "right_hand": frame.get("right_hand") or [],
            }
        )

    def closeEvent(self, event):
        self.stop_voice_capture()
        self.stop_capture()
        self.session.blender.close()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
