"""Live camera session: vision → motion → optional Blender stream."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2

from backend.avatar.blender_bridge import BlenderBridge
from backend.motion.pipeline import MotionPipeline
from backend.vision.sign_recognizer import SignRecognizer


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class SessionConfig:
    camera_index: int = 0
    width: int = 1280
    height: int = 720
    fps: int = 30
    blender_host: str = "127.0.0.1"
    blender_port: int = 5005
    min_cutoff: float = 1.2
    beta: float = 0.05
    language: str = "KSL"
    avatar_blend_path: str = str(
        Path.home() / "Desktop" / "Everything" / "APERTURE_SIGNS_AVATAR_V1.blend"
    )


class LiveSession:
    def __init__(self, config: SessionConfig | None = None):
        self.config = config or SessionConfig()
        self.tracker = None
        self.recognizer = SignRecognizer(sequence_length=30)
        self.motion = MotionPipeline(
            min_cutoff=self.config.min_cutoff,
            beta=self.config.beta,
        )
        self.blender = BlenderBridge(self.config.blender_host, self.config.blender_port)
        self.camera = None
        self.running = False
        self.fps = 0.0
        self.latency_ms = 0.0
        self.last_error: str | None = None
        self.mediapipe_error: str | None = None
        self._previous_time = time.perf_counter()
        self.recording = False
        self.record_handle = None
        self.record_path: Path | None = None
        self.frames_processed = 0
        self.avatar_playback_active = False

    def start_camera(self) -> str | None:
        self.last_error = None
        try:
            from backend.vision.hand_tracker import ApertureTracker, MEDIAPIPE_IMPORT_ERROR
        except Exception as exc:  # pragma: no cover - import surface
            self.mediapipe_error = str(exc)
            return self.mediapipe_error

        try:
            self.tracker = ApertureTracker(apply_smoothing=False)
        except RuntimeError as exc:
            self.mediapipe_error = str(exc) if MEDIAPIPE_IMPORT_ERROR is None else str(MEDIAPIPE_IMPORT_ERROR)
            return str(exc)

        cap = cv2.VideoCapture(self.config.camera_index)
        cap.set(
            cv2.CAP_PROP_FOURCC,
            cv2.VideoWriter_fourcc(*"MJPG"),
        )
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
        cap.set(cv2.CAP_PROP_FPS, self.config.fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
        if not cap.isOpened():
            cap.release()
            if self.tracker is not None and getattr(self.tracker, "holistic", None) is not None:
                self.tracker.holistic.close()
            self.tracker = None
            self.last_error = "Camera unavailable. Connect a camera and start capture again."
            return self.last_error
        self.camera = cap
        self.running = True
        self._previous_time = time.perf_counter()
        return None

    def stop_camera(self) -> None:
        self.running = False
        if self.camera is not None:
            self.camera.release()
            self.camera = None
        if self.tracker is not None and getattr(self.tracker, "holistic", None) is not None:
            self.tracker.holistic.close()
            self.tracker = None
        self.stop_recording()

    def start_recording(self) -> str:
        folder = PROJECT_ROOT / "data" / "sessions"
        folder.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        self.record_path = folder / f"session-{stamp}.jsonl"
        self.record_handle = self.record_path.open("w", encoding="utf-8")
        self.recording = True
        return str(self.record_path)

    def stop_recording(self) -> None:
        self.recording = False
        if self.record_handle is not None:
            self.record_handle.close()
            self.record_handle = None

    def process_once(self) -> dict[str, Any] | None:
        if not self.running or self.camera is None or self.tracker is None:
            return None
        started = time.perf_counter()
        ok, frame = self.camera.read()
        if not ok:
            self.last_error = "Camera feed lost. Check the camera connection."
            self.running = False
            return None

        frame = cv2.flip(frame, 1)
        data, _results = self.tracker.process(frame)
        filtered, motion = self.motion.ingest(data)
        recognition = self.recognizer.analyze(self.tracker.history)
        prediction = recognition["prediction"]
        if not self.avatar_playback_active:
            self.blender.send(motion.to_blender())

        if self.recording and self.record_handle is not None:
            self.record_handle.write(json.dumps(motion.preview()) + "\n")

        ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
        if not ok:
            self.last_error = "Could not encode a camera frame."
            return None

        now = time.perf_counter()
        delta = max(now - self._previous_time, 0.001)
        self._previous_time = now
        instantaneous = 1.0 / delta
        self.fps = 0.85 * self.fps + 0.15 * instantaneous if self.fps else instantaneous
        self.latency_ms = (now - started) * 1000.0
        self.frames_processed += 1

        recognized = prediction["label"]
        if prediction["status"] == "stabilizing":
            recognized = f"{prediction['candidate']}..."

        return {
            "frame": encoded.tobytes(),
            "overlay": {
                **motion.overlay(filtered, include_face=True),
                "face_connections": [
                    list(edge) for edge in sorted(self.tracker.mp_face_mesh.FACEMESH_CONTOURS)
                ],
                "lip_connections": [
                    list(edge) for edge in sorted(self.tracker.mp_face_mesh.FACEMESH_LIPS)
                ],
            },
            "motion": motion.preview(),
            "playback": self.motion.playback_tail(90),
            "metrics": {
                "camera": True,
                "hands": motion.hands,
                "confidence": motion.pose_confidence,
                "fps": self.fps,
                "latency_ms": self.latency_ms,
                "filter_ms": self.motion.last_ingest_ms,
                "landmarks": motion.landmark_count,
                "motion": motion.motion_energy,
                "sign": recognized,
                "sign_confidence": prediction["confidence"],
                "recognition_status": prediction["status"],
                "model_loaded": self.recognizer.model is not None,
                "model_error": self.recognizer.model_error,
                "sequence": motion.sequence,
                "recording": self.recording,
            },
        }

    def components(self) -> dict[str, Any]:
        model_ok = self.recognizer.model is not None
        return {
            "vision": "LIVE" if self.running else "STOPPED",
            "landmarks": "TRACKING" if self.running else "IDLE",
            "motion": "ACTIVE" if self.running else "IDLE",
            "blender": self.blender.status(),
            "classifier": "LOADED" if model_ok else "UNAVAILABLE",
            "language": "VOCABULARY",
            "speech": "BROWSER TTS",
            "language_note": (
                f"{self.config.language} sign recognition is experimental. "
                "Speech-to-sign uses exact recorded vocabulary samples, not sentence translation."
            ),
            "model_error": self.recognizer.model_error,
            "camera_error": self.last_error,
            "mediapipe_error": self.mediapipe_error,
            "frames_processed": self.frames_processed,
            "recording": self.recording,
            "record_path": str(self.record_path) if self.record_path else None,
            "smoothing": {
                "min_cutoff": self.motion.smoother.min_cutoff,
                "beta": self.motion.smoother.beta,
                "enabled": self.motion.enabled,
            },
            "config": {
                "camera_index": self.config.camera_index,
                "language": self.config.language,
                "blender_host": self.blender.host,
                "blender_port": self.blender.port,
                "avatar_blend_path": self.config.avatar_blend_path,
            },
        }

    def set_smoothing(self, enabled: bool, min_cutoff: float | None = None, beta: float | None = None) -> None:
        self.motion.enabled = bool(enabled)
        cutoff = self.motion.smoother.min_cutoff if min_cutoff is None else float(min_cutoff)
        beta_value = self.motion.smoother.beta if beta is None else float(beta)
        self.motion.configure_smoothing(cutoff, beta_value)
        self.config.min_cutoff = cutoff
        self.config.beta = beta_value

    def configure_blender(self, host: str, port: int) -> dict[str, Any]:
        self.blender.configure(host, int(port))
        self.config.blender_host = host
        self.config.blender_port = int(port)
        return self.blender.status()

    def set_language(self, language: str) -> None:
        self.config.language = language.strip().upper() or "KSL"

    def set_camera_index(self, index: int) -> None:
        self.config.camera_index = int(index)
