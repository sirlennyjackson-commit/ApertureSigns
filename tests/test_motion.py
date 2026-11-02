from types import SimpleNamespace

import numpy as np

from backend.avatar.blender_bridge import BlenderBridge
from backend.motion.filter import OneEuroFilter
from backend.motion.models import MotionFrame
from backend.motion.pipeline import MotionPipeline
from backend.runtime.session import LiveSession
from app_ui_v1 import CaptureWorker


def test_one_euro_filter_tracks_a_step():
    euro = OneEuroFilter(min_cutoff=1.0, beta=0.04)
    first = euro.apply(0.0, 0.0)
    later = euro.apply(1.0, 0.2)
    assert first == 0.0
    assert 0.0 < later < 1.0


def test_motion_pipeline_filters_and_buffers_frames():
    pipeline = MotionPipeline(buffer_size=10)
    landmarks = {
        "pose": [
            {"x": 0.5, "y": 0.4, "z": 0.0, "visibility": 0.9} if index == 0 else {"x": 0.5, "y": 0.5, "z": 0.0, "visibility": 0.8}
            for index in range(29)
        ],
        "left_hand": [{"x": 0.2, "y": 0.3, "z": 0.0}],
        "right_hand": [],
        "face": [{"x": 0.5, "y": 0.2, "z": 0.0}],
    }
    landmarks["pose"][11] = {"x": 0.4, "y": 0.5, "z": 0.0, "visibility": 0.9}
    filtered, frame = pipeline.ingest(landmarks, timestamp=1.0)
    assert frame.sequence == 1
    assert frame.hands == 1
    assert "left_shoulder" in frame.joints
    assert filtered["face"] == landmarks["face"]
    assert "face" not in frame.overlay(filtered)
    assert frame.overlay(filtered, include_face=True)["face"] == landmarks["face"]
    assert pipeline.playback_tail(1)[0]["sequence"] == 1


def test_blender_bridge_does_not_send_until_started():
    bridge = BlenderBridge()
    assert bridge.send({"seq": 1}) is False
    status = bridge.start()
    assert status["enabled"] is True
    assert bridge.send({"seq": 1, "joints": {}}) is True
    bridge.stop()
    bridge.close()


def test_session_reports_vocabulary_and_browser_speech():
    session = LiveSession()
    status = session.components()
    assert status["language"] == "VOCABULARY"
    assert status["speech"] == "BROWSER TTS"
    assert "not sentence translation" in status["language_note"]
    session.set_language("asl")
    assert session.components()["config"]["language"] == "ASL"


def test_capture_worker_emits_frame_and_cleans_up():
    class Session:
        running = True
        last_error = None
        stopped = False

        def start_camera(self):
            return None

        def process_once(self):
            self.running = False
            return {"frame": b"frame", "overlay": {}, "motion": {}, "metrics": {}}

        def components(self):
            return {"vision": "LIVE"}

        def stop_camera(self):
            self.stopped = True

    session = Session()
    worker = CaptureWorker(session)
    frames = []
    errors = []
    worker.frame_ready.connect(lambda *payload: frames.append(payload))
    worker.camera_error.connect(errors.append)

    worker.run()

    assert len(frames) == 1
    assert errors == []
    assert session.stopped is True


def test_session_frames_include_face_and_lip_contours():
    class Camera:
        def read(self):
            return True, np.zeros((8, 8, 3), dtype=np.uint8)

        def release(self):
            pass

    class Tracker:
        history = []
        mp_face_mesh = SimpleNamespace(
            FACEMESH_CONTOURS={(2, 3), (1, 2)},
            FACEMESH_LIPS={(5, 6), (4, 5)},
        )

        def process(self, _frame):
            return {
                "face": [{"x": 0.5, "y": 0.4, "z": 0.0}],
                "pose": [],
                "left_hand": [],
                "right_hand": [],
            }, None

    session = LiveSession()
    session.running = True
    session.camera = Camera()
    session.tracker = Tracker()

    payload = session.process_once()

    assert payload is not None
    assert payload["overlay"]["face"] == [{"x": 0.5, "y": 0.4, "z": 0.0}]
    assert payload["overlay"]["face_connections"] == [[1, 2], [2, 3]]
    assert payload["overlay"]["lip_connections"] == [[4, 5], [5, 6]]
    session.stop_camera()


def test_motion_frame_blender_payload_includes_source():
    frame = MotionFrame.from_landmarks(
        {
            "pose": [{"x": 0.5, "y": 0.5, "z": 0.0, "visibility": 1.0}] * 17,
            "left_hand": [],
            "right_hand": [{"x": 0.8, "y": 0.4, "z": 0.1}],
        },
        timestamp=2.0,
        sequence=7,
    )
    payload = frame.to_blender()
    assert payload["source"] == "aperture-signs"
    assert payload["seq"] == 7
    assert payload["right_hand"]
