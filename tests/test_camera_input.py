from types import SimpleNamespace

import numpy as np

from backend.vision import camera_input


def test_camera_input_reports_unavailable_camera(monkeypatch, capsys):
    class Camera:
        released = False

        def isOpened(self):
            return False

        def release(self):
            self.released = True

    camera = Camera()
    monkeypatch.setattr(camera_input.cv2, "VideoCapture", lambda _source: camera)

    assert camera_input.main(["--camera", "4"]) == 1
    assert camera.released
    assert "Could not open" in capsys.readouterr().out


def test_camera_input_q_key_releases_camera_and_tracker(monkeypatch):
    class Camera:
        released = False

        def isOpened(self):
            return True

        def set(self, *_args):
            return True

        def read(self):
            return True, np.zeros((8, 8, 3), dtype=np.uint8)

        def release(self):
            self.released = True

    class Tracker:
        def __init__(self, **_kwargs):
            self.holistic = SimpleNamespace(close=lambda: setattr(self, "closed", True))
            self.closed = False

        def process(self, _frame):
            return {
                "face": [],
                "pose": [],
                "left_hand": [],
                "right_hand": [],
            }, SimpleNamespace(
                pose_landmarks=None,
                left_hand_landmarks=None,
                right_hand_landmarks=None,
                face_landmarks=None,
            )

    camera = Camera()
    tracker = Tracker()
    monkeypatch.setattr(camera_input.cv2, "VideoCapture", lambda _source: camera)
    monkeypatch.setattr(camera_input, "ApertureTracker", lambda **_kwargs: tracker)
    monkeypatch.setattr(camera_input.cv2, "namedWindow", lambda *_args: None)
    monkeypatch.setattr(camera_input.cv2, "imshow", lambda *_args: None)
    monkeypatch.setattr(camera_input.cv2, "waitKey", lambda _delay: ord("q"))
    monkeypatch.setattr(camera_input.cv2, "destroyAllWindows", lambda: None)

    assert camera_input.main([]) == 0
    assert camera.released
    assert tracker.closed
