"""Headless MediaPipe Holistic wrapper (no UI, importable by servers and scripts)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from backend.ml.landmarks import Frame, empty_frame, face_metrics


@dataclass
class Tracked:
    frame: Frame                      # smoothed landmarks, ready for the recognizer
    face_mesh: np.ndarray | None      # (N, 3) full mesh, for drawing only
    results: Any                      # raw MediaPipe results, for drawing only


def _to_array(landmarks, with_visibility: bool = False) -> np.ndarray:
    if with_visibility:
        return np.array([[l.x, l.y, l.z, l.visibility] for l in landmarks.landmark], np.float32)
    return np.array([[l.x, l.y, l.z] for l in landmarks.landmark], np.float32)


class ApertureTracker:
    """Detects hands, pose and face, and smooths them over time.

    Smoothing is an exponential moving average (``smoothing_alpha``; 1.0 = off).
    Unlike a naive EMA it *resets* a part after ``max_gap`` missed frames, so a
    hand that re-appears somewhere new is not dragged in from its old position.
    """

    def __init__(
        self,
        smoothing_alpha: float = 0.6,
        max_gap: int = 4,
        model_complexity: int = 1,
        min_detection_confidence: float = 0.35,
        min_tracking_confidence: float = 0.30,
        refine_face_landmarks: bool = False,
    ):
        import mediapipe as mp  # imported lazily so the API server does not need it

        self.smoothing_alpha = smoothing_alpha
        self.max_gap = max_gap
        self._state: dict[str, np.ndarray | None] = {"left": None, "right": None, "pose": None}
        self._missed = {k: 0 for k in self._state}
        self.holistic = mp.solutions.holistic.Holistic(
            static_image_mode=False,
            model_complexity=model_complexity,
            smooth_landmarks=True,
            refine_face_landmarks=refine_face_landmarks,
            enable_segmentation=False,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def _smooth(self, key: str, current: np.ndarray | None) -> np.ndarray | None:
        if current is None:
            self._missed[key] += 1
            if self._missed[key] > self.max_gap:
                self._state[key] = None
            return None
        self._missed[key] = 0
        previous = self._state[key]
        if previous is None or self.smoothing_alpha >= 1.0:
            smoothed = current
        else:
            smoothed = current.copy()
            smoothed[:, :3] = previous[:, :3] + self.smoothing_alpha * (current[:, :3] - previous[:, :3])
        self._state[key] = smoothed
        return smoothed

    def process(self, bgr: np.ndarray) -> Tracked:
        import cv2

        results = self.holistic.process(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        raw = {
            "left": _to_array(results.left_hand_landmarks) if results.left_hand_landmarks else None,
            "right": _to_array(results.right_hand_landmarks) if results.right_hand_landmarks else None,
            "pose": _to_array(results.pose_landmarks, True) if results.pose_landmarks else None,
        }
        frame = empty_frame()
        for key, value in raw.items():
            frame[key] = self._smooth(key, value)
        face = _to_array(results.face_landmarks) if results.face_landmarks else None
        frame["face"] = face_metrics(face)
        return Tracked(frame=frame, face_mesh=face, results=results)

    def close(self) -> None:
        self.holistic.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
