"""Organize filtered landmarks into motion frames and a playback buffer."""

from __future__ import annotations

import time
from collections import deque

from backend.motion.filter import LandmarkSmoother
from backend.motion.models import MotionFrame


class MotionPipeline:
    def __init__(self, buffer_size: int = 180, min_cutoff: float = 1.2, beta: float = 0.05):
        self.smoother = LandmarkSmoother(min_cutoff=min_cutoff, beta=beta)
        self.buffer: deque[MotionFrame] = deque(maxlen=buffer_size)
        self.sequence = 0
        self.previous: MotionFrame | None = None
        self.last_ingest_ms = 0.0
        self.enabled = True

    def configure_smoothing(self, min_cutoff: float, beta: float) -> None:
        self.smoother.configure(min_cutoff, beta)

    def reset(self) -> None:
        self.smoother.reset()
        self.buffer.clear()
        self.sequence = 0
        self.previous = None

    def ingest(self, landmarks: dict, timestamp: float | None = None) -> tuple[dict, MotionFrame]:
        started = time.perf_counter()
        stamp = time.time() if timestamp is None else timestamp
        filtered = self.smoother.smooth(landmarks, stamp) if self.enabled else dict(landmarks)
        self.sequence += 1
        frame = MotionFrame.from_landmarks(
            filtered,
            timestamp=stamp,
            sequence=self.sequence,
            previous=self.previous,
        )
        self.previous = frame
        self.buffer.append(frame)
        self.last_ingest_ms = (time.perf_counter() - started) * 1000.0
        return filtered, frame

    def playback_tail(self, count: int = 60, compact: bool = True) -> list[dict]:
        frames = list(self.buffer)[-count:]
        if not compact:
            return [frame.preview() for frame in frames]
        return [
            {
                "sequence": frame.sequence,
                "joints": frame.joints,
                "hands": frame.hands,
                "energy": frame.motion_energy,
                "left_hand": frame.left_hand,
                "right_hand": frame.right_hand,
            }
            for frame in frames
        ]
