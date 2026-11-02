"""One Euro filter for landmark streams.

Reduces jitter while keeping fast hand motion responsive.
"""

from __future__ import annotations

import math


class OneEuroFilter:
    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.04, dcutoff: float = 1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.dcutoff = dcutoff
        self._x = None
        self._dx = 0.0
        self._timestamp = None

    def reset(self) -> None:
        self._x = None
        self._dx = 0.0
        self._timestamp = None

    def apply(self, value: float, timestamp: float) -> float:
        if self._x is None or self._timestamp is None:
            self._x = value
            self._timestamp = timestamp
            return value

        dt = timestamp - self._timestamp
        if dt <= 0.0:
            return self._x

        dx = (value - self._x) / dt
        self._dx = self._low_pass(dx, self._dx, self._alpha(dt, self.dcutoff))
        cutoff = self.min_cutoff + self.beta * abs(self._dx)
        self._x = self._low_pass(value, self._x, self._alpha(dt, cutoff))
        self._timestamp = timestamp
        return self._x

    @staticmethod
    def _alpha(dt: float, cutoff: float) -> float:
        tau = 1.0 / (2.0 * math.pi * max(cutoff, 1e-6))
        return 1.0 / (1.0 + tau / dt)

    @staticmethod
    def _low_pass(value: float, previous: float, alpha: float) -> float:
        return alpha * value + (1.0 - alpha) * previous


class LandmarkSmoother:
    """Independent One Euro filters for each landmark axis."""

    # Face mesh is dense; smoothing it every frame costs more than it helps the lab views.
    GROUPS = ("pose", "left_hand", "right_hand")

    def __init__(self, min_cutoff: float = 1.2, beta: float = 0.05):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self._filters: dict[tuple, OneEuroFilter] = {}

    def configure(self, min_cutoff: float, beta: float) -> None:
        self.min_cutoff = min_cutoff
        self.beta = beta
        for euro in self._filters.values():
            euro.min_cutoff = min_cutoff
            euro.beta = beta

    def reset(self) -> None:
        self._filters.clear()

    def smooth(self, landmarks: dict, timestamp: float) -> dict:
        filtered = {}
        for group in self.GROUPS:
            points = landmarks.get(group) or []
            if not points:
                filtered[group] = []
                continue
            out = []
            for index, point in enumerate(points):
                item = dict(point)
                for axis in ("x", "y", "z"):
                    key = (group, index, axis)
                    euro = self._filters.get(key)
                    if euro is None:
                        euro = OneEuroFilter(self.min_cutoff, self.beta)
                        self._filters[key] = euro
                    item[axis] = euro.apply(float(point[axis]), timestamp)
                out.append(item)
            filtered[group] = out
        for group, points in landmarks.items():
            if group not in filtered:
                filtered[group] = points
        return filtered
