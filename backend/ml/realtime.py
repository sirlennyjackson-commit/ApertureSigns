"""Streaming recognition: sliding window + confidence gate + debouncing.

Feed one frame at a time with :meth:`RealtimeRecognizer.push`. A
:class:`SignEvent` is returned only when a sign is *confirmed*, i.e. the same
confident, non-idle label won ``min_votes`` consecutive windows. After an event
the recognizer stays quiet until the signer rests, changes sign, or hands leave
the frame, so one held sign is not emitted repeatedly.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from backend.config.settings import IDLE_LABELS
from backend.ml.classifier import SignClassifier
from backend.ml.features import describe, featurize, movement_energy
from backend.ml.landmarks import frames_to_sequence, hand_activity


@dataclass(frozen=True)
class SignEvent:
    label: str
    confidence: float
    frame: int

    def as_dict(self) -> dict:
        return {"label": self.label, "confidence": round(self.confidence, 4), "frame": self.frame}


class RealtimeRecognizer:
    def __init__(
        self,
        classifier: SignClassifier,
        *,
        window: int = 45,
        stride: int = 3,
        min_frames: int | None = None,
        min_confidence: float = 0.75,
        min_votes: int = 2,
        min_activity: float = 0.3,
        idle_labels=IDLE_LABELS,
    ):
        self.classifier = classifier
        self.window = window
        self.stride = max(1, stride)
        self.min_frames = min_frames if min_frames is not None else max(8, window // 2)
        self.min_confidence = min_confidence
        self.min_votes = max(1, min_votes)
        self.min_activity = min_activity
        self.idle_labels = tuple(idle_labels)
        self.reset()

    def reset(self) -> None:
        self._buffer: deque = deque(maxlen=self.window)
        self._count = 0
        self._streak_label: str | None = None
        self._streak = 0
        self._armed = True
        self._emitted: str | None = None
        self._label: str | None = None
        self._confidence = 0.0
        self._energy = 0.0
        self._hands = False

    def state(self) -> dict:
        return {
            "label": self._label,
            "confidence": round(self._confidence, 4),
            "hands": self._hands,
            "buffered": len(self._buffer),
            "energy": round(self._energy, 5),
        }

    def push(self, frame: dict) -> SignEvent | None:
        self._buffer.append(frame)
        self._count += 1
        if len(self._buffer) < self.min_frames or self._count % self.stride:
            return None

        seq = frames_to_sequence(self._buffer)
        activity = hand_activity(seq)
        self._hands = activity > 0
        if activity < self.min_activity:
            self._label, self._confidence = None, 0.0
            self._streak_label, self._streak, self._armed = None, 0, True
            return None

        feats = featurize(seq)
        self._energy = movement_energy(feats)
        pred = self.classifier.predict_descriptor(describe(feats))
        self._label, self._confidence = pred.label, pred.confidence

        if pred.label in self.idle_labels:
            self._streak_label, self._streak, self._armed = None, 0, True
            return None
        if pred.confidence < self.min_confidence:
            self._streak_label, self._streak = None, 0
            return None

        if pred.label == self._streak_label:
            self._streak += 1
        else:
            self._streak_label, self._streak = pred.label, 1
        if pred.label != self._emitted:
            self._armed = True

        if self._armed and self._streak >= self.min_votes:
            self._armed, self._emitted = False, pred.label
            return SignEvent(pred.label, pred.confidence, self._count)
        return None
