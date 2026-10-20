"""Pure-NumPy sign classifier (inference only).

Training happens in ``backend.training.train`` (scikit-learn). The trained
scaler + multinomial logistic regression are exported as plain arrays in a
``.npz`` file, so the server needs no scikit-learn and no pickle (nothing in the
model file can execute code when loaded).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from backend.ml.features import DESCRIPTOR_DIM, FEATURE_VERSION, sequence_descriptor


@dataclass(frozen=True)
class Prediction:
    label: str
    confidence: float
    top: tuple  # ((label, confidence), ...) best first

    def as_dict(self) -> dict:
        return {
            "label": self.label,
            "confidence": round(self.confidence, 4),
            "top": [{"label": l, "confidence": round(c, 4)} for l, c in self.top],
        }


class SignClassifier:
    def __init__(self, labels, mean, scale, coef, intercept, meta: dict | None = None):
        self.labels = [str(x) for x in labels]
        self.mean = np.asarray(mean, np.float32)
        scale = np.asarray(scale, np.float32)
        self.scale = np.where(scale < 1e-8, 1.0, scale).astype(np.float32)
        self.coef = np.asarray(coef, np.float32)
        self.intercept = np.asarray(intercept, np.float32)
        self.meta = dict(meta or {})
        n_out = 2 if self.coef.shape[0] == 1 else self.coef.shape[0]
        if n_out != len(self.labels):
            raise ValueError("labels do not match the model's output size")
        if self.coef.shape[1] != DESCRIPTOR_DIM:
            raise ValueError(
                f"model expects {self.coef.shape[1]} features but this code produces "
                f"{DESCRIPTOR_DIM}; retrain the model"
            )

    # ------------------------------------------------------------------ #
    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(np.asarray(x, np.float32))
        logits = ((x - self.mean) / self.scale) @ self.coef.T + self.intercept
        if self.coef.shape[0] == 1:  # binary: softmax([0, d]) == [sigmoid(-d), sigmoid(d)]
            logits = np.concatenate([np.zeros_like(logits), logits], axis=1)
        logits = logits - logits.max(axis=1, keepdims=True)
        e = np.exp(logits)
        return e / e.sum(axis=1, keepdims=True)

    def predict_descriptor(self, descriptor: np.ndarray, top_k: int = 3) -> Prediction:
        p = self.predict_proba(descriptor)[0]
        order = np.argsort(p)[::-1][:top_k]
        top = tuple((self.labels[i], float(p[i])) for i in order)
        return Prediction(label=top[0][0], confidence=top[0][1], top=top)

    def predict(self, seq, top_k: int = 3) -> Prediction:
        return self.predict_descriptor(sequence_descriptor(seq), top_k)

    # ------------------------------------------------------------------ #
    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        meta = {**self.meta, "feature_version": FEATURE_VERSION, "descriptor_dim": DESCRIPTOR_DIM}
        np.savez_compressed(
            path,
            labels=np.array(self.labels, dtype="U64"),
            mean=self.mean,
            scale=self.scale,
            coef=self.coef,
            intercept=self.intercept,
            meta=np.array(json.dumps(meta)),
        )
        return path if path.suffix == ".npz" else path.with_suffix(path.suffix + ".npz")

    @classmethod
    def load(cls, path: str | Path) -> SignClassifier:
        with np.load(path, allow_pickle=False) as z:
            meta = json.loads(str(z["meta"].item()))
            if meta.get("feature_version") != FEATURE_VERSION:
                raise ValueError(
                    f"model was trained with feature version {meta.get('feature_version')}, "
                    f"code is version {FEATURE_VERSION}; retrain the model"
                )
            return cls(z["labels"].tolist(), z["mean"], z["scale"], z["coef"], z["intercept"], meta)
