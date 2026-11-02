import numpy as np
from collections import Counter, deque
import logging
import math
import pickle
from pathlib import Path
import sys


LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.training.features_v2 import FEATURE_DIM, FEATURE_VERSION, extract


class SignRecognizer:
    """
    Temporal landmark feature extractor.

    The tracker supplies a sequence of face, pose, and hand
    landmarks. This class converts that sequence into a
    normalized feature vector suitable for sign recognition.
    """

    def __init__(self, sequence_length=30, model_path=None):
        self.sequence_length = sequence_length
        self._prediction_votes = deque(maxlen=5)
        self.model = None
        self.classes = []
        self.confidence_threshold = 0.4
        self.model_error = None

        path = Path(model_path) if model_path else (
            PROJECT_ROOT / "models" / "sign" / "classifier" / "aperture_sign_v2.joblib"
        )
        try:
            import joblib

            bundle = joblib.load(path)
            if not isinstance(bundle, dict):
                raise ValueError("Classifier bundle must be a dictionary")
            if bundle.get("feature_version") != FEATURE_VERSION:
                raise ValueError("Classifier feature version does not match runtime")
            model = bundle["model"]
            classes = bundle["classes"]
            if not isinstance(classes, (list, tuple)) or not classes:
                raise ValueError("Classifier bundle has no classes")
            if not callable(getattr(model, "predict_proba", None)):
                raise ValueError("Classifier does not provide predict_proba")
            model_classes = getattr(model, "classes_", None)
            if model_classes is not None and len(model_classes) != len(classes):
                raise ValueError("Classifier class count does not match bundle")
            feature_count = getattr(model, "n_features_in_", FEATURE_DIM)
            if feature_count != FEATURE_DIM:
                raise ValueError(
                    f"Classifier expects {feature_count} features; runtime produces {FEATURE_DIM}"
                )

            threshold = float(bundle.get("threshold", 0.4))
            if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
                raise ValueError("Classifier threshold must be between 0 and 1")

            self.model = model
            self.classes = list(classes)
            self.confidence_threshold = threshold
        except (
            AttributeError,
            EOFError,
            ImportError,
            KeyError,
            OSError,
            pickle.UnpicklingError,
            TypeError,
            ValueError,
        ) as exc:
            self.model_error = str(exc)
            LOGGER.warning("Sign classifier unavailable at %s: %s", path, exc)

    def _hand_features(self, hand):
        if not hand:
            return np.zeros(63, dtype=np.float32)

        wrist = hand[0]

        features = []

        for landmark in hand:
            features.extend([
                landmark["x"] - wrist["x"],
                landmark["y"] - wrist["y"],
                landmark["z"] - wrist["z"],
            ])

        return np.asarray(features, dtype=np.float32)

    def frame_features(self, frame):
        """
        Convert one tracked frame into a fixed-size vector.

        Current representation:
        left hand  = 21 x 3
        right hand = 21 x 3
        pose       = 33 x 3
        """

        left = self._hand_features(frame["left_hand"])
        right = self._hand_features(frame["right_hand"])

        pose = frame["pose"]

        if pose:
            pose_features = []

            for landmark in pose:
                pose_features.extend([
                    landmark["x"],
                    landmark["y"],
                    landmark["z"],
                ])

            pose_features = np.asarray(
                pose_features,
                dtype=np.float32,
            )
        else:
            pose_features = np.zeros(99, dtype=np.float32)

        return np.concatenate([
            left,
            right,
            pose_features,
        ])

    def sequence_features(self, history):
        """
        Convert the recent temporal history into
        a fixed-size sequence.

        Short sequences are padded with zeros.
        Long sequences use the most recent frames.
        """

        frames = list(history)

        if len(frames) > self.sequence_length:
            frames = frames[-self.sequence_length:]

        feature_frames = [
            self.frame_features(frame)
            for frame in frames
        ]

        if not feature_frames:
            return np.zeros(
                (self.sequence_length, 225),
                dtype=np.float32,
            )

        while len(feature_frames) < self.sequence_length:
            feature_frames.insert(
                0,
                np.zeros_like(feature_frames[0]),
            )

        return np.asarray(
            feature_frames,
            dtype=np.float32,
        )

    def motion_features(self, sequence):
        """
        Calculate frame-to-frame movement.

        This is particularly important for motion-based
        signs such as J and Z.
        """

        if len(sequence) < 2:
            return np.zeros_like(sequence)

        return np.diff(
            sequence,
            axis=0,
        )

    def analyze(self, history):
        sequence = self.sequence_features(history)
        motion = self.motion_features(sequence)

        movement_energy = float(
            np.mean(np.abs(motion))
        )

        return {
            "sequence": sequence,
            "motion": motion,
            "movement_energy": movement_energy,
            "frames": len(history),
            "prediction": self.predict(sequence, len(history)),
        }

    def predict(self, sequence, frame_count):
        if self.model is None:
            return {
                "status": "unavailable",
                "label": "MODEL OFFLINE",
                "candidate": None,
                "confidence": 0.0,
            }

        if frame_count < self.sequence_length:
            self._prediction_votes.append(None)
            return {
                "status": "warming_up",
                "label": "COLLECTING",
                "candidate": None,
                "confidence": 0.0,
            }

        hand_frames = np.any(np.abs(sequence[:, :126]) > 1e-6, axis=1)
        if int(hand_frames.sum()) < max(5, self.sequence_length // 5):
            self._prediction_votes.append(None)
            return {
                "status": "idle",
                "label": "SHOW SIGN",
                "candidate": None,
                "confidence": 0.0,
            }

        features = extract(sequence).reshape(1, -1)
        probabilities = self.model.predict_proba(features)[0]
        best_index = int(np.argmax(probabilities))
        candidate = str(self.classes[best_index])
        confidence = float(probabilities[best_index])
        accepted = confidence >= self.confidence_threshold
        self._prediction_votes.append(candidate if accepted else None)

        votes = Counter(label for label in self._prediction_votes if label is not None)
        if votes:
            label, count = votes.most_common(1)[0]
            if count >= 3:
                return {
                    "status": "recognized",
                    "label": label,
                    "candidate": candidate,
                    "confidence": confidence,
                }

        return {
            "status": "stabilizing" if accepted else "uncertain",
            "label": "UNKNOWN",
            "candidate": candidate if accepted else None,
            "confidence": confidence,
        }
