"""Landmark containers, validation, mirroring and (de)serialisation.

A *frame* is a dict with keys ``left``, ``right``, ``pose``, ``face``. Each
value is a NumPy array or ``None`` when that part was not detected:

    left / right : (21, 3)  x, y, z   (MediaPipe hand landmarks)
    pose         : (33, 4)  x, y, z, visibility
    face         : (3,)     mouth_open, brow_raise_left, brow_raise_right

A *sequence* stacks frames along time and encodes "missing" as NaN:

    left / right : (T, 21, 3)   pose : (T, 33, 4)   face : (T, 3)
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

SEQ_KEYS = ("left", "right", "pose", "face")
SHAPES = {"left": (21, 3), "right": (21, 3), "pose": (33, 4), "face": (3,)}

Frame = dict  # key -> np.ndarray | None
SeqDict = dict  # key -> np.ndarray with a leading time axis

# MediaPipe pose indices that swap when the body is mirrored left <-> right.
_PAIRS = [(1, 4), (2, 5), (3, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16),
          (17, 18), (19, 20), (21, 22), (23, 24), (25, 26), (27, 28), (29, 30),
          (31, 32)]
POSE_MIRROR_PERM = list(range(33))
for _a, _b in _PAIRS:
    POSE_MIRROR_PERM[_a], POSE_MIRROR_PERM[_b] = _b, _a

# FaceMesh indices used for the three non-manual metrics.
_TOP, _CHIN, _LIP_UP, _LIP_LOW = 10, 152, 13, 14
_L_BROW, _L_EYE, _R_BROW, _R_EYE = 105, 159, 334, 386


# --------------------------------------------------------------------------- #
# Building sequences
# --------------------------------------------------------------------------- #
def empty_frame() -> Frame:
    return {k: None for k in SEQ_KEYS}


def empty_sequence(t: int) -> SeqDict:
    return {k: np.full((t, *SHAPES[k]), np.nan, dtype=np.float32) for k in SEQ_KEYS}


def frames_to_sequence(frames: Sequence[Frame]) -> SeqDict:
    seq = empty_sequence(len(frames))
    for i, frame in enumerate(frames):
        for key in SEQ_KEYS:
            value = frame.get(key)
            if value is not None:
                seq[key][i] = value
    return seq


def hand_activity(seq: Mapping) -> float:
    """Fraction of frames in which at least one hand is visible."""
    if len(seq["left"]) == 0:
        return 0.0
    left = np.isfinite(seq["left"][:, 0, 0])
    right = np.isfinite(seq["right"][:, 0, 0])
    return float(np.mean(left | right))


# --------------------------------------------------------------------------- #
# Face metrics
# --------------------------------------------------------------------------- #
def face_metrics(face: np.ndarray) -> np.ndarray | None:
    """Reduce a full face mesh to (mouth_open, brow_left, brow_right).

    All values are normalised by face height, so they are scale invariant.
    """
    if face is None or len(face) <= _R_EYE:
        return None
    height = abs(float(face[_CHIN, 1] - face[_TOP, 1]))
    if height < 1e-6:
        return None
    return np.array(
        [
            abs(face[_LIP_LOW, 1] - face[_LIP_UP, 1]) / height,
            (face[_L_EYE, 1] - face[_L_BROW, 1]) / height,
            (face[_R_EYE, 1] - face[_R_BROW, 1]) / height,
        ],
        dtype=np.float32,
    )


# --------------------------------------------------------------------------- #
# Network payloads (browser -> server)
# --------------------------------------------------------------------------- #
def _array(value, shape: tuple, name: str, pad_vis: bool = False):
    if value is None:
        return None
    try:
        arr = np.asarray(value, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"'{name}' must be numeric") from exc
    if pad_vis and arr.ndim == 2 and arr.shape[1] == 3:
        arr = np.concatenate([arr, np.ones((arr.shape[0], 1), np.float32)], axis=1)
    if arr.shape != shape:
        raise ValueError(f"'{name}' must have shape {shape}, got {arr.shape}")
    if not np.isfinite(arr).all() or np.abs(arr).max() > 10:
        raise ValueError(f"'{name}' contains non-finite or out-of-range values")
    return arr


def frame_from_payload(payload: Mapping) -> Frame:
    """Validate a JSON frame: ``{"left": [[x,y,z]*21], "right": ..., "pose": ...}``.

    Any key may be missing or ``null``. ``face_metrics`` is a 3-number list.
    """
    if not isinstance(payload, Mapping):
        raise ValueError("frame must be a JSON object")
    return {
        "left": _array(payload.get("left"), SHAPES["left"], "left"),
        "right": _array(payload.get("right"), SHAPES["right"], "right"),
        "pose": _array(payload.get("pose"), SHAPES["pose"], "pose", pad_vis=True),
        "face": _array(payload.get("face_metrics"), SHAPES["face"], "face_metrics"),
    }


# --------------------------------------------------------------------------- #
# Mirroring (for left-handed signers / data augmentation)
# --------------------------------------------------------------------------- #
def mirror_sequence(seq: Mapping) -> SeqDict:
    out = {k: np.array(v, copy=True) for k, v in seq.items()}
    out["left"], out["right"] = out["right"], out["left"]
    for key in ("left", "right", "pose"):
        out[key][..., 0] = 1.0 - out[key][..., 0]
    out["pose"] = out["pose"][:, POSE_MIRROR_PERM]
    out["face"] = out["face"][:, [0, 2, 1]]
    return out


# --------------------------------------------------------------------------- #
# Disk format
# --------------------------------------------------------------------------- #
def save_sequence(path: str | Path, seq: Mapping) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **{k: np.asarray(seq[k], np.float32) for k in SEQ_KEYS})


def load_sequence(path: str | Path) -> SeqDict:
    with np.load(path, allow_pickle=False) as z:
        missing = [k for k in SEQ_KEYS if k not in z]
        if missing:
            raise ValueError(f"missing arrays {missing}")
        seq = {k: z[k].astype(np.float32) for k in SEQ_KEYS}
    t = len(seq["pose"])
    for key in SEQ_KEYS:
        if seq[key].shape != (t, *SHAPES[key]):
            raise ValueError(f"'{key}' has shape {seq[key].shape}, expected (T, *{SHAPES[key]})")
    if t < 2 or not math.isfinite(float(np.nansum(seq["pose"]))):
        raise ValueError("sequence too short or corrupt")
    return seq
