"""Turn landmark sequences into normalised features.

Everything is expressed relative to the signer's *body frame* (origin = shoulder
midpoint, unit = shoulder width) and to each hand's own size, so the features do
not depend on how far the person is from the camera or where they stand.

Per frame (FEATURE_DIM = 145):
    left hand shape   60   20 landmarks (wrist removed) / hand size
    right hand shape  60
    hand locations     4   wrist x,y of each hand in the body frame
    upper body        14   nose, shoulders, elbows, wrists (x,y) in the body frame
    face metrics       3   mouth open, left/right brow raise
    presence masks     4   left hand, right hand, pose, face
"""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np

FEATURE_VERSION = 2
BODY_POINTS = (0, 11, 12, 13, 14, 15, 16)
MIN_VISIBILITY = 0.3
MIN_SHOULDER_WIDTH = 0.02
MIN_HAND_SIZE = 1e-3
DEFAULT_ORIGIN = np.array([0.5, 0.5], np.float32)
DEFAULT_SCALE = 0.25
CLIP = 6.0
STEPS = 16  # time-normalised length used by `describe`

_HAND = 20 * 3
OFF_LSHAPE, OFF_RSHAPE = 0, _HAND
OFF_LLOC = 2 * _HAND
OFF_RLOC = OFF_LLOC + 2
OFF_POSE = OFF_RLOC + 2
OFF_FACE = OFF_POSE + len(BODY_POINTS) * 2
OFF_MASK = OFF_FACE + 3
FEATURE_DIM = OFF_MASK + 4
DESCRIPTOR_DIM = FEATURE_DIM * 10


def _fill(values: np.ndarray, ok: np.ndarray) -> np.ndarray:
    """Forward-fill, then back-fill, rows where ``ok`` is False."""
    out = values.copy()
    idx = np.where(ok, np.arange(len(ok)), -1)
    np.maximum.accumulate(idx, out=idx)
    seen = idx >= 0
    out[seen] = values[idx[seen]]
    first = int(np.argmax(ok))
    out[:first] = values[first]
    return out


def body_frame(pose: np.ndarray):
    """Return (origin (T,2), scale (T,), ok (T,)) from shoulders, carried forward."""
    t = pose.shape[0]
    ls, rs = pose[:, 11, :], pose[:, 12, :]
    with np.errstate(invalid="ignore"):
        ok = (
            np.isfinite(ls[:, :3]).all(1)
            & np.isfinite(rs[:, :3]).all(1)
            & (ls[:, 3] >= MIN_VISIBILITY)
            & (rs[:, 3] >= MIN_VISIBILITY)
        )
        origin = (ls[:, :2] + rs[:, :2]) / 2
        scale = np.linalg.norm(ls[:, :2] - rs[:, :2], axis=1)
        ok &= scale > MIN_SHOULDER_WIDTH
    if not ok.any():
        return np.tile(DEFAULT_ORIGIN, (t, 1)), np.full(t, DEFAULT_SCALE, np.float32), ok
    return _fill(origin, ok), _fill(scale[:, None], ok)[:, 0], ok


def featurize(seq: Mapping) -> np.ndarray:
    """(T, FEATURE_DIM) float32 features for a landmark sequence."""
    pose = seq["pose"]
    t = pose.shape[0]
    origin, scale, pose_ok = body_frame(pose)

    shapes, locs, present = [], [], []
    for key in ("left", "right"):
        hand = seq[key]
        here = np.isfinite(hand[:, 0, 0])
        h = np.nan_to_num(hand, nan=0.0)
        size = np.maximum(np.linalg.norm(h[:, 9, :2] - h[:, 0, :2], axis=1), MIN_HAND_SIZE)
        shape = (h[:, 1:, :] - h[:, :1, :]) / size[:, None, None]
        loc = (h[:, 0, :2] - origin) / scale[:, None]
        shape[~here] = 0.0
        loc[~here] = 0.0
        shapes.append(shape.reshape(t, -1))
        locs.append(loc)
        present.append(here)

    body = np.nan_to_num((pose[:, BODY_POINTS, :2] - origin[:, None, :]) / scale[:, None, None])
    body[~pose_ok] = 0.0

    face_ok = np.isfinite(seq["face"][:, 0])
    face = np.nan_to_num(seq["face"], nan=0.0)
    masks = np.stack([*present, pose_ok, face_ok], axis=1)

    feats = np.concatenate(
        [shapes[0], shapes[1], locs[0], locs[1], body.reshape(t, -1), face, masks.astype(np.float32)],
        axis=1,
    )
    return np.clip(feats, -CLIP, CLIP).astype(np.float32)


def movement_energy(feats: np.ndarray) -> float:
    """Mean frame-to-frame hand-location change (for the HUD)."""
    if len(feats) < 2:
        return 0.0
    return float(np.abs(np.diff(feats[:, OFF_LLOC:OFF_POSE], axis=0)).mean())


def resample(x: np.ndarray, n: int) -> np.ndarray:
    """Linear time-resample (T, D) -> (n, D)."""
    t = len(x)
    if t == 0:
        return np.zeros((n, x.shape[1] if x.ndim == 2 else FEATURE_DIM), np.float32)
    if t == 1:
        return np.repeat(x, n, axis=0)
    pos = np.linspace(0, t - 1, n)
    i0 = np.floor(pos).astype(int)
    i1 = np.minimum(i0 + 1, t - 1)
    w = (pos - i0)[:, None]
    return (x[i0] * (1 - w) + x[i1] * w).astype(np.float32)


def describe(feats: np.ndarray) -> np.ndarray:
    """Fixed-size descriptor (DESCRIPTOR_DIM,) of a variable-length feature clip.

    Time-normalised so fast and slow signers produce comparable vectors.
    """
    x = resample(feats, STEPS)
    vel = np.abs(np.diff(x, axis=0))
    parts = [x.mean(0), x.std(0), x.min(0), x.max(0)]
    parts += [c.mean(0) for c in np.array_split(x, 3)]
    parts += [vel.mean(0), x[0], x[-1]]
    return np.concatenate(parts).astype(np.float32)


def sequence_descriptor(seq: Mapping) -> np.ndarray:
    return describe(featurize(seq))
