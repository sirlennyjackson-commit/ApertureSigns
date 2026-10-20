"""Landmark-space data augmentation.

Hand-collected sign datasets are tiny. These transforms (time warp, small
rotation / scale / shift, jitter, tracking dropouts, optional mirroring for
left-handed signers) multiply the effective data without recording more clips.
"""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from backend.ml.landmarks import mirror_sequence


def _time_warp(seq, rng, gamma=(0.75, 1.35), crop=0.1):
    t = len(seq["pose"])
    start = int(rng.uniform(0, crop) * t)
    end = max(t - int(rng.uniform(0, crop) * t), start + 2)
    u = np.linspace(0.0, 1.0, t) ** rng.uniform(*gamma)
    idx = np.clip(np.round(start + u * (end - 1 - start)).astype(int), 0, t - 1)
    return {k: v[idx] for k, v in seq.items()}  # fancy indexing => copies


def _geometry(seq, rng, rot_deg, scale, shift):
    theta = np.deg2rad(rng.uniform(-rot_deg, rot_deg))
    s = rng.uniform(1 - scale, 1 + scale)
    offset = rng.uniform(-shift, shift, size=2)
    c, n = np.cos(theta), np.sin(theta)
    rot = np.array([[c, -n], [n, c]], np.float32) * s
    for key in ("left", "right", "pose"):
        xy = seq[key][..., :2] - 0.5
        seq[key][..., :2] = xy @ rot.T + 0.5 + offset
        seq[key][..., 2] *= s


def augment_sequence(
    seq: Mapping,
    rng: np.random.Generator,
    *,
    mirror_p: float = 0.0,
    rot_deg: float = 10.0,
    scale: float = 0.1,
    shift: float = 0.05,
    jitter: float = 0.004,
    drop_p: float = 0.03,
):
    """Return a randomly perturbed copy of ``seq`` (the input is not modified)."""
    out = _time_warp(seq, rng)
    _geometry(out, rng, rot_deg, scale, shift)
    for key in ("left", "right", "pose"):
        out[key][..., :3] += rng.normal(0, jitter, out[key][..., :3].shape).astype(np.float32)
    dropped = rng.random(len(out["pose"])) < drop_p
    out["left"][dropped] = np.nan
    out["right"][dropped] = np.nan
    if mirror_p and rng.random() < mirror_p:
        out = mirror_sequence(out)
    return out
