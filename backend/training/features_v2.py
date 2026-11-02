"""
Aperture Signs feature extractor v2.

Input : (T, 225) sequence in the existing layout
        [left hand 63 (wrist-relative) | right hand 63 (wrist-relative) | pose 99 (absolute)]
Output: fixed-length vector (704,)
"""
import numpy as np

FEATURE_VERSION = 2
POSE_SUB = [0, 11, 12, 13, 14, 15, 16]      # nose, shoulders, elbows, wrists
OUT_FRAMES = 30
D = 63 + 63 + len(POSE_SUB) * 2             # 140 per frame
FEATURE_DIM = 5 * D + 4                     # 704


def _present(a):
    return np.abs(a.reshape(len(a), -1)).sum(axis=1) > 1e-6


def extract(seq):
    seq = np.asarray(seq, dtype=np.float32)
    T = seq.shape[0]
    left = seq[:, 0:63].reshape(T, 21, 3)
    right = seq[:, 63:126].reshape(T, 21, 3)
    pose = seq[:, 126:225].reshape(T, 33, 3)

    lp, rp, pp = _present(left), _present(right), _present(pose)
    active = lp | rp | pp
    if not active.any():
        return np.zeros(FEATURE_DIM, dtype=np.float32)

    idx = np.nonzero(active)[0]
    sel = np.linspace(idx[0], idx[-1], OUT_FRAMES).round().astype(int)
    left, right, pose = left[sel], right[sel], pose[sel]
    lp, rp, pp = lp[sel], rp[sel], pp[sel]

    # body normalisation (one centre/scale for the whole sign)
    width = np.linalg.norm(pose[:, 11, :2] - pose[:, 12, :2], axis=1)
    ctr = (pose[:, 11, :2] + pose[:, 12, :2]) / 2
    good = pp & (width > 1e-3)
    w = float(np.median(width[good])) if good.any() else 1.0
    c = np.median(ctr[good], axis=0) if good.any() else np.zeros(2, dtype=np.float32)

    ps = (pose[:, POSE_SUB, :2] - c) / w
    ps[~pp] = 0

    def hand_n(h, present):
        scale = np.linalg.norm(h[:, 9, :2], axis=1)
        s = float(np.median(scale[present])) if present.any() else 1.0
        s = max(s, 1e-3)
        out = h / s
        out[~present] = 0
        return out.reshape(OUT_FRAMES, 63)

    F = np.concatenate(
        [hand_n(left, lp), hand_n(right, rp), ps.reshape(OUT_FRAMES, -1)], axis=1
    )

    mask = np.concatenate(
        [np.repeat(lp[:, None], 63, 1), np.repeat(rp[:, None], 63, 1),
         np.repeat(pp[:, None], len(POSE_SUB) * 2, 1)], axis=1)
    vm = mask[1:] & mask[:-1]
    V = np.diff(F, axis=0) * vm

    thirds = [F[i].mean(axis=0) for i in np.array_split(np.arange(OUT_FRAMES), 3)]

    def path(pts, present):
        ok = present[1:] & present[:-1]
        return float((np.linalg.norm(np.diff(pts, axis=0), axis=1) * ok).sum())

    extra = np.array([lp.mean(), rp.mean(), path(ps[:, 5], pp), path(ps[:, 6], pp)],
                     dtype=np.float32)

    return np.concatenate(
        thirds + [F.std(axis=0), np.abs(V).mean(axis=0), extra]
    ).astype(np.float32)


def extract_many(X):
    return np.stack([extract(s) for s in X])