"""Synthetic landmark generators so tests need no camera or MediaPipe."""
import numpy as np
import pytest

from backend.ml.landmarks import empty_frame

_TEMPLATE = np.random.default_rng(0).normal(0, 0.02, (21, 3)).astype(np.float32)
_TEMPLATE[0] = 0
_TEMPLATE[9] = [0.0, -0.06, 0.0]  # middle-finger base: gives the hand a real size


def make_pose(shift=(0.0, 0.0), scale=1.0):
    pose = np.zeros((33, 4), np.float32)
    pose[:, 3] = 1.0
    pose[:, 0], pose[:, 1] = 0.5, 0.6
    pose[0, :2] = [0.5, 0.35]                          # nose
    pose[11, :2], pose[12, :2] = [0.6, 0.5], [0.4, 0.5]  # shoulders
    pose[13, :2], pose[14, :2] = [0.65, 0.65], [0.35, 0.65]
    pose[15, :2], pose[16, :2] = [0.65, 0.8], [0.35, 0.8]
    pose[:, :2] = (pose[:, :2] - 0.5) * scale + 0.5 + np.array(shift, np.float32)
    return pose


def make_hand(center, scale=1.0):
    hand = _TEMPLATE * scale
    hand[:, :2] += np.array(center, np.float32)
    return hand


def path(kind, i, n):
    u = i / max(n - 1, 1)
    if kind == "wave":
        return 0.65 + 0.12 * np.sin(u * 4 * np.pi), 0.35
    if kind == "raise":
        return 0.62, 0.8 - 0.5 * u
    if kind == "circle":
        return 0.62 + 0.1 * np.cos(u * 2 * np.pi), 0.5 + 0.1 * np.sin(u * 2 * np.pi)
    raise ValueError(kind)


def make_frames(kind, rng, n=45, shift=(0.0, 0.0), scale=1.0):
    frames = []
    for i in range(n):
        f = empty_frame()
        f["pose"] = make_pose(shift, scale)
        if kind != "idle":
            x, y = path(kind, i, n)
            cx = (x - 0.5) * scale + 0.5 + shift[0]
            cy = (y - 0.5) * scale + 0.5 + shift[1]
            hand = make_hand((cx, cy), scale)
            hand += rng.normal(0, 0.002, hand.shape).astype(np.float32)
            f["right"] = hand
        frames.append(f)
    return frames


@pytest.fixture
def rng():
    return np.random.default_rng(42)
