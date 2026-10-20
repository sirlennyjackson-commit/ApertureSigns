import numpy as np
import pytest

from backend.ml.features import DESCRIPTOR_DIM, FEATURE_DIM, OFF_MASK, featurize, sequence_descriptor
from backend.ml.landmarks import (
    frame_from_payload,
    frames_to_sequence,
    hand_activity,
    load_sequence,
    mirror_sequence,
    save_sequence,
)
from tests.conftest import make_frames


def test_shapes(rng):
    seq = frames_to_sequence(make_frames("wave", rng))
    assert featurize(seq).shape == (45, FEATURE_DIM)
    assert sequence_descriptor(seq).shape == (DESCRIPTOR_DIM,)


def test_features_ignore_camera_position_and_distance(rng):
    base = featurize(frames_to_sequence(make_frames("wave", np.random.default_rng(1))))
    moved = featurize(frames_to_sequence(
        make_frames("wave", np.random.default_rng(1), shift=(0.08, -0.05), scale=0.8)))
    # Compare everything except the noisy mask columns.
    assert np.allclose(base[:, :OFF_MASK], moved[:, :OFF_MASK], atol=0.15)


def test_missing_hand_is_masked(rng):
    seq = frames_to_sequence(make_frames("idle", rng))
    feats = featurize(seq)
    assert hand_activity(seq) == 0.0
    assert feats[:, OFF_MASK:OFF_MASK + 2].sum() == 0  # no hand present
    assert feats[:, OFF_MASK + 2].all()                # pose present


def test_all_nan_sequence_does_not_crash():
    from backend.ml.landmarks import empty_sequence
    assert np.isfinite(featurize(empty_sequence(10))).all()


def test_mirror_is_involution(rng):
    seq = frames_to_sequence(make_frames("raise", rng))
    twice = mirror_sequence(mirror_sequence(seq))
    for key in seq:
        assert np.allclose(seq[key], twice[key], equal_nan=True, atol=1e-6)


def test_save_load_roundtrip(tmp_path, rng):
    seq = frames_to_sequence(make_frames("circle", rng))
    save_sequence(tmp_path / "a" / "x.npz", seq)
    back = load_sequence(tmp_path / "a" / "x.npz")
    for key in seq:
        assert np.allclose(seq[key], back[key], equal_nan=True)


def test_payload_validation():
    good = {"right": [[0.5, 0.5, 0.0]] * 21, "pose": [[0.5, 0.5, 0.0]] * 33}
    frame = frame_from_payload(good)
    assert frame["pose"].shape == (33, 4) and frame["left"] is None
    for bad in ({"right": [[0, 0, 0]] * 5}, {"right": [[float("nan")] * 3] * 21},
                {"pose": "oops"}, [1, 2, 3]):
        with pytest.raises(ValueError):
            frame_from_payload(bad)
