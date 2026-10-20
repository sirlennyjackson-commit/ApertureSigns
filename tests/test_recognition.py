import numpy as np
import pytest

pytest.importorskip("sklearn")

from backend.ml.augment import augment_sequence
from backend.ml.classifier import SignClassifier
from backend.ml.landmarks import frames_to_sequence
from backend.ml.realtime import RealtimeRecognizer
from backend.training.train import build_xy, fit
from tests.conftest import make_frames

KINDS = ["wave", "raise", "circle", "idle"]
IDLE = "_idle"


def _label(kind):
    return IDLE if kind == "idle" else kind


@pytest.fixture(scope="module")
def model():
    rng = np.random.default_rng(7)
    seqs, labels = [], []
    for kind in KINDS:
        for _ in range(6):
            seqs.append(frames_to_sequence(make_frames(kind, rng, n=int(rng.integers(35, 55)))))
            labels.append(_label(kind))
    X, y = build_xy(seqs, labels, augment=4, mirror_p=0.0, rng=rng)
    return fit(X, y)


def test_unseen_clips_are_classified(model):
    rng = np.random.default_rng(123)
    for kind in KINDS:
        seq = frames_to_sequence(make_frames(kind, rng, n=47, shift=(0.03, 0.02), scale=0.9))
        pred = model.predict(seq)
        assert pred.label == _label(kind), (kind, pred)
        assert pred.confidence > 0.6


def test_numpy_inference_matches_sklearn(model):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    rng = np.random.default_rng(5)
    seqs = [frames_to_sequence(make_frames(k, rng)) for k in KINDS for _ in range(4)]
    labels = [_label(k) for k in KINDS for _ in range(4)]
    X, y = build_xy(seqs, labels, augment=0, mirror_p=0, rng=rng)
    scaler = StandardScaler().fit(X)
    sk = LogisticRegression(C=0.1, max_iter=3000).fit(scaler.transform(X), y)
    ours = SignClassifier(sk.classes_, scaler.mean_, scaler.scale_, sk.coef_, sk.intercept_)
    assert np.allclose(ours.predict_proba(X), sk.predict_proba(scaler.transform(X)), atol=1e-4)


def test_binary_model_probabilities():
    from sklearn.linear_model import LogisticRegression
    rng = np.random.default_rng(1)
    seqs = [frames_to_sequence(make_frames(k, rng)) for k in ("wave", "raise") for _ in range(5)]
    labels = ["wave"] * 5 + ["raise"] * 5
    X, y = build_xy(seqs, labels, augment=0, mirror_p=0, rng=rng)
    sk = LogisticRegression(max_iter=3000).fit(X, y)
    ours = SignClassifier(sk.classes_, np.zeros(X.shape[1]), np.ones(X.shape[1]), sk.coef_, sk.intercept_)
    assert np.allclose(ours.predict_proba(X), sk.predict_proba(X), atol=1e-4)


def test_model_save_load(tmp_path, model):
    path = model.save(tmp_path / "m.npz")
    again = SignClassifier.load(path)
    seq = frames_to_sequence(make_frames("wave", np.random.default_rng(9)))
    assert again.predict(seq).label == model.predict(seq).label == "wave"


def test_outdated_model_is_rejected(tmp_path, model):
    path = model.save(tmp_path / "m.npz")
    data = dict(np.load(path))
    data["meta"] = np.array('{"feature_version": 0}')
    np.savez(tmp_path / "old.npz", **data)
    with pytest.raises(ValueError, match="retrain"):
        SignClassifier.load(tmp_path / "old.npz")


def test_augmentation_does_not_mutate_input():
    rng = np.random.default_rng(0)
    seq = frames_to_sequence(make_frames("wave", rng))
    before = {k: v.copy() for k, v in seq.items()}
    out = augment_sequence(seq, rng, mirror_p=1.0)
    for k in seq:
        assert np.array_equal(seq[k], before[k], equal_nan=True)
    assert out["right"].shape[1:] == (21, 3)


def test_realtime_emits_once_per_sign(model):
    rng = np.random.default_rng(11)
    rec = RealtimeRecognizer(model, window=45, stride=3, min_confidence=0.6, min_votes=2)
    events = [rec.push(f) for f in make_frames("wave", rng, n=90)]
    labels = [e.label for e in events if e]
    assert labels == ["wave"]  # confirmed once, not spammed


def test_realtime_stays_quiet_when_idle(model):
    rng = np.random.default_rng(12)
    rec = RealtimeRecognizer(model, window=45, stride=3, min_confidence=0.6)
    assert not any(rec.push(f) for f in make_frames("idle", rng, n=90))
    assert rec.state()["hands"] is False
