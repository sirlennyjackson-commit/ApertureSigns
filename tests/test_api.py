import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.main import app
from tests.conftest import make_frames


def _payload(frame):
    out = {}
    for key in ("left", "right", "pose"):
        out[key] = None if frame[key] is None else frame[key].tolist()
    return out


@pytest.fixture
def client():
    with TestClient(app) as c:
        app.state.classifier = None
        yield c


@pytest.fixture
def trained():
    pytest.importorskip("sklearn")
    from backend.ml.landmarks import frames_to_sequence
    from backend.training.train import build_xy, fit
    rng = np.random.default_rng(3)
    seqs, labels = [], []
    for kind in ("wave", "raise", "idle"):
        for _ in range(5):
            seqs.append(frames_to_sequence(make_frames(kind, rng)))
            labels.append("_idle" if kind == "idle" else kind)
    X, y = build_xy(seqs, labels, augment=3, mirror_p=0, rng=rng)
    return fit(X, y)


def test_basic_endpoints(client):
    assert client.get("/").json()["status"] == "online"
    health = client.get("/health").json()
    assert health["status"] == "healthy" and health["model_loaded"] is False


def test_pipeline_contract(client):
    body = client.post("/pipeline", json={"text": "  hello   Wanjiku "}).json()
    assert set(body) == {"speech", "translation", "avatar"}
    assert body["speech"] == "hello Wanjiku"
    assert body["avatar"]["timeline"]


def test_pipeline_validation(client):
    assert client.post("/pipeline", json={"text": ""}).status_code == 422
    assert client.post("/pipeline", json={"text": "x" * 1001}).status_code == 422


def test_recognition_needs_model(client):
    assert client.post("/recognize", json={"frames": [{}] * 10}).status_code == 503
    with client.websocket_connect("/ws/recognize") as ws:
        assert ws.receive_json()["detail"] == "model_not_loaded"


def test_recognize_endpoint(client, trained):
    app.state.classifier = trained
    frames = [_payload(f) for f in make_frames("raise", np.random.default_rng(8))]
    body = client.post("/recognize", json={"frames": frames}).json()
    assert body["label"] == "raise"
    bad = client.post("/recognize", json={"frames": [{"right": [[0, 0, 0]]}] * 10})
    assert bad.status_code == 422
    no_hands = [_payload(f) for f in make_frames("idle", np.random.default_rng(8))]
    assert client.post("/recognize", json={"frames": no_hands}).json()["reason"] == "no_hands_detected"


def test_websocket_stream(client, trained):
    app.state.classifier = trained
    with client.websocket_connect("/ws/recognize") as ws:
        ready = ws.receive_json()
        assert ready["type"] == "ready" and "wave" in ready["labels"]
        ws.send_text("not json")
        assert ws.receive_json()["type"] == "error"
        ws.send_json({"right": [[1, 2]]})
        assert ws.receive_json()["type"] == "error"
        signs = []
        for frame in make_frames("wave", np.random.default_rng(4), n=90):
            ws.send_json(_payload(frame))
        ws.send_json({"type": "reset"})
        seen = []
        while True:
            msg = ws.receive_json()
            seen.append(msg)
            if msg["type"] == "reset":
                break
        signs = [m["label"] for m in seen if m["type"] == "sign"]
        assert signs == ["wave"]


# ---------------------------------------------------------------- v0.3 ----
def _clip(kind, seed=0, n=20):
    return [_payload(f) for f in make_frames(kind, np.random.default_rng(seed), n=n)]


@pytest.fixture
def collect_on(tmp_path, monkeypatch):
    from backend.config import settings as cfg
    monkeypatch.setenv("APERTURE_ENABLE_COLLECT", "1")
    monkeypatch.setenv("APERTURE_DATA_DIR", str(tmp_path))
    cfg.get_settings.cache_clear()
    yield tmp_path
    cfg.get_settings.cache_clear()


def test_collect_disabled_by_default(client):
    body = {"label": "hi", "signer": "a", "consent": True, "frames": _clip("wave")}
    assert client.post("/collect", json=body).status_code == 403
    assert client.get("/collect/stats").status_code == 403


def test_collect_saves_clips(client, collect_on):
    body = {"label": "hello", "signer": "amina", "consent": True, "frames": _clip("wave")}
    first = client.post("/collect", json=body).json()
    assert first["count"] == 1 and first["saved"].startswith("amina__")
    client.post("/collect", json=body)
    assert client.get("/collect/stats").json() == {"labels": {"hello": 2}, "signers": ["amina"]}
    from backend.ml.landmarks import load_sequence
    assert len(load_sequence(next((collect_on / "sequences" / "hello").glob("*.npz")))["pose"]) == 20


def test_collect_rejects_bad_requests(client, collect_on):
    good = {"label": "hello", "signer": "amina", "consent": True, "frames": _clip("wave")}
    assert client.post("/collect", json={**good, "consent": False}).status_code == 422
    for bad_label in ("../etc", "a/b", "", "x" * 41):
        assert client.post("/collect", json={**good, "label": bad_label}).status_code == 422
    assert client.post("/collect", json={**good, "signer": "a__b"}).status_code == 422
    assert client.post("/collect", json={**good, "frames": _clip("idle")}).status_code == 422  # no hands
    idle = {**good, "label": "_idle", "frames": _clip("idle")}
    assert client.post("/collect", json=idle).status_code == 200  # resting is allowed for _idle


def test_stt_not_installed(client, monkeypatch):
    import backend.main as m
    monkeypatch.setattr(m, "get_engine", lambda *_: None)
    assert client.post("/stt", content=b"abc").status_code == 501


def test_websocket_rejects_oversized_and_binary(client, trained, monkeypatch):
    from backend.config import settings as cfg
    monkeypatch.setenv("APERTURE_MAX_MESSAGE_BYTES", "200")
    cfg.get_settings.cache_clear()
    app.state.classifier = trained
    try:
        with client.websocket_connect("/ws/recognize") as ws:
            ws.receive_json()
            ws.send_text("x" * 500)
            assert ws.receive_json()["detail"] == "message too large"
            ws.send_bytes(b"\x00")
            assert ws.receive_json()["type"] == "error"
    finally:
        cfg.get_settings.cache_clear()


def test_frontend_is_served(client):
    assert client.get("/app/").status_code == 200
