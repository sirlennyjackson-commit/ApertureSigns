from fastapi.testclient import TestClient

from backend.main import app
from backend.training.features_v2 import FEATURE_DIM, extract
from backend.training.smoke_test import run_smoke_test
from backend.vision.sign_recognizer import SignRecognizer


client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_pipeline_returns_recorded_vocabulary_playback():
    response = client.post("/pipeline", json={"text": "  me   have "})

    assert response.status_code == 200
    result = response.json()
    assert result["speech"] == "me have"
    assert result["translation"]["status"] == "vocabulary_only"
    assert result["translation"]["is_translation"] is False
    assert "ME" in result["translation"]["supported_vocabulary"]
    assert [item["status"] for item in result["translation"]["sign_sequence"]] == [
        "available",
        "available",
    ]
    assert result["avatar"]["status"] == "ready"
    assert result["avatar"]["animation"]["type"] == "recorded_landmark_playback"
    assert len(result["avatar"]["animation"]["frames"]) == 60
    assert "joints" in result["avatar"]["animation"]["frames"][0]


def test_pipeline_marks_unknown_vocabulary_instead_of_faking_a_sign():
    response = client.post("/pipeline", json={"text": "me unlistedword"})

    assert response.status_code == 200
    result = response.json()
    assert result["translation"]["status"] == "partial"
    assert [item["status"] for item in result["translation"]["sign_sequence"]] == [
        "available",
        "unmapped",
    ]
    assert result["avatar"]["status"] == "partial"


def test_v2_model_runs_inference_on_saved_sample():
    result = run_smoke_test()

    assert result["status"] == "ok"
    assert result["feature_count"] == FEATURE_DIM
    assert result["classes"] == 60
    assert 0.0 <= result["confidence"] <= 1.0


def test_recognizer_reports_missing_model_without_crashing(tmp_path):
    recognizer = SignRecognizer(model_path=tmp_path / "missing.joblib")

    assert recognizer.model is None
    assert recognizer.model_error
    assert recognizer.analyze([])["prediction"]["status"] == "unavailable"


def test_empty_sequence_features_have_expected_size():
    import numpy as np

    features = extract(np.zeros((30, 225), dtype=np.float32))

    assert features.shape == (FEATURE_DIM,)
