"""Load the v2 classifier and run inference on a saved dataset sequence."""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np

from backend.training.features_v2 import FEATURE_DIM, FEATURE_VERSION, extract


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = PROJECT_ROOT / "data" / "processed" / "sign_dataset.npz"
DEFAULT_MODEL = (
    PROJECT_ROOT / "models" / "sign" / "classifier" / "aperture_sign_v2.joblib"
)


def run_smoke_test(dataset_path=DEFAULT_DATASET, model_path=DEFAULT_MODEL):
    bundle = joblib.load(model_path)
    if not isinstance(bundle, dict):
        raise ValueError("Classifier bundle must be a dictionary")
    if bundle.get("feature_version") != FEATURE_VERSION:
        raise ValueError("Classifier feature version does not match runtime")

    model = bundle["model"]
    classes = bundle["classes"]
    if not callable(getattr(model, "predict_proba", None)):
        raise ValueError("Classifier does not provide predict_proba")

    expected_feature_count = getattr(model, "n_features_in_", FEATURE_DIM)
    if expected_feature_count != FEATURE_DIM:
        raise ValueError(
            f"Classifier expects {expected_feature_count} features; "
            f"runtime produces {FEATURE_DIM}"
        )

    with np.load(dataset_path, allow_pickle=False) as dataset:
        sequences = np.asarray(dataset["X"], dtype=np.float32)

    if sequences.ndim != 3 or sequences.shape[1:] != (30, 225):
        raise ValueError(
            f"Expected dataset shape (samples, 30, 225); got {sequences.shape}"
        )
    usable = np.flatnonzero(np.any(np.abs(sequences) > 1e-6, axis=(1, 2)))
    if not len(usable):
        raise ValueError("Dataset contains no non-empty landmark sequences")

    sequence_index = int(usable[0])
    features = extract(sequences[sequence_index])
    if features.shape != (FEATURE_DIM,):
        raise ValueError(
            f"Expected {FEATURE_DIM} features; runtime produced {features.shape}"
        )

    probabilities = np.asarray(model.predict_proba(features.reshape(1, -1)))
    if probabilities.shape != (1, len(classes)):
        raise ValueError(
            f"Expected probability shape (1, {len(classes)}); got {probabilities.shape}"
        )
    if not np.isfinite(probabilities).all() or not np.isclose(
        probabilities.sum(), 1.0, atol=1e-5
    ):
        raise ValueError("Classifier returned invalid probabilities")

    best_index = int(np.argmax(probabilities[0]))
    return {
        "status": "ok",
        "model": str(model_path),
        "dataset": str(dataset_path),
        "sequence_index": sequence_index,
        "feature_count": FEATURE_DIM,
        "classes": len(classes),
        "prediction": str(classes[best_index]),
        "confidence": float(probabilities[0, best_index]),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    args = parser.parse_args()
    print(json.dumps(run_smoke_test(args.dataset, args.model), indent=2))


if __name__ == "__main__":
    main()
