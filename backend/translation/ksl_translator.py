from functools import lru_cache
from pathlib import Path
import re

import numpy as np


DATASET_PATH = Path(__file__).resolve().parents[2] / "data" / "processed" / "sign_dataset.npz"


@lru_cache(maxsize=1)
def _load_sign_samples() -> dict[str, np.ndarray]:
    if not DATASET_PATH.is_file():
        return {}

    samples: dict[str, np.ndarray] = {}
    with np.load(DATASET_PATH, allow_pickle=True) as dataset:
        sequences = dataset["X"]
        labels = dataset["labels"]
        label_ids = dataset["y"]
        for sample_index, label_index in enumerate(label_ids):
            label = str(labels[int(label_index)]).strip().upper()
            sequence = np.asarray(sequences[sample_index], dtype=np.float32)
            if label.startswith(".") or label in samples:
                continue
            if sequence.shape == (30, 225) and np.isfinite(sequence).all():
                samples[label] = sequence
    return samples


def supported_vocabulary() -> list[str]:
    return sorted(_load_sign_samples())


def translate_to_ksl(text: str) -> dict:
    """Look up exact words in the recorded vocabulary; this is not grammar translation."""
    tokens = re.findall(r"[\w']+", text, flags=re.UNICODE)
    samples = _load_sign_samples()
    sign_sequence = []
    motion_sequences = []

    for token in tokens:
        label = token.upper()
        sequence = samples.get(label)
        available = sequence is not None
        sign_sequence.append(
            {
                "source_token": token,
                "status": "available" if available else "unmapped",
                "gloss": label if available else None,
            }
        )
        motion_sequences.append(sequence.tolist() if available else None)

    mapped_count = sum(item["status"] == "available" for item in sign_sequence)
    status = (
        "vocabulary_only" if sign_sequence and mapped_count == len(sign_sequence)
        else "partial" if mapped_count
        else "unavailable"
    )
    return {
        "input_text": text,
        "language": "KSL",
        "status": status,
        "is_translation": False,
        "message": "Exact word lookup in recorded sign samples; English-to-KSL grammar translation is not available.",
        "sign_sequence": sign_sequence,
        "supported_vocabulary": sorted(samples),
        "_motion_sequences": motion_sequences,
    }
