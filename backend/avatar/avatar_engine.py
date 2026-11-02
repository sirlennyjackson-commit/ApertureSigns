import numpy as np


POSE_JOINTS = {
    0: "nose",
    11: "left_shoulder",
    12: "right_shoulder",
    13: "left_elbow",
    14: "right_elbow",
    15: "left_wrist",
    16: "right_wrist",
    23: "left_hip",
    24: "right_hip",
    25: "left_knee",
    26: "right_knee",
    27: "left_ankle",
    28: "right_ankle",
    31: "left_foot_index",
    32: "right_foot_index",
}


def _frame_from_landmarks(sequence: np.ndarray, frame_index: int, token: str) -> dict:
    pose = sequence[frame_index, 126:].reshape(33, 3)
    left = sequence[frame_index, :63].reshape(21, 3)
    right = sequence[frame_index, 63:126].reshape(21, 3)
    joints = {name: pose[index].tolist() for index, name in POSE_JOINTS.items()}

    return {
        "source_token": token,
        "joints": joints,
        "left_hand": (left + pose[15]).tolist(),
        "right_hand": (right + pose[16]).tolist(),
    }


def generate_avatar_sequence(sign_sequence: list, motion_sequences: list | None = None) -> dict:
    """Convert recorded landmark samples into frames for the desktop signing preview."""
    entries = sign_sequence or []
    samples = motion_sequences or []
    frames = []
    signed_words = []

    for index, item in enumerate(entries):
        sample = samples[index] if index < len(samples) else None
        if item.get("status") != "available" or sample is None:
            continue
        sequence = np.asarray(sample, dtype=np.float32)
        if sequence.shape != (30, 225) or not np.isfinite(sequence).all():
            continue
        token = str(item.get("gloss") or item.get("source_token") or "SIGN")
        signed_words.append(token)
        frames.extend(
            _frame_from_landmarks(sequence, frame_index, token)
            for frame_index in range(len(sequence))
        )

    mapped_count = sum(item.get("status") == "available" for item in entries)
    status = "ready" if entries and mapped_count == len(entries) and frames else "partial" if frames else "unavailable"
    return {
        "status": status,
        "speech_output": " ".join(signed_words),
        "animation": {
            "type": "recorded_landmark_playback",
            "frame_interval_ms": 80,
            "frames": frames,
        },
    }
