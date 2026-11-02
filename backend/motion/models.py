"""Motion frames used by the desktop pipeline and Blender bridge."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


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
}


def _xyz(point: dict | None) -> list[float] | None:
    if not point:
        return None
    return [float(point["x"]), float(point["y"]), float(point["z"])]


@dataclass
class MotionFrame:
    timestamp: float
    sequence: int
    joints: dict[str, list[float]] = field(default_factory=dict)
    left_hand: list[list[float]] = field(default_factory=list)
    right_hand: list[list[float]] = field(default_factory=list)
    face: list[list[float]] = field(default_factory=list)
    landmark_count: int = 0
    hands: int = 0
    pose_confidence: float | None = None
    motion_energy: float = 0.0

    @classmethod
    def from_landmarks(
        cls,
        landmarks: dict,
        timestamp: float,
        sequence: int,
        previous: "MotionFrame | None" = None,
    ) -> "MotionFrame":
        pose = landmarks.get("pose") or []
        left = landmarks.get("left_hand") or []
        right = landmarks.get("right_hand") or []
        face = landmarks.get("face") or []

        joints: dict[str, list[float]] = {}
        visibilities: list[float] = []
        for index, name in POSE_JOINTS.items():
            if index < len(pose):
                joints[name] = _xyz(pose[index])  # type: ignore[assignment]
                if "visibility" in pose[index]:
                    visibilities.append(float(pose[index]["visibility"]))

        left_xyz = [xyz for point in left if (xyz := _xyz(point))]
        right_xyz = [xyz for point in right if (xyz := _xyz(point))]
        face_xyz = [xyz for point in face if (xyz := _xyz(point))]

        energy = 0.0
        if previous is not None and previous.joints and joints:
            deltas = []
            for name, xyz in joints.items():
                old = previous.joints.get(name)
                if old and xyz:
                    deltas.append(sum(abs(a - b) for a, b in zip(xyz, old)) / 3.0)
            if deltas:
                energy = sum(deltas) / len(deltas)

        confidence = (sum(visibilities) / len(visibilities) * 100.0) if visibilities else None
        landmark_count = len(pose) + len(left) + len(right) + len(face)
        return cls(
            timestamp=timestamp,
            sequence=sequence,
            joints=joints,
            left_hand=left_xyz,  # type: ignore[arg-type]
            right_hand=right_xyz,  # type: ignore[arg-type]
            face=face_xyz,  # type: ignore[arg-type]
            landmark_count=landmark_count,
            hands=int(bool(left)) + int(bool(right)),
            pose_confidence=confidence,
            motion_energy=energy,
        )

    def overlay(self, source: dict, include_face: bool = False) -> dict:
        """Normalized points for the live overlay canvases."""
        payload = {
            "pose": source.get("pose") or [],
            "left_hand": source.get("left_hand") or [],
            "right_hand": source.get("right_hand") or [],
        }
        if include_face:
            payload["face"] = source.get("face") or []
        return payload

    def to_blender(self) -> dict[str, Any]:
        return {
            "v": 1,
            "source": "aperture-signs",
            "t": self.timestamp,
            "seq": self.sequence,
            "joints": self.joints,
            "left_hand": self.left_hand,
            "right_hand": self.right_hand,
            "confidence": self.pose_confidence,
            "energy": self.motion_energy,
        }

    def preview(self) -> dict[str, Any]:
        return {
            "sequence": self.sequence,
            "timestamp": self.timestamp,
            "joints": self.joints,
            "left_hand": self.left_hand,
            "right_hand": self.right_hand,
            "landmark_count": self.landmark_count,
            "hands": self.hands,
            "confidence": self.pose_confidence,
            "energy": self.motion_energy,
        }
