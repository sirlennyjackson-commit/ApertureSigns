"""Central, environment-driven configuration.

Every value can be overridden with an ``APERTURE_*`` environment variable,
e.g. ``APERTURE_MODEL_PATH=/srv/models/ksl.npz``.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# Labels that mean "resting / not signing". Recorded as a normal class so the
# model learns what *not* signing looks like.
IDLE_LABELS = ("_idle",)


def _path(name: str, default: Path) -> Path:
    value = os.getenv(name)
    return Path(value).expanduser() if value else default


def _num(name: str, default: float, cast=float):
    value = os.getenv(name)
    return cast(value) if value else default


@dataclass(frozen=True)
class Settings:
    model_path: Path
    data_dir: Path
    window_frames: int
    stride: int
    min_confidence: float
    min_votes: int
    min_hand_activity: float
    cors_origins: tuple[str, ...]
    enable_collect: bool
    max_message_bytes: int
    frontend_dir: Path
    whisper_model: str


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    origins = os.getenv(
        "APERTURE_CORS_ORIGINS", "http://localhost:3000,http://localhost:5173"
    )
    return Settings(
        model_path=_path("APERTURE_MODEL_PATH", ROOT / "models" / "signs.npz"),
        data_dir=_path("APERTURE_DATA_DIR", ROOT / "data"),
        # ~1.5 s of signing at 30 fps. Keep in sync with `collect --seconds`.
        window_frames=_num("APERTURE_WINDOW_FRAMES", 45, int),
        stride=_num("APERTURE_STRIDE", 3, int),
        min_confidence=_num("APERTURE_MIN_CONFIDENCE", 0.75),
        min_votes=_num("APERTURE_MIN_VOTES", 2, int),
        min_hand_activity=_num("APERTURE_MIN_HAND_ACTIVITY", 0.3),
        cors_origins=tuple(o.strip() for o in origins.split(",") if o.strip()),
        # Browser recording writes files to disk: opt-in only.
        enable_collect=os.getenv("APERTURE_ENABLE_COLLECT", "0").lower() in ("1", "true", "yes"),
        max_message_bytes=_num("APERTURE_MAX_MESSAGE_BYTES", 65536, int),
        frontend_dir=_path("APERTURE_FRONTEND_DIR", ROOT / "frontend"),
        whisper_model=os.getenv("APERTURE_WHISPER_MODEL", "small"),
    )
