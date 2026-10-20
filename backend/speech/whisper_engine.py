"""Offline speech-to-text via faster-whisper (optional dependency).

    pip install -r requirements-stt.txt

Runs fully on your own server: audio never goes to a third party. Whisper
covers English and Swahili. Returns ``None`` from :func:`get_engine` when the
package is not installed, and the API answers 501.
"""
from __future__ import annotations

import io
from functools import lru_cache


class WhisperEngine:
    def __init__(self, model_name: str):
        from faster_whisper import WhisperModel  # heavy import, kept lazy

        self._model = WhisperModel(model_name, compute_type="int8")

    def transcribe(self, audio: bytes, language: str | None = None) -> str:
        segments, _info = self._model.transcribe(io.BytesIO(audio), language=language, vad_filter=True)
        return " ".join(s.text.strip() for s in segments).strip()


@lru_cache(maxsize=1)
def get_engine(model_name: str = "small") -> WhisperEngine | None:
    try:
        return WhisperEngine(model_name)
    except ImportError:
        return None
