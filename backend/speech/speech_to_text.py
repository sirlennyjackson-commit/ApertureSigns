"""Speech -> text.

Still a pass-through: the API receives text. The seam below is where a real
engine plugs in. For an accessibility product prefer an *offline* engine
(e.g. faster-whisper, which supports Swahili and English) so audio never has to
leave the device or server.
"""
from __future__ import annotations

import re
from typing import Protocol


class SpeechEngine(Protocol):
    def transcribe(self, audio: bytes, language: str | None = None) -> str: ...


def normalize_transcript(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def speech_to_text(audio_text: str) -> str:
    return normalize_transcript(audio_text)
