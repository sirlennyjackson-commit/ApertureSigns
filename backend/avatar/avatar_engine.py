"""Sign sequence -> avatar timeline.

Produces a timed list of keyframes a renderer can play. ``clip`` is ``None``
until motion-capture / animation assets exist for a gloss; the frontend can
fall back to showing the gloss text meanwhile.
"""
from __future__ import annotations

SIGN_MS = 900
NUMBER_MS = 600
LETTER_MS = 350
GAP_MS = 120


def _normalise(item) -> dict:
    if isinstance(item, str):  # legacy contract: plain gloss strings
        return {"gloss": item, "type": "lexical"}
    return dict(item)


def generate_avatar_sequence(sign_sequence: list) -> dict:
    timeline, cursor = [], 0
    for raw in sign_sequence:
        item = _normalise(raw)
        kind = item.get("type", "lexical")
        letters = item.get("letters")
        if kind == "fingerspelling" and letters:
            duration = LETTER_MS * len(letters)
        elif kind == "number":
            duration = NUMBER_MS
        else:
            duration = SIGN_MS
        timeline.append({
            "gloss": item.get("gloss"),
            "type": kind,
            "letters": letters,
            "start_ms": cursor,
            "duration_ms": duration,
            "clip": None,
        })
        cursor += duration + GAP_MS
    return {
        "status": "timeline_ready" if timeline else "empty",
        "sign_sequence": sign_sequence,
        "animation": None,
        "timeline": timeline,
        "total_ms": max(cursor - GAP_MS, 0),
    }
