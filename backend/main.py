"""Aperture Signs API.

    uvicorn backend.main:app --reload        # then open http://localhost:8000/app/
"""
from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI, HTTPException, Request, WebSocket
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from backend.avatar.avatar_engine import generate_avatar_sequence
from backend.config.settings import IDLE_LABELS, get_settings
from backend.ml.classifier import SignClassifier
from backend.ml.landmarks import frame_from_payload, frames_to_sequence, hand_activity, save_sequence
from backend.ml.realtime import RealtimeRecognizer
from backend.speech.speech_to_text import speech_to_text
from backend.speech.whisper_engine import get_engine
from backend.translation.ksl_translator import load_lexicon, translate_to_ksl

VERSION = "0.3.0"
MAX_AUDIO_BYTES = 10 * 1024 * 1024
log = logging.getLogger("aperture")


def load_model(app: FastAPI) -> SignClassifier | None:
    """(Re)load the trained model from disk into ``app.state.classifier``."""
    path = get_settings().model_path
    app.state.classifier = None
    if not path.exists():
        log.warning("No model at %s: recognition endpoints return 503 until you train one", path)
        return None
    try:
        app.state.classifier = SignClassifier.load(path)
        log.info("Loaded model %s (%d labels)", path, len(app.state.classifier.labels))
    except Exception:  # a corrupt or outdated model must not stop the API
        log.exception("Could not load model %s", path)
    return app.state.classifier


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model(app)
    yield


app = FastAPI(
    title="Aperture Signs API",
    description="Real-time sign language accessibility platform",
    version=VERSION,
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(get_settings().cors_origins),
    allow_methods=["*"],
    allow_headers=["*"],
)


class PipelineRequest(BaseModel):
    text: str = Field(min_length=1, max_length=1000)


class RecognizeRequest(BaseModel):
    frames: list[dict] = Field(min_length=8, max_length=300)


class CollectRequest(BaseModel):
    # `signer` has no underscore because the file name is "<signer>__<time>.npz".
    label: str = Field(pattern=r"^[A-Za-z0-9_-]{1,40}$")
    signer: str = Field(pattern=r"^[A-Za-z0-9-]{1,40}$")
    consent: bool
    frames: list[dict] = Field(min_length=8, max_length=300)


def _classifier(app_: FastAPI) -> SignClassifier:
    clf = app_.state.classifier
    if clf is None:
        raise HTTPException(status_code=503, detail="No trained model loaded. See README: collect -> train.")
    return clf


def _require_collect() -> None:
    if not get_settings().enable_collect:
        raise HTTPException(status_code=403, detail="Recording is disabled. Set APERTURE_ENABLE_COLLECT=1.")


def _parse_frames(raw: list[dict]) -> dict:
    try:
        return frames_to_sequence([frame_from_payload(f) for f in raw])
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


# --------------------------------------------------------------------------- #
# Basics
# --------------------------------------------------------------------------- #
@app.get("/")
def root():
    return {"name": "Aperture Signs", "status": "online", "version": VERSION}


@app.get("/health")
def health():
    clf = app.state.classifier
    return {
        "status": "healthy",
        "service": "aperture-signs-backend",
        "model_loaded": clf is not None,
        "model_labels": len(clf.labels) if clf else 0,
        "collect_enabled": get_settings().enable_collect,
    }


@app.get("/signs")
def signs():
    clf = app.state.classifier
    return {"recognizable": clf.labels if clf else [], "lexicon_size": len(load_lexicon())}


# --------------------------------------------------------------------------- #
# Text / speech -> signs
# --------------------------------------------------------------------------- #
@app.post("/pipeline")
def pipeline(request: PipelineRequest):
    spoken_text = speech_to_text(request.text)
    translation = translate_to_ksl(spoken_text)
    avatar = generate_avatar_sequence(translation["sign_sequence"])
    return {"speech": spoken_text, "translation": translation, "avatar": avatar}


@app.post("/stt")
async def stt(request: Request, language: str | None = None):
    """Offline transcription of an uploaded audio body (webm/wav/mp3...)."""
    engine = await run_in_threadpool(get_engine, get_settings().whisper_model)
    if engine is None:
        raise HTTPException(
            status_code=501, detail="Speech-to-text not installed: pip install -r requirements-stt.txt"
        )
    if int(request.headers.get("content-length") or 0) > MAX_AUDIO_BYTES:
        raise HTTPException(status_code=413, detail="Audio too large (10 MB max)")
    audio = await request.body()
    if not audio:
        raise HTTPException(status_code=400, detail="Empty request body")
    if len(audio) > MAX_AUDIO_BYTES:
        raise HTTPException(status_code=413, detail="Audio too large (10 MB max)")
    text = await run_in_threadpool(engine.transcribe, audio, language)
    return {"text": speech_to_text(text)}


# --------------------------------------------------------------------------- #
# Signs -> text
# --------------------------------------------------------------------------- #
@app.post("/recognize")
def recognize(request: RecognizeRequest):
    """Classify ONE isolated sign from a short clip of landmark frames."""
    clf = _classifier(app)
    seq = _parse_frames(request.frames)
    if hand_activity(seq) < get_settings().min_hand_activity:
        return {"label": None, "confidence": 0.0, "reason": "no_hands_detected", "top": []}
    return clf.predict(seq).as_dict()


@app.websocket("/ws/recognize")
async def ws_recognize(ws: WebSocket):
    """Streaming recognition. See docs/protocol.md for the message format."""
    await ws.accept()
    clf = ws.app.state.classifier
    if clf is None:
        await ws.send_json({"type": "error", "detail": "model_not_loaded"})
        await ws.close(code=1011)
        return

    s = get_settings()
    recognizer = RealtimeRecognizer(
        clf, window=s.window_frames, stride=s.stride, min_confidence=s.min_confidence,
        min_votes=s.min_votes, min_activity=s.min_hand_activity,
    )
    await ws.send_json({"type": "ready", "labels": clf.labels, "window": s.window_frames})
    n = 0
    while True:
        raw = await ws.receive()
        if raw["type"] == "websocket.disconnect":
            break
        text = raw.get("text")
        if text is None:
            await ws.send_json({"type": "error", "detail": "send JSON text messages"})
            continue
        if len(text) > s.max_message_bytes:
            await ws.send_json({"type": "error", "detail": "message too large"})
            continue
        try:
            message = json.loads(text)
        except json.JSONDecodeError:
            await ws.send_json({"type": "error", "detail": "invalid JSON"})
            continue
        if isinstance(message, dict) and message.get("type") == "reset":
            recognizer.reset()
            await ws.send_json({"type": "reset"})
            continue
        try:
            frame = frame_from_payload(message)
        except ValueError as exc:
            await ws.send_json({"type": "error", "detail": str(exc)})
            continue
        event = recognizer.push(frame)
        n += 1
        if event:
            await ws.send_json({"type": "sign", **event.as_dict()})
        if n % 5 == 0:
            await ws.send_json({"type": "state", **recognizer.state()})


# --------------------------------------------------------------------------- #
# Data collection from the browser (opt-in: APERTURE_ENABLE_COLLECT=1)
# --------------------------------------------------------------------------- #
@app.post("/collect")
def collect(request: CollectRequest):
    _require_collect()
    if not request.consent:
        raise HTTPException(status_code=422, detail="The signer must consent to recording.")
    s = get_settings()
    seq = _parse_frames(request.frames)
    if request.label not in IDLE_LABELS and hand_activity(seq) < s.min_hand_activity:
        raise HTTPException(status_code=422, detail="Hands were not visible enough; clip discarded.")
    label_dir = s.data_dir / "sequences" / request.label
    name = f"{request.signer}__{datetime.now():%Y%m%d-%H%M%S-%f}.npz"
    save_sequence(label_dir / name, seq)
    return {"saved": name, "label": request.label, "count": len(list(label_dir.glob("*.npz")))}


@app.get("/collect/stats")
def collect_stats():
    _require_collect()
    root = get_settings().data_dir / "sequences"
    labels: dict[str, int] = {}
    signers: set[str] = set()
    for path in root.glob("*/*.npz") if root.exists() else []:
        labels[path.parent.name] = labels.get(path.parent.name, 0) + 1
        signers.add(path.stem.split("__", 1)[0])
    return {"labels": dict(sorted(labels.items())), "signers": sorted(signers)}


@app.post("/model/reload")
def model_reload():
    """Pick up a freshly trained model without restarting the server."""
    _require_collect()
    clf = load_model(app)
    return {"model_loaded": clf is not None, "labels": clf.labels if clf else []}


# Static web client (frontend/). Mounted last so API routes always win.
_frontend = get_settings().frontend_dir
if _frontend.exists():
    app.mount("/app", StaticFiles(directory=_frontend, html=True), name="app")
