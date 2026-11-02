from fastapi import FastAPI
from pydantic import BaseModel

from backend.speech.speech_to_text import speech_to_text
from backend.translation.ksl_translator import translate_to_ksl
from backend.avatar.avatar_engine import generate_avatar_sequence


app = FastAPI(
    title="Aperture Signs API",
    description="Real-time sign language accessibility platform",
    version="0.1.0",
)


class PipelineRequest(BaseModel):
    text: str


@app.get("/")
def root():
    return {
        "name": "Aperture Signs",
        "status": "online",
        "version": "0.1.0",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "service": "aperture-signs-backend",
    }


@app.post("/pipeline")
def pipeline(request: PipelineRequest):
    spoken_text = speech_to_text(request.text)

    translation = translate_to_ksl(spoken_text)
    motion_sequences = translation.pop("_motion_sequences")
    avatar = generate_avatar_sequence(translation["sign_sequence"], motion_sequences)

    return {
        "speech": spoken_text,
        "translation": translation,
        "avatar": avatar,
        "signed_output": avatar.get("speech_output", spoken_text),
    }
