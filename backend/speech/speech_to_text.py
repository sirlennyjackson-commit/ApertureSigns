def speech_to_text(audio_text: str) -> str:
    """
    Normalize text already supplied by the caller.

    This function does not record or transcribe audio.
    """
    return " ".join(audio_text.split())
