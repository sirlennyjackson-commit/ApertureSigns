# Aperture Signs

Real-time sign language accessibility platform (Kenyan Sign Language and ASL).

```
 webcam / browser
      |  MediaPipe Holistic (hands + pose + face)
      v
 landmarks --> features --> sliding window --> classifier --> confirmed signs
 (21+21+33)    (body-relative,   (45 frames,     (NumPy       (debounced,
               scale-invariant)   every 3rd)      inference)   WebSocket / HUD)

 text / speech --> gloss translation --> avatar timeline      (POST /pipeline)
```

## Layout

| Path | What it is |
|---|---|
| `backend/vision/` | `tracker.py` (headless MediaPipe), `hand_tracker.py` (desktop HUD), `drawing.py` |
| `backend/ml/` | landmarks, features, augmentation, NumPy classifier, real-time recogniser |
| `backend/training/` | `collect.py` (record clips), `train.py` (train + cross-validate + export) |
| `backend/translation/` | text -> sign gloss (`lexicon.json` is a placeholder seed) |
| `backend/avatar/` | gloss -> timed avatar timeline |
| `backend/main.py` | FastAPI: `/pipeline`, `/recognize`, `WS /ws/recognize`, `/collect`, `/stt`, `/signs`, `/health` |
| `frontend/` | Web client (no build step): live recognition, text to signs, in-browser recording |
| `tests/` | synthetic-landmark tests (no camera, no MediaPipe needed) |

## Quick start

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt          # everything (API + camera + training + tests)

# 1. Record data: ~20 clips per sign, plus a resting class. Several signers if you can.
python -m backend.training.collect --label hello  --samples 20 --signer amina
python -m backend.training.collect --label thanks --samples 20 --signer amina
python -m backend.training.collect --label _idle  --samples 20 --signer amina

# 2. Train (prints cross-validated accuracy, saves models/signs.npz)
python -m backend.training.train

# 3. Try it
python -m backend.vision.hand_tracker        # desktop HUD with live transcript
uvicorn backend.main:app --reload            # API on :8000 (docs at /docs), web client at /app/

pytest -q && ruff check .
```
API-only deployments need just `pip install -r requirements.txt`.

## Web client (no desktop camera tools needed)

```bash
APERTURE_ENABLE_COLLECT=1 uvicorn backend.main:app --reload     # or: make run
# open http://localhost:8000/app/
```
1. **Record data** tab: tick the consent box, record ~20 clips per sign (+ `_idle`).
2. `python -m backend.training.train`, then press **Reload model** in the same tab.
3. **Signs to text** tab: start the camera and sign. **Text to signs** tab: type or dictate.

The browser runs MediaPipe itself and sends only landmark numbers, never video. The
browser has no face landmarker, so face metrics are absent there. Train on clips recorded
with the **same client you deploy** (browser clips for the browser, desktop clips for the
desktop HUD) so training and live data match.

Docker: `docker compose up --build`. HTTPS and the production checklist: `docs/deploy.md`.
Optional offline dictation (English + Swahili): `pip install -r requirements-stt.txt`.

## Configuration (env vars)

`APERTURE_MODEL_PATH`, `APERTURE_WINDOW_FRAMES` (45), `APERTURE_STRIDE` (3),
`APERTURE_MIN_CONFIDENCE` (0.75), `APERTURE_MIN_VOTES` (2),
`APERTURE_MIN_HAND_ACTIVITY` (0.3), `APERTURE_CORS_ORIGINS`.

Keep `collect --seconds` x your camera fps close to `APERTURE_WINDOW_FRAMES`
(1.5 s x 30 fps = 45), so live windows look like the clips the model trained on.

## What works today, and what does not

- Works: isolated-sign recognition from your own recorded vocabulary, live debounced
  recognition over WebSocket, offline training with signer-aware validation.
- Does **not** yet: continuous signing / sentences (needs a sequence model with
  segmentation, e.g. CTC), and the avatar has no animation clips.
- Translation is a **lexicon lookup, not real KSL grammar**. Have Deaf KSL signers
  build and review the lexicon, and the training vocabulary.
- Accuracy is only meaningful on signers the model has not seen. `train.py` reports
  that when you record 2+ signers; single-signer numbers are optimistic.

## Roadmap

1. Record a first real vocabulary (30-50 signs) with 3+ signers.
2. Neural sequence model (GRU/Transformer) once there are 50+ clips per sign.
3. Avatar clips / rig for the timeline; real KSL grammar with Deaf linguists.
