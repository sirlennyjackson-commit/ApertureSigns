# Aperture Signs

Aperture Signs is an early KSL accessibility prototype. It includes a desktop camera/landmark UI, an experimental sign classifier, and a FastAPI pipeline demo. It is **not a production KSL translator**.

## Requirements

- Windows and Python 3.11 (the project workspace currently selects Python 3.11.9)
- A camera for live tracking
- The model and processed dataset files in this repository for classifier inference and the model smoke test

Create or use the project virtual environment and install the complete runtime dependencies:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Install test dependencies when running the test suite:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
```

`ui_requirements.txt` remains as a compatibility entry point and includes the complete `requirements.txt` manifest.

If upgrading an environment that previously installed multiple OpenCV wheels, remove those overlapping distributions before reinstalling the project manifest:

```powershell
.\.venv\Scripts\python.exe -m pip uninstall -y opencv-python opencv-python-headless opencv-contrib-python
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip check
```

## Run the desktop UI

```powershell
.\.venv\Scripts\python.exe app_ui_v1.py
```

The UI opens without MediaPipe or a classifier model when Qt is installed, and reports the vision/model problem in its status panel rather than failing during module import. Camera access and live tracking still require MediaPipe and a working camera. The separate OpenCV tracker can be started with:

```powershell
.\.venv\Scripts\python.exe backend\vision\hand_tracker.py
```

The Aurora Translate screen connects both interaction directions. Live Capture's stabilized single-word predictions appear in its sign transcript and are spoken through the desktop WebEngine system voice; speech can be muted in Aurora. Typed English and Windows SAPI voice input can look up recorded vocabulary words and play their landmark sequences in the avatar. Open “Browse recorded words” to see and select the supported vocabulary. Voice input uses Windows dictation; typing remains available as a fallback.

Avatar Studio and Aurora Translate include the same full-body, low-poly 2D signing avatar. It follows live body landmarks and recorded hand landmarks during sign playback. The optional Blender bridge remains available for streaming to an external rig.

Speech-to-sign currently performs exact word lookup against the checked-in KSL landmark dataset. It is not English-to-KSL grammar translation, unmapped words are not fabricated, and avatar playback is a landmark visualization rather than a production signing character. Sign recognition remains experimental and should not be relied on for consequential communication.

## Run the API

Start the development server:

```powershell
.\.venv\Scripts\python.exe -m uvicorn backend.main:app --reload
```

Then check `http://127.0.0.1:8000/health` or open `http://127.0.0.1:8000/docs`. The `/pipeline` request accepts JSON such as:

```json
{"text": "hello there"}
```

The response normalizes supplied text, performs exact vocabulary lookup against `data/processed/sign_dataset.npz`, and returns recorded landmark frames for matched words. It does not receive or transcribe audio and does not translate sentence grammar into KSL.

## Smoke checks

Run the model inference smoke check on a saved dataset sequence:

```powershell
.\.venv\Scripts\python.exe -m backend.training.smoke_test
```

Run API, feature, model, and missing-model fallback tests:

```powershell
.\.venv\Scripts\python.exe -m pytest
```

## Model and limitations

The live recognizer uses `models/sign/classifier/aperture_sign_v2.joblib` and `backend/training/features_v2.py`. The checked-in v2 report records **38.6% signer-independent top-1** and **62.9% top-3** accuracy across 60 classes. At confidence ≥ 0.40, reported coverage is **39.9%** and accuracy among accepted predictions is **62.3%**. These figures are experimental and are not evidence of production readiness.

`models/sign/classifier/model_metadata.json` reports results for the older `aperture_sign_classifier.joblib`; do not use those metrics as the v2 evaluation. The dataset is small and imbalanced, and signer-independent accuracy remains low. Collecting more balanced examples across signers and evaluating with held-out signers are necessary next steps.

## Training

Dataset preparation requires the original landmark ZIP archive. Supply its own path rather than using the developer-specific path in the legacy batch launcher:

```powershell
.\.venv\Scripts\python.exe backend\training\prepare_dataset.py --zip "C:\path\to\Pose Data.zip" --min-examples 5
.\.venv\Scripts\python.exe backend\training\train_v2.py
```

Training replaces the v2 model/report outputs. Keep a copy of any model you need before retraining.
