# Sign model training

The current live recognizer uses the v2 feature representation and classifier:

- Input sequence: 30 frames × 225 landmarks/features
- Extracted classifier input: 704 features
- Model output: one of 60 sign classes in the checked-in v2 model

Install the project requirements from the repository root, then prepare a dataset from a local landmark ZIP:

```powershell
.\.venv\Scripts\python.exe backend\training\prepare_dataset.py --zip "C:\path\to\Pose Data.zip" --min-examples 5
.\.venv\Scripts\python.exe backend\training\train_v2.py
```

The training script writes `models/sign/classifier/aperture_sign_v2.joblib`, `report_v2.txt`, and `confusion_v2.csv`. It evaluates signer-independent folds; retain the report alongside each model when replacing artifacts.

Verify that the checked-in model loads and predicts on a saved landmark sequence with:

```powershell
.\.venv\Scripts\python.exe -m backend.training.smoke_test
```

The current report records 38.6% signer-independent top-1 accuracy and 62.9% top-3 accuracy across 60 classes. At confidence ≥ 0.40, its reported coverage is 39.9% and accuracy on accepted predictions is 62.3%. These results are experimental and are not production-ready.
