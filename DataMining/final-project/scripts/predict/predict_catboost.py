import json

import catboost as cb
import numpy as np
import pandas as pd
from pathlib import Path

from diabetes_binary_classifier.features import add_features

DATA_DIR = Path("blobs/raw")
MODEL_DIR = Path("blobs/models/catboost")
OUTPUT_DIR = Path("blobs/submit/catboost")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Check required files
if not MODEL_DIR.exists():
    raise FileNotFoundError(f"Model directory not found: {MODEL_DIR}")
if not (MODEL_DIR / "info.json").exists():
    raise FileNotFoundError(f"info.json not found in {MODEL_DIR}")

# Load info
with open(MODEL_DIR / "info.json") as f:
    info = json.load(f)

n_folds = info["n_folds"]
threshold = info["threshold"]
print(f"Loaded model info: n_folds={n_folds}, threshold={threshold:.3f}")

# Load test data
test_df = pd.read_csv(DATA_DIR / "test.csv")
test_ids = test_df["ID"].copy()
test_df = test_df.drop(columns=["ID"])
test_df = add_features(test_df)

# Load models and predict
test_preds = np.zeros(len(test_df))

for fold in range(n_folds):
    model_path = MODEL_DIR / f"catboost_fold{fold}.cbm"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    model = cb.CatBoostClassifier()
    model.load_model(str(model_path))
    test_preds += model.predict_proba(test_df)[:, 1] / n_folds
    print(f"Loaded fold {fold}")

# Apply threshold and save
test_binary = (test_preds > threshold).astype(int)
submission = pd.DataFrame({"ID": test_ids, "TARGET": test_binary})
submission.to_csv(OUTPUT_DIR / "submission_catboost.csv", index=False)
print(f"Saved to {OUTPUT_DIR / 'submission_catboost.csv'}")

# Save probabilities for ensemble
np.save(OUTPUT_DIR / "test_proba.npy", test_preds)
print("Saved test probabilities for ensemble")
