import sys
from pathlib import Path

import numpy as np
import pandas as pd

TABULAR_MODEL_DIR = Path("blobs/models/tabular")
TABULAR_SUBMIT_DIR = Path("blobs/submit/tabular")
LGB_MODEL_DIR = Path("blobs/models/lightgbm")
LGB_SUBMIT_DIR = Path("blobs/submit/lightgbm")
OUTPUT_DIR = Path("blobs/submit/ensemble")
DATA_DIR = Path("blobs/raw")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

required_files = [
    TABULAR_SUBMIT_DIR / "test_proba.npy",
    TABULAR_MODEL_DIR / "oof_proba.npy",
    LGB_SUBMIT_DIR / "test_proba.npy",
    LGB_MODEL_DIR / "oof_proba.npy",
    DATA_DIR / "train.csv",
    DATA_DIR / "test.csv",
]
missing = [path for path in required_files if not path.exists()]
if missing:
    print("Missing required files. Cannot run ensemble:")
    for path in missing:
        print(f"- {path}")
    print("\nRun these first:")
    print("uv run python scripts/train/train_tabular_net.py")
    print("uv run python scripts/predict/predict_tabular_net.py")
    print("uv run python scripts/train/train_lightgbm.py")
    print("uv run python scripts/predict/predict_lightgbm.py")
    sys.exit(1)

# Load predictions
nn_test = np.load(TABULAR_SUBMIT_DIR / "test_proba.npy")
nn_oof = np.load(TABULAR_MODEL_DIR / "oof_proba.npy")
lgb_test = np.load(LGB_SUBMIT_DIR / "test_proba.npy")
lgb_oof = np.load(LGB_MODEL_DIR / "oof_proba.npy")

# Load labels
train_df = pd.read_csv(DATA_DIR / "train.csv")
test_df = pd.read_csv(DATA_DIR / "test.csv")
y = train_df["target"].values
test_ids = test_df["ID"]

# Grid search best weight
best_w, best_th, best_f1 = 0.5, 0.5, 0

for w in np.arange(0.2, 0.8, 0.05):
    blended_oof = w * nn_oof + (1 - w) * lgb_oof

    for th in np.arange(0.15, 0.55, 0.005):
        pred_binary = (blended_oof > th).astype(int)
        tp = ((pred_binary == 1) & (y == 1)).sum()
        fp = ((pred_binary == 1) & (y == 0)).sum()
        fn = ((pred_binary == 0) & (y == 1)).sum()
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)

        if f1 > best_f1:
            best_w, best_th, best_f1 = w, th, f1

print(f"Best weight: NN={best_w:.2f}, LGB={1-best_w:.2f}")
print(f"Best threshold: {best_th:.3f}")
print(f"OOF F1: {best_f1:.4f}")

# Final prediction
final_proba = best_w * nn_test + (1 - best_w) * lgb_test
final_pred = (final_proba > best_th).astype(int)

submission = pd.DataFrame({"ID": test_ids, "TARGET": final_pred})
submission.to_csv(OUTPUT_DIR / "submission_ensemble_tabular_lightgbm.csv", index=False)
print(f"\nSaved to {OUTPUT_DIR / 'submission_ensemble_tabular_lightgbm.csv'}")
