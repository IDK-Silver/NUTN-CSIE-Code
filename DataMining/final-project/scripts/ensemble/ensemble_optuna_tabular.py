import numpy as np
import pandas as pd
from pathlib import Path

TABULAR_MODEL_DIR = Path("blobs/models/tabular")
TABULAR_SUBMIT_DIR = Path("blobs/submit/tabular")
OPTUNA_MODEL_DIR = Path("blobs/models/tabular_optuna")
OPTUNA_SUBMIT_DIR = Path("blobs/submit/tabular_optuna")
OUTPUT_DIR = Path("blobs/submit/ensemble_optuna_tabular")
DATA_DIR = Path("blobs/raw")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

required_files = [
    TABULAR_SUBMIT_DIR / "test_proba.npy",
    TABULAR_MODEL_DIR / "oof_proba.npy",
    OPTUNA_SUBMIT_DIR / "test_proba.npy",
    OPTUNA_MODEL_DIR / "oof_proba.npy",
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
    print("uv run python scripts/train/train_tabular_net_optuna.py")
    print("uv run python scripts/predict/predict_tabular_net_optuna.py")
    raise SystemExit(1)

v2_test = np.load(TABULAR_SUBMIT_DIR / "test_proba.npy")
v2_oof = np.load(TABULAR_MODEL_DIR / "oof_proba.npy")
optuna_test = np.load(OPTUNA_SUBMIT_DIR / "test_proba.npy")
optuna_oof = np.load(OPTUNA_MODEL_DIR / "oof_proba.npy")

train_df = pd.read_csv(DATA_DIR / "train.csv")
test_df = pd.read_csv(DATA_DIR / "test.csv")
y = train_df["target"].values
test_ids = test_df["ID"]

best_w, best_th, best_f1 = 0.5, 0.5, 0

for w in np.arange(0.0, 1.05, 0.05):
    blended_oof = w * optuna_oof + (1 - w) * v2_oof

    for th in np.arange(0.25, 0.50, 0.005):
        pred_binary = (blended_oof > th).astype(int)
        tp = ((pred_binary == 1) & (y == 1)).sum()
        fp = ((pred_binary == 1) & (y == 0)).sum()
        fn = ((pred_binary == 0) & (y == 1)).sum()
        f1 = 2 * tp / (2 * tp + fp + fn + 1e-8)

        if f1 > best_f1:
            best_w, best_th, best_f1 = w, th, f1

print(f"Best weight: Optuna={best_w:.2f}, TabularNet={1 - best_w:.2f}")
print(f"Best threshold: {best_th:.3f}")
print(f"OOF F1: {best_f1:.4f}")

final_proba = best_w * optuna_test + (1 - best_w) * v2_test
final_pred = (final_proba > best_th).astype(int)

submission = pd.DataFrame({"ID": test_ids, "TARGET": final_pred})
submission.to_csv(OUTPUT_DIR / "submission_ensemble_optuna_tabular.csv", index=False)
print(f"Saved to {OUTPUT_DIR / 'submission_ensemble_optuna_tabular.csv'}")
