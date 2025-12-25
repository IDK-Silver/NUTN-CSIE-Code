import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression

DATA_DIR = Path("blobs/raw")
OPTUNA_DIR = Path("blobs/submit/tabular_optuna")
OUTPUT_DIR = Path("blobs/submit/tabular_optuna_post")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

N_FOLDS = 5
SEED = 42
THRESHOLDS = np.arange(0.15, 0.55, 0.005)

required_files = [
    OPTUNA_DIR / "tabular_net_optuna_oof_proba.npy",
    OPTUNA_DIR / "tabular_net_optuna_test_proba.npy",
    DATA_DIR / "train.csv",
    DATA_DIR / "test.csv",
]
missing = [path for path in required_files if not path.exists()]
if missing:
    print("Missing required files. Cannot run postprocess:")
    for path in missing:
        print(f"- {path}")
    raise SystemExit(1)

train_df = pd.read_csv(DATA_DIR / "train.csv")
test_df = pd.read_csv(DATA_DIR / "test.csv")
y = train_df["target"].values.astype(int)
test_ids = test_df["ID"].values

oof_preds = np.load(OPTUNA_DIR / "tabular_net_optuna_oof_proba.npy")
test_preds = np.load(OPTUNA_DIR / "tabular_net_optuna_test_proba.npy")


def best_threshold(preds, targets, thresholds):
    preds = preds.reshape(-1, 1)
    thresholds = thresholds.reshape(1, -1)
    pred_binary = preds > thresholds
    targets = targets.reshape(-1, 1)
    tp = np.sum(pred_binary & (targets == 1), axis=0)
    fp = np.sum(pred_binary & (targets == 0), axis=0)
    fn = np.sum(~pred_binary & (targets == 1), axis=0)
    f1 = 2 * tp / (2 * tp + fp + fn + 1e-8)
    best_idx = np.argmax(f1)
    return thresholds[0, best_idx], f1[best_idx]


def f1_at_threshold(preds, targets, threshold):
    pred_binary = (preds > threshold).astype(int)
    tp = np.sum((pred_binary == 1) & (targets == 1))
    fp = np.sum((pred_binary == 1) & (targets == 0))
    fn = np.sum((pred_binary == 0) & (targets == 1))
    return 2 * tp / (2 * tp + fp + fn + 1e-8)


# Nested threshold (fold-wise, then average)
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
fold_thresholds = []

for _, val_idx in skf.split(np.zeros(len(y)), y):
    fold_preds = oof_preds[val_idx]
    fold_targets = y[val_idx]
    th, _ = best_threshold(fold_preds, fold_targets, THRESHOLDS)
    fold_thresholds.append(th)

nested_th = float(np.mean(fold_thresholds))
nested_oof_f1 = f1_at_threshold(oof_preds, y, nested_th)

nested_pred = (test_preds > nested_th).astype(int)
submission = pd.DataFrame({"ID": test_ids, "TARGET": nested_pred})
submission.to_csv(OUTPUT_DIR / "submission_tabular_net_optuna_nested.csv", index=False)

print(f"Nested threshold: {nested_th:.3f}")
print(f"Nested OOF F1: {nested_oof_f1:.4f}")
print(f"Saved to {OUTPUT_DIR / 'submission_tabular_net_optuna_nested.csv'}")

# Calibration (Platt scaling) + global threshold
calibrator = LogisticRegression(solver="lbfgs", max_iter=1000)
calibrator.fit(oof_preds.reshape(-1, 1), y)
calib_oof = calibrator.predict_proba(oof_preds.reshape(-1, 1))[:, 1]
calib_test = calibrator.predict_proba(test_preds.reshape(-1, 1))[:, 1]

calib_th, calib_oof_f1 = best_threshold(calib_oof, y, THRESHOLDS)

calib_pred = (calib_test > calib_th).astype(int)
submission = pd.DataFrame({"ID": test_ids, "TARGET": calib_pred})
submission.to_csv(OUTPUT_DIR / "submission_tabular_net_optuna_calibrated.csv", index=False)

print(f"Calibrated threshold: {calib_th:.3f}")
print(f"Calibrated OOF F1: {calib_oof_f1:.4f}")
print(f"Saved to {OUTPUT_DIR / 'submission_tabular_net_optuna_calibrated.csv'}")
