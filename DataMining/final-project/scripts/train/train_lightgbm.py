import json

import lightgbm as lgb
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold

from diabetes_binary_classifier.features import add_features

DATA_DIR = Path("blobs/raw")
MODEL_DIR = Path("blobs/models/lightgbm")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Load data
train_df = pd.read_csv(DATA_DIR / "train.csv")
train_df = train_df.drop(columns=["ID"])
train_df = add_features(train_df)

X = train_df.drop(columns=["target"])
y = train_df["target"].values

feature_names = X.columns.tolist()
print(f"Features: {len(feature_names)}")

# LightGBM params
params = {
    "objective": "binary",
    "metric": "binary_logloss",
    "boosting_type": "gbdt",
    "n_estimators": 2000,
    "learning_rate": 0.02,
    "max_depth": 8,
    "num_leaves": 64,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "min_child_samples": 50,
    "scale_pos_weight": (y == 0).sum() / (y == 1).sum(),
    "random_state": 42,
    "verbose": -1,
    "n_jobs": -1,
}

# K-Fold
N_FOLDS = 5
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

oof_preds = np.zeros(len(X))
f1_scores = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    print(f"\n{'='*50}\nFold {fold+1}/{N_FOLDS}\n{'='*50}")

    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    model = lgb.LGBMClassifier(**params)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(200)],
    )

    # Save model
    model.booster_.save_model(str(MODEL_DIR / f"lightgbm_fold{fold}.txt"))

    # OOF predictions
    val_proba = model.predict_proba(X_val)[:, 1]
    oof_preds[val_idx] = val_proba

    # Find best threshold for this fold
    best_th, best_f1 = 0.5, 0
    for th in np.arange(0.15, 0.55, 0.005):
        pred_binary = (val_proba > th).astype(int)
        tp = ((pred_binary == 1) & (y_val == 1)).sum()
        fp = ((pred_binary == 1) & (y_val == 0)).sum()
        fn = ((pred_binary == 0) & (y_val == 1)).sum()
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        if f1 > best_f1:
            best_th, best_f1 = th, f1

    f1_scores.append(best_f1)
    print(f"Fold {fold+1} F1: {best_f1:.4f} (th={best_th:.3f})")

print(f"\n{'='*50}")
print(f"Fold F1 scores: {[f'{f:.4f}' for f in f1_scores]}")
print(f"Mean F1: {np.mean(f1_scores):.4f} (+/- {np.std(f1_scores):.4f})")

# Find optimal threshold on OOF
best_th, best_f1 = 0.5, 0
for th in np.arange(0.15, 0.55, 0.005):
    pred_binary = (oof_preds > th).astype(int)
    tp = ((pred_binary == 1) & (y == 1)).sum()
    fp = ((pred_binary == 1) & (y == 0)).sum()
    fn = ((pred_binary == 0) & (y == 1)).sum()
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    if f1 > best_f1:
        best_th, best_f1 = th, f1

print(f"\nOptimal threshold (OOF): {best_th:.3f}")
print(f"OOF F1: {best_f1:.4f}")

# Feature importance
importance = pd.DataFrame(
    {"feature": feature_names, "importance": model.feature_importances_}
).sort_values("importance", ascending=False)
print(f"\nTop 10 features:\n{importance.head(10)}")

# Save info.json
info = {
    "threshold": float(best_th),
    "n_folds": N_FOLDS,
    "cv_score": float(np.mean(f1_scores)),
    "cv_std": float(np.std(f1_scores)),
    "oof_f1": float(best_f1),
    "feature_names": feature_names,
}
with open(MODEL_DIR / "info.json", "w") as f:
    json.dump(info, f, indent=2)

# Save OOF for ensemble
np.save(MODEL_DIR / "oof_proba.npy", oof_preds)
print(f"\nSaved model and info to {MODEL_DIR}")
