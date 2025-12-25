import lightgbm as lgb
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from pathlib import Path

DATA_DIR = Path("blobs/raw")
OUTPUT_DIR = Path("blobs/submit/lightgbm")
MODEL_DIR = Path("blobs/models/lightgbm")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def add_features(df):
    df = df.copy()
    df['metabolic_risk'] = df['HighBP'] + df['HighChol'] + (df['BMI'] > 30).astype(int)
    df['cardio_risk'] = df['Stroke'] + df['HeartDiseaseorAttack'] + df['HighBP'] + df['HighChol']
    df['lifestyle'] = df['PhysActivity'] + df['Fruits'] + df['Veggies'] - df['Smoker'] - df['HvyAlcoholConsump']
    df['health_gap'] = df['GenHlth'] - (df['PhysHlth'] + df['MentHlth']) / 30
    df['bmi_cat'] = pd.cut(
        df['BMI'],
        bins=[0, 18.5, 25, 30, 35, 100],
        labels=[0, 1, 2, 3, 4]
    ).astype(int)
    df['age_bmi'] = df['Age'] * df['BMI']
    df['age_health'] = df['Age'] * df['GenHlth']
    df['age_cardio'] = df['Age'] * df['cardio_risk']
    return df


# Load data
train_df = pd.read_csv(DATA_DIR / "train.csv")
test_df = pd.read_csv(DATA_DIR / "test.csv")

test_ids = test_df["ID"].copy()
train_df = train_df.drop(columns=["ID"])
test_df = test_df.drop(columns=["ID"])

train_df = add_features(train_df)
test_df = add_features(test_df)

X = train_df.drop(columns=["target"])
y = train_df["target"].values
X_test = test_df

feature_names = X.columns.tolist()
print(f"Features: {len(feature_names)}")

# LightGBM params (optimized for F1 on imbalanced data)
params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'boosting_type': 'gbdt',
    'n_estimators': 2000,
    'learning_rate': 0.02,
    'max_depth': 8,
    'num_leaves': 64,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'min_child_samples': 50,
    'scale_pos_weight': (y == 0).sum() / (y == 1).sum(),
    'random_state': 42,
    'verbose': -1,
    'n_jobs': -1,
}

# K-Fold
N_FOLDS = 5
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

oof_preds = np.zeros(len(X))
test_preds = np.zeros(len(X_test))
f1_scores = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    print(f"\n{'='*50}\nFold {fold+1}/{N_FOLDS}\n{'='*50}")

    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    model = lgb.LGBMClassifier(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[
            lgb.early_stopping(100, verbose=False),
            lgb.log_evaluation(200)
        ]
    )

    # Save model
    model.booster_.save_model(str(MODEL_DIR / f"lightgbm_fold{fold}.txt"))

    # Predict probabilities
    val_proba = model.predict_proba(X_val)[:, 1]
    oof_preds[val_idx] = val_proba
    test_preds += model.predict_proba(X_test)[:, 1] / N_FOLDS

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
importance = pd.DataFrame({
    'feature': feature_names,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)
print(f"\nTop 10 features:\n{importance.head(10)}")

# Save submission
test_binary = (test_preds > best_th).astype(int)
submission = pd.DataFrame({"ID": test_ids, "TARGET": test_binary})
submission.to_csv(OUTPUT_DIR / "submission_lightgbm.csv", index=False)
print(f"\nSaved to {OUTPUT_DIR / 'submission_lightgbm.csv'}")

# Save probabilities for ensemble
np.save(OUTPUT_DIR / "lightgbm_test_proba.npy", test_preds)
np.save(OUTPUT_DIR / "lightgbm_oof_proba.npy", oof_preds)
print("Saved probabilities for ensemble")
