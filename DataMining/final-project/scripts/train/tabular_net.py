import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path

# Config
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATA_DIR = Path("blobs/raw")
OUTPUT_DIR = Path("blobs/submit/tabular")
MODEL_DIR = Path("blobs/models/tabular")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def add_features(df):
    df = df.copy()
    # Metabolic risk
    df['metabolic_risk'] = df['HighBP'] + df['HighChol'] + (df['BMI'] > 30).astype(int)
    # Cardiovascular risk
    df['cardio_risk'] = df['Stroke'] + df['HeartDiseaseorAttack'] + df['HighBP'] + df['HighChol']
    # Lifestyle score
    df['lifestyle'] = df['PhysActivity'] + df['Fruits'] + df['Veggies'] - df['Smoker'] - df['HvyAlcoholConsump']
    # Health gap
    df['health_gap'] = df['GenHlth'] - (df['PhysHlth'] + df['MentHlth']) / 30
    # BMI category
    df['bmi_cat'] = pd.cut(
        df['BMI'],
        bins=[0, 18.5, 25, 30, 35, 100],
        labels=[0, 1, 2, 3, 4]
    ).astype(int)
    # Interactions
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

# Feature engineering
train_df = add_features(train_df)
test_df = add_features(test_df)

X = train_df.drop(columns=["target"]).values.astype(np.float32)
y = train_df["target"].values.astype(np.float32)
X_test = test_df.values.astype(np.float32)

# Normalize
scaler = StandardScaler()
X = scaler.fit_transform(X)
X_test = scaler.transform(X_test)

print(f"Features: {X.shape[1]}")


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        bce = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, reduction='none'
        )
        pt = torch.exp(-bce)
        focal = self.alpha * (1 - pt) ** self.gamma * bce
        return focal.mean()


class TabularNet(nn.Module):
    def __init__(self, in_features, hidden_dims=[512, 256, 128, 64]):
        super().__init__()
        layers = []
        prev_dim = in_features
        for h in hidden_dims:
            layers += [
                nn.Linear(prev_dim, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(0.2),
            ]
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_fold(X_train, y_train, X_val, y_val, fold_idx, epochs=150, batch_size=1024, lr=1e-3):
    train_ds = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    val_ds = TensorDataset(torch.tensor(X_val), torch.tensor(y_val))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    model = TabularNet(X_train.shape[1]).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    criterion = FocalLoss(alpha=0.25, gamma=2.0)

    best_f1 = 0
    best_model = None
    best_threshold = 0.5
    patience = 15
    no_improve = 0

    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
        scheduler.step()

        # Validation
        model.eval()
        val_preds = []
        val_targets = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(DEVICE)
                val_preds.append(torch.sigmoid(model(xb)).cpu())
                val_targets.append(yb)

        val_preds = torch.cat(val_preds).numpy()
        val_targets = torch.cat(val_targets).numpy()

        # Fine-grained threshold search
        best_th, best_th_f1 = 0.5, 0
        for th in np.arange(0.15, 0.55, 0.005):
            pred_binary = (val_preds > th).astype(int)
            tp = ((pred_binary == 1) & (val_targets == 1)).sum()
            fp = ((pred_binary == 1) & (val_targets == 0)).sum()
            fn = ((pred_binary == 0) & (val_targets == 1)).sum()
            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)
            f1 = 2 * precision * recall / (precision + recall + 1e-8)
            if f1 > best_th_f1:
                best_th, best_th_f1 = th, f1

        if best_th_f1 > best_f1:
            best_f1 = best_th_f1
            best_threshold = best_th
            best_model = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if epoch % 10 == 0:
            print(
                f"Epoch {epoch}: val_f1={best_th_f1:.4f} (th={best_th:.3f}), best={best_f1:.4f}"
            )

        if no_improve >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

    model.load_state_dict(best_model)
    torch.save(best_model, MODEL_DIR / f"tabular_net_fold{fold_idx}.pt")
    return model, best_threshold, best_f1


# K-Fold training
N_FOLDS = 5
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

oof_preds = np.zeros(len(X))
test_preds = np.zeros(len(X_test))
thresholds = []
f1_scores = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    print(f"\n{'='*50}\nFold {fold+1}/{N_FOLDS}\n{'='*50}")

    model, threshold, f1 = train_fold(
        X[train_idx], y[train_idx],
        X[val_idx], y[val_idx],
        fold_idx=fold,
        epochs=150,
        batch_size=1024,
        lr=1e-3
    )
    thresholds.append(threshold)
    f1_scores.append(f1)

    model.eval()
    with torch.no_grad():
        oof_preds[val_idx] = torch.sigmoid(
            model(torch.tensor(X[val_idx]).to(DEVICE))
        ).cpu().numpy()

        test_preds += torch.sigmoid(
            model(torch.tensor(X_test).to(DEVICE))
        ).cpu().numpy() / N_FOLDS

print(f"\n{'='*50}")
print(f"Fold F1 scores: {[f'{f:.4f}' for f in f1_scores]}")
print(f"Mean F1: {np.mean(f1_scores):.4f} (+/- {np.std(f1_scores):.4f})")

# Final threshold from OOF
best_final_th, best_final_f1 = 0.5, 0
for th in np.arange(0.15, 0.55, 0.005):
    pred_binary = (oof_preds > th).astype(int)
    tp = ((pred_binary == 1) & (y == 1)).sum()
    fp = ((pred_binary == 1) & (y == 0)).sum()
    fn = ((pred_binary == 0) & (y == 1)).sum()
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    if f1 > best_final_f1:
        best_final_th, best_final_f1 = th, f1

print(f"\nOptimal threshold (OOF): {best_final_th:.3f}")
print(f"OOF F1: {best_final_f1:.4f}")

# Save submission
test_binary = (test_preds > best_final_th).astype(int)
submission = pd.DataFrame({"ID": test_ids, "TARGET": test_binary})
submission.to_csv(OUTPUT_DIR / "submission_tabular_net.csv", index=False)
print(f"\nSaved to {OUTPUT_DIR / 'submission_tabular_net.csv'}")

# Save probabilities for ensemble
np.save(OUTPUT_DIR / "tabular_net_test_proba.npy", test_preds)
np.save(OUTPUT_DIR / "tabular_net_oof_proba.npy", oof_preds)
print("Saved probabilities for ensemble")
