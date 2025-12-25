import json
import pickle

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from diabetes_binary_classifier.features import add_features

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")
DATA_DIR = Path("blobs/raw")
MODEL_DIR = Path("blobs/models/tabular_multi_seed")
MODEL_DIR.mkdir(parents=True, exist_ok=True)


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        bce = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )
        pt = torch.exp(-bce)
        return (self.alpha * (1 - pt) ** self.gamma * bce).mean()


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


def train_fold(X_train, y_train, X_val, y_val, epochs=150, batch_size=1024, lr=1e-3):
    train_ds = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    val_ds = TensorDataset(torch.tensor(X_val), torch.tensor(y_val))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    model = TabularNet(X_train.shape[1]).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    criterion = FocalLoss()

    best_f1 = 0.0
    best_model = None
    patience = 15
    no_improve = 0

    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            criterion(model(xb), yb).backward()
            optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            val_preds = torch.cat(
                [torch.sigmoid(model(xb.to(DEVICE))).cpu() for xb, _ in val_loader]
            ).numpy()

        best_th_f1 = 0.0
        for th in np.arange(0.15, 0.55, 0.005):
            pred_binary = (val_preds > th).astype(int)
            tp = ((pred_binary == 1) & (y_val == 1)).sum()
            fp = ((pred_binary == 1) & (y_val == 0)).sum()
            fn = ((pred_binary == 0) & (y_val == 1)).sum()
            f1 = 2 * tp / (2 * tp + fp + fn + 1e-8)
            if f1 > best_th_f1:
                best_th_f1 = f1

        if best_th_f1 > best_f1:
            best_f1 = best_th_f1
            best_model = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience:
            break

    model.load_state_dict(best_model)
    return model, best_f1


# Load and preprocess
train_df = pd.read_csv(DATA_DIR / "train.csv")
train_df = train_df.drop(columns=["ID"])
train_df = add_features(train_df)

X = train_df.drop(columns=["target"]).values.astype(np.float32)
y = train_df["target"].values.astype(np.float32)

scaler = StandardScaler()
X = scaler.fit_transform(X)

# Save scaler
with open(MODEL_DIR / "scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print(f"Features: {X.shape[1]}")

N_SEEDS = 5
N_FOLDS = 5
SEEDS = [np.random.randint(0, 10000) for _ in range(N_SEEDS)]
print(f"Random seeds: {SEEDS}")
all_oof = []
all_f1_scores = []

for seed in SEEDS:
    print(f"\n{'#' * 60}\nSeed {seed}\n{'#' * 60}")

    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    oof_preds = np.zeros(len(X))
    seed_f1_scores = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"Seed {seed} Fold {fold + 1}/{N_FOLDS}", end=" ")

        model, f1 = train_fold(X[train_idx], y[train_idx], X[val_idx], y[val_idx])
        print(f"F1: {f1:.4f}")
        seed_f1_scores.append(f1)

        model.eval()
        with torch.no_grad():
            oof_preds[val_idx] = (
                torch.sigmoid(model(torch.tensor(X[val_idx]).to(DEVICE))).cpu().numpy()
            )

        torch.save(
            model.state_dict(), MODEL_DIR / f"tabular_net_seed{seed}_fold{fold}.pt"
        )

    all_oof.append(oof_preds)
    all_f1_scores.extend(seed_f1_scores)

final_oof = np.mean(all_oof, axis=0)

best_th, best_f1 = 0.5, 0.0
for th in np.arange(0.15, 0.55, 0.005):
    pred_binary = (final_oof > th).astype(int)
    tp = ((pred_binary == 1) & (y == 1)).sum()
    fp = ((pred_binary == 1) & (y == 0)).sum()
    fn = ((pred_binary == 0) & (y == 1)).sum()
    f1 = 2 * tp / (2 * tp + fp + fn + 1e-8)
    if f1 > best_f1:
        best_th, best_f1 = th, f1

print(f"\n{'=' * 60}")
print(f"Final OOF F1: {best_f1:.4f} (threshold: {best_th:.3f})")

# Save info.json
info = {
    "threshold": float(best_th),
    "n_folds": N_FOLDS,
    "seeds": SEEDS,
    "cv_score": float(np.mean(all_f1_scores)),
    "cv_std": float(np.std(all_f1_scores)),
    "oof_f1": float(best_f1),
    "in_features": int(X.shape[1]),
}
with open(MODEL_DIR / "info.json", "w") as f:
    json.dump(info, f, indent=2)

# Save OOF for ensemble
np.save(MODEL_DIR / "oof_proba.npy", final_oof)
print(f"\nSaved model and info to {MODEL_DIR}")
