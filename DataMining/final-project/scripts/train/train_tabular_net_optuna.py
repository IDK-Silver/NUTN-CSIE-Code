import json
import os
import pickle

import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from diabetes_binary_classifier.features import add_features

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = Path("blobs/raw")
MODEL_DIR = Path("blobs/models/tabular_optuna")
MODEL_DIR.mkdir(parents=True, exist_ok=True)
PIN_MEMORY = DEVICE.type == "cuda"
NON_BLOCKING = DEVICE.type == "cuda"
NUM_WORKERS = min(8, os.cpu_count() or 4)
DATA_LOADER_KWARGS = {}
if NUM_WORKERS > 0:
    DATA_LOADER_KWARGS = {
        "num_workers": NUM_WORKERS,
        "persistent_workers": True,
        "prefetch_factor": 2,
    }

if DEVICE.type == "cuda":
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")


# Load data
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

X_tensor_cpu = torch.from_numpy(X)
y_tensor_cpu = torch.from_numpy(y)
y_tensor = y_tensor_cpu.to(DEVICE)

torch.cuda.empty_cache() if DEVICE.type == "cuda" else None

print(f"Features: {X.shape[1]}")


def max_f1_for_thresholds(preds, targets, thresholds):
    preds = preds.unsqueeze(1)
    thresholds = thresholds.unsqueeze(0)
    pred_binary = (preds > thresholds).float()
    targets = targets.unsqueeze(1)
    tp = (pred_binary * targets).sum(dim=0)
    fp = (pred_binary * (1 - targets)).sum(dim=0)
    fn = ((1 - pred_binary) * targets).sum(dim=0)
    f1 = 2 * tp / (2 * tp + fp + fn + 1e-8)
    return f1.max()


def best_threshold(preds, targets, thresholds):
    preds = preds.unsqueeze(1)
    thresholds = thresholds.unsqueeze(0)
    pred_binary = (preds > thresholds).float()
    targets = targets.unsqueeze(1)
    tp = (pred_binary * targets).sum(dim=0)
    fp = (pred_binary * (1 - targets)).sum(dim=0)
    fn = ((1 - pred_binary) * targets).sum(dim=0)
    f1 = 2 * tp / (2 * tp + fp + fn + 1e-8)
    best_idx = torch.argmax(f1)
    return thresholds[0, best_idx].item(), f1[best_idx].item()


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
    def __init__(self, in_features, hidden_dims, dropout):
        super().__init__()
        layers = []
        prev_dim = in_features
        for h in hidden_dims:
            layers += [
                nn.Linear(prev_dim, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_and_evaluate(params, n_folds=5):
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True)
    oof_preds = torch.zeros(X_tensor_cpu.size(0), device=DEVICE, dtype=torch.float32)

    use_amp = DEVICE.type == "cuda"
    thresholds_epoch = torch.arange(0.2, 0.5, 0.01, device=DEVICE)

    for train_idx, val_idx in skf.split(X, y):
        val_idx_t = torch.tensor(val_idx, device=DEVICE)

        X_train = X_tensor_cpu[train_idx]
        y_train = y_tensor_cpu[train_idx]
        X_val = X_tensor_cpu[val_idx]
        y_val = y_tensor[val_idx_t]

        model = TabularNet(
            X_train.shape[1], params["hidden_dims"], params["dropout"]
        ).to(DEVICE)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"]
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, params["epochs"]
        )
        criterion = FocalLoss(alpha=params["focal_alpha"], gamma=params["focal_gamma"])
        grad_scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

        train_ds = TensorDataset(X_train, y_train)
        train_loader = DataLoader(
            train_ds,
            batch_size=params["batch_size"],
            shuffle=True,
            pin_memory=PIN_MEMORY,
            **DATA_LOADER_KWARGS,
        )
        X_val_tensor = X_val.to(DEVICE, non_blocking=NON_BLOCKING)

        best_f1 = 0.0
        best_model = None
        patience = 15
        no_improve = 0

        for _ in range(params["epochs"]):
            model.train()
            for xb, yb in train_loader:
                xb = xb.to(DEVICE, non_blocking=NON_BLOCKING)
                yb = yb.to(DEVICE, non_blocking=NON_BLOCKING)
                optimizer.zero_grad()
                with torch.amp.autocast("cuda", enabled=use_amp):
                    logits = model(xb)
                    loss = criterion(logits, yb)
                grad_scaler.scale(loss).backward()
                grad_scaler.step(optimizer)
                grad_scaler.update()
            scheduler.step()

            model.eval()
            with torch.inference_mode():
                with torch.amp.autocast("cuda", enabled=use_amp):
                    val_preds = torch.sigmoid(model(X_val_tensor)).float()

            best_th_f1 = max_f1_for_thresholds(val_preds, y_val, thresholds_epoch).item()

            if best_th_f1 > best_f1:
                best_f1 = best_th_f1
                best_model = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1

            if no_improve >= patience:
                break

        model.load_state_dict(best_model)
        model.eval()
        with torch.inference_mode():
            with torch.amp.autocast("cuda", enabled=use_amp):
                oof_preds[val_idx_t] = torch.sigmoid(model(X_val_tensor)).float()

    thresholds_oof = torch.arange(0.2, 0.5, 0.005, device=DEVICE)
    best_f1 = max_f1_for_thresholds(oof_preds, y_tensor, thresholds_oof)
    return best_f1.item()


def objective(trial):
    n_layers = trial.suggest_int("n_layers", 3, 5)
    first_dim = trial.suggest_categorical("first_dim", [256, 384, 512, 768])
    hidden_dims = [first_dim // (2**i) for i in range(n_layers)]
    focal_alpha = trial.suggest_float("focal_alpha", 0.1, 0.5)
    focal_gamma = trial.suggest_float("focal_gamma", 1.0, 3.0)

    params = {
        "hidden_dims": hidden_dims,
        "dropout": trial.suggest_float("dropout", 0.1, 0.4),
        "lr": trial.suggest_float("lr", 5e-4, 5e-3, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [512, 1024, 2048, 4096]),
        "focal_alpha": focal_alpha,
        "focal_gamma": focal_gamma,
        "epochs": 150,
    }

    return train_and_evaluate(params, n_folds=5)


study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50, show_progress_bar=True)

print(f"\nBest OOF F1: {study.best_value:.4f}")
print(f"Best params: {study.best_params}")

print("\n" + "=" * 60)
print("Training final model with best params...")
print("=" * 60)

best = study.best_params
n_layers = best["n_layers"]
first_dim = best["first_dim"]
hidden_dims = [first_dim // (2**i) for i in range(n_layers)]

final_params = {
    "hidden_dims": hidden_dims,
    "dropout": best["dropout"],
    "lr": best["lr"],
    "weight_decay": best["weight_decay"],
    "batch_size": best["batch_size"],
    "focal_alpha": best.get("focal_alpha", 0.25),
    "focal_gamma": best.get("focal_gamma", 2.0),
    "epochs": 150,
}

skf = StratifiedKFold(n_splits=5, shuffle=True)
oof_preds = torch.zeros(X_tensor_cpu.size(0), device=DEVICE, dtype=torch.float32)

use_amp = DEVICE.type == "cuda"
thresholds_epoch = torch.arange(0.2, 0.5, 0.01, device=DEVICE)
f1_scores = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    print(f"Fold {fold + 1}/5")

    val_idx_t = torch.tensor(val_idx, device=DEVICE)

    X_train = X_tensor_cpu[train_idx]
    y_train = y_tensor_cpu[train_idx]
    X_val = X_tensor_cpu[val_idx]
    y_val = y_tensor[val_idx_t]

    model = TabularNet(
        X_train.shape[1], final_params["hidden_dims"], final_params["dropout"]
    ).to(DEVICE)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=final_params["lr"],
        weight_decay=final_params["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, final_params["epochs"]
    )
    criterion = FocalLoss(
        alpha=final_params["focal_alpha"], gamma=final_params["focal_gamma"]
    )
    grad_scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    train_ds = TensorDataset(X_train, y_train)
    train_loader = DataLoader(
        train_ds,
        batch_size=final_params["batch_size"],
        shuffle=True,
        pin_memory=PIN_MEMORY,
        **DATA_LOADER_KWARGS,
    )
    X_val_tensor = X_val.to(DEVICE, non_blocking=NON_BLOCKING)

    best_f1 = 0.0
    best_model = None
    patience = 15
    no_improve = 0

    for _ in range(final_params["epochs"]):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(DEVICE, non_blocking=NON_BLOCKING)
            yb = yb.to(DEVICE, non_blocking=NON_BLOCKING)
            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(xb)
                loss = criterion(logits, yb)
            grad_scaler.scale(loss).backward()
            grad_scaler.step(optimizer)
            grad_scaler.update()
        scheduler.step()

        model.eval()
        with torch.inference_mode():
            with torch.amp.autocast("cuda", enabled=use_amp):
                val_preds = torch.sigmoid(model(X_val_tensor)).float()

        best_th_f1 = max_f1_for_thresholds(val_preds, y_val, thresholds_epoch).item()

        if best_th_f1 > best_f1:
            best_f1 = best_th_f1
            best_model = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience:
            break

    model.load_state_dict(best_model)
    model.eval()
    with torch.inference_mode():
        with torch.amp.autocast("cuda", enabled=use_amp):
            oof_preds[val_idx_t] = torch.sigmoid(model(X_val_tensor)).float()

    torch.save(model.state_dict(), MODEL_DIR / f"tabular_net_optuna_fold{fold}.pt")
    f1_scores.append(best_f1)
    print(f"  Best F1: {best_f1:.4f}")

thresholds_final = torch.arange(0.15, 0.55, 0.005, device=DEVICE)
best_th, best_oof_f1 = best_threshold(oof_preds, y_tensor, thresholds_final)

print(f"\nFinal OOF F1: {best_oof_f1:.4f} (th={best_th:.3f})")

# Save info.json
info = {
    "threshold": float(best_th),
    "n_folds": 5,
    "cv_score": float(np.mean(f1_scores)),
    "cv_std": float(np.std(f1_scores)),
    "oof_f1": float(best_oof_f1),
    "in_features": int(X.shape[1]),
    "hidden_dims": hidden_dims,
    "dropout": float(final_params["dropout"]),
    "best_params": best,
}
with open(MODEL_DIR / "info.json", "w") as f:
    json.dump(info, f, indent=2)

# Save OOF for ensemble
np.save(MODEL_DIR / "oof_proba.npy", oof_preds.cpu().numpy())
print(f"\nSaved model and info to {MODEL_DIR}")
