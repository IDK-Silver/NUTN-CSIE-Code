import json
import pickle

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path

from diabetes_binary_classifier.features import add_features

# Config
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")
DATA_DIR = Path("blobs/raw")
MODEL_DIR = Path("blobs/models/tabular_optuna")
OUTPUT_DIR = Path("blobs/submit/tabular_optuna")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


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


# Check required files
if not MODEL_DIR.exists():
    raise FileNotFoundError(f"Model directory not found: {MODEL_DIR}")
if not (MODEL_DIR / "info.json").exists():
    raise FileNotFoundError(f"info.json not found in {MODEL_DIR}")
if not (MODEL_DIR / "scaler.pkl").exists():
    raise FileNotFoundError(f"scaler.pkl not found in {MODEL_DIR}")

# Load info
with open(MODEL_DIR / "info.json") as f:
    info = json.load(f)

n_folds = info["n_folds"]
threshold = info["threshold"]
in_features = info["in_features"]
hidden_dims = info["hidden_dims"]
dropout = info["dropout"]
print(f"Loaded model info: n_folds={n_folds}, threshold={threshold:.3f}")
print(f"Architecture: hidden_dims={hidden_dims}, dropout={dropout}")

# Load scaler
with open(MODEL_DIR / "scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Load test data
test_df = pd.read_csv(DATA_DIR / "test.csv")
test_ids = test_df["ID"].copy()
test_df = test_df.drop(columns=["ID"])
test_df = add_features(test_df)

X_test = test_df.values.astype(np.float32)
X_test = scaler.transform(X_test)
X_test_tensor = torch.from_numpy(X_test).to(DEVICE)

# Load models and predict
test_preds = torch.zeros(len(X_test), device=DEVICE)

for fold in range(n_folds):
    model_path = MODEL_DIR / f"tabular_net_optuna_fold{fold}.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    model = TabularNet(in_features, hidden_dims, dropout).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    with torch.inference_mode():
        preds = torch.sigmoid(model(X_test_tensor))
        test_preds += preds / n_folds

    print(f"Loaded fold {fold}")

test_preds_np = test_preds.cpu().numpy()

# Apply threshold and save
test_binary = (test_preds_np > threshold).astype(int)
submission = pd.DataFrame({"ID": test_ids, "TARGET": test_binary})
submission.to_csv(OUTPUT_DIR / "submission_tabular_net_optuna.csv", index=False)
print(f"Saved to {OUTPUT_DIR / 'submission_tabular_net_optuna.csv'}")

# Save probabilities for ensemble
np.save(OUTPUT_DIR / "test_proba.npy", test_preds_np)
print("Saved test probabilities for ensemble")
