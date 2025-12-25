import json
import pickle

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path

from diabetes_binary_classifier.features import add_features

# Config
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = Path("blobs/raw")
MODEL_DIR = Path("blobs/models/tabular_multi_seed")
OUTPUT_DIR = Path("blobs/submit/tabular_multi_seed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


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
seeds = info["seeds"]
threshold = info["threshold"]
in_features = info["in_features"]
print(f"Loaded model info: n_folds={n_folds}, seeds={seeds}, threshold={threshold:.3f}")

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
X_test_tensor = torch.tensor(X_test).to(DEVICE)

# Load models and predict
all_test_preds = []

for seed in seeds:
    test_preds = np.zeros(len(X_test))

    for fold in range(n_folds):
        model_path = MODEL_DIR / f"tabular_net_seed{seed}_fold{fold}.pt"
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        model = TabularNet(in_features).to(DEVICE)
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model.eval()

        with torch.no_grad():
            preds = torch.sigmoid(model(X_test_tensor)).cpu().numpy()
            test_preds += preds / n_folds

    all_test_preds.append(test_preds)
    print(f"Loaded seed {seed}")

final_test = np.mean(all_test_preds, axis=0)

# Apply threshold and save
test_binary = (final_test > threshold).astype(int)
submission = pd.DataFrame({"ID": test_ids, "TARGET": test_binary})
submission.to_csv(OUTPUT_DIR / "submission_tabular_net_multi_seed.csv", index=False)
print(f"Saved to {OUTPUT_DIR / 'submission_tabular_net_multi_seed.csv'}")

# Save probabilities for ensemble
np.save(OUTPUT_DIR / "test_proba.npy", final_test)
print("Saved test probabilities for ensemble")
