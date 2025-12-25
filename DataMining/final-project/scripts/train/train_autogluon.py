import json

import pandas as pd
from autogluon.tabular import TabularPredictor
from pathlib import Path

DATA_DIR = Path("blobs/raw")
MODEL_DIR = Path("blobs/models/AutogluonModels")

MODEL_DIR.mkdir(parents=True, exist_ok=True)

train_path = DATA_DIR / "train.csv"
if not train_path.exists():
    raise FileNotFoundError(f"Train file not found: {train_path}")

# Load data
train_df = pd.read_csv(train_path)
train_df = train_df.drop(columns=["ID"])

# Train with F1 optimization
predictor = TabularPredictor(label="target", eval_metric="f1", path=str(MODEL_DIR))


def get_num_gpus():
    try:
        import torch
    except Exception:
        return 0
    return torch.cuda.device_count()


num_gpus = get_num_gpus()

predictor.fit(
    train_data=train_df,
    presets="best_quality",
    time_limit=None,
    num_gpus=num_gpus,
    ag_args_ensemble={"fold_fitting_strategy": "sequential_local"},
)

# Get leaderboard and best model info
leaderboard = predictor.leaderboard()
print(leaderboard)

best_model = predictor.model_best
best_score = leaderboard.loc[leaderboard["model"] == best_model, "score_val"].values[0]

# Save info.json
info = {
    "best_model": best_model,
    "cv_score": float(best_score),
    "eval_metric": "f1",
}
with open(MODEL_DIR / "info.json", "w") as f:
    json.dump(info, f, indent=2)

print(f"\nBest model: {best_model}")
print(f"CV F1 score: {best_score:.4f}")
print(f"Saved model and info to {MODEL_DIR}")
