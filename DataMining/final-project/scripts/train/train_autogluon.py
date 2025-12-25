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
predictor = TabularPredictor(
    label="target",
    eval_metric="f1",
    path=str(MODEL_DIR)
)

predictor.fit(
    train_data=train_df,
    presets="best_quality",
    time_limit=None,
    num_gpus=1,
    ag_args_ensemble={"fold_fitting_strategy": "sequential_local"},
)

print(predictor.leaderboard())
