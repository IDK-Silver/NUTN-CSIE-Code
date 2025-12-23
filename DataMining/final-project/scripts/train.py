import pandas as pd
from autogluon.tabular import TabularPredictor
from pathlib import Path

DATA_DIR = Path("blobs/raw")
OUTPUT_DIR = Path("blobs/output")
MODEL_DIR = Path("AutogluonModels")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load data
train_df = pd.read_csv(DATA_DIR / "train.csv")
test_df = pd.read_csv(DATA_DIR / "test.csv")

# Store test IDs, drop ID column (not a feature)
test_ids = test_df["ID"].copy()
train_df = train_df.drop(columns=["ID"])
test_df = test_df.drop(columns=["ID"])

# Train with F1 optimization
predictor = TabularPredictor(
    label="target",
    eval_metric="f1",
    path=str(MODEL_DIR)
)

predictor.fit(
    train_data=train_df,
    presets="best_quality",
    time_limit=None
)

# Predict and save submission
predictions = predictor.predict(test_df)
submission = pd.DataFrame({"ID": test_ids, "target": predictions})
submission.to_csv(OUTPUT_DIR / "submission.csv", index=False)

print(f"Submission saved to {OUTPUT_DIR / 'submission.csv'}")
print(predictor.leaderboard())
