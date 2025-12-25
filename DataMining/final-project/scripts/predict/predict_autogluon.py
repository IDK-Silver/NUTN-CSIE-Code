from pathlib import Path

import pandas as pd
from autogluon.tabular import TabularPredictor

DATA_DIR = Path("blobs/raw")
MODEL_DIR = Path("blobs/models/AutogluonModels")
OUTPUT_DIR = Path("blobs/submit/autogluon")


def main() -> None:
    if not MODEL_DIR.exists():
        raise FileNotFoundError(f"Model directory not found: {MODEL_DIR}")
    test_path = DATA_DIR / "test.csv"
    if not test_path.exists():
        raise FileNotFoundError(f"Test file not found: {test_path}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    test_df = pd.read_csv(test_path)
    test_ids = test_df["ID"].copy()
    test_df = test_df.drop(columns=["ID"])

    predictor = TabularPredictor.load(MODEL_DIR)
    predictions = predictor.predict(test_df)

    output_path = OUTPUT_DIR / "submission_autogluon.csv"
    submission = pd.DataFrame({"ID": test_ids, "TARGET": predictions})
    submission.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")


if __name__ == "__main__":
    main()
