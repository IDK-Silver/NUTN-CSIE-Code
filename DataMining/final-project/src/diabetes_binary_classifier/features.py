import pandas as pd


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add engineered features for diabetes prediction."""
    df = df.copy()
    df["metabolic_risk"] = (
        df["HighBP"] + df["HighChol"] + (df["BMI"] > 30).astype(int)
    )
    df["cardio_risk"] = (
        df["Stroke"] + df["HeartDiseaseorAttack"] + df["HighBP"] + df["HighChol"]
    )
    df["lifestyle"] = (
        df["PhysActivity"]
        + df["Fruits"]
        + df["Veggies"]
        - df["Smoker"]
        - df["HvyAlcoholConsump"]
    )
    df["health_gap"] = df["GenHlth"] - (df["PhysHlth"] + df["MentHlth"]) / 30
    df["bmi_cat"] = pd.cut(
        df["BMI"],
        bins=[0, 18.5, 25, 30, 35, 100],
        labels=[0, 1, 2, 3, 4],
    ).astype(int)
    df["age_bmi"] = df["Age"] * df["BMI"]
    df["age_health"] = df["Age"] * df["GenHlth"]
    df["age_cardio"] = df["Age"] * df["cardio_risk"]
    return df
