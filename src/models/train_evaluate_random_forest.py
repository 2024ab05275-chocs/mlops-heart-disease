# src/model.py

import pickle
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_validate
from sklearn.ensemble import RandomForestClassifier
from src.utils.config_loader import load_config

rf = None


def train_random_forest_pipeline(
    processed_file_path: str = None,
    save_scaler_path: str = "data/processed/standard_scaler.pkl",
) -> dict:
    # Load config if file path not provided
    if not processed_file_path:
        config = load_config()
        processed_file_path = config["data"]["processed"]["file_path"]

    # Load dataset
    df = pd.read_csv(processed_file_path)

    X = df.drop("target", axis=1)
    y = df["target"]

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Save scaler
    if save_scaler_path:
        Path(save_scaler_path).parent.mkdir(parents=True, exist_ok=True)
        with open(save_scaler_path, "wb") as f:
            pickle.dump(scaler, f)

    # Train Random Forest
    rf = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42)
    rf.fit(X_scaled, y)

    # Cross-validation metrics
    scores_rf = cross_validate(
        rf, X_scaled, y, cv=5, scoring=["accuracy",
                                        "precision",
                                        "recall",
                                        "roc_auc"]
    )

    rf_metrics = {
        "accuracy": scores_rf["test_accuracy"].mean(),
        "precision": scores_rf["test_precision"].mean(),
        "recall": scores_rf["test_recall"].mean(),
        "roc_auc": scores_rf["test_roc_auc"].mean(),
    }

    # Print metrics
    print("\n-------------------------------------------------")
    print("Random Forest Metrics:")
    print("-------------------------------------------------\n")
    for metric, value in rf_metrics.items():
        print(f"* {metric}: {value:.3f}")
    print("-------------------------------------------------\n")

    return rf, rf_metrics, scaler


# Optional main guard to run as script
if __name__ == "__main__":
    rf, rf_metrics, scaler = train_random_forest_pipeline()
    rf = rf
