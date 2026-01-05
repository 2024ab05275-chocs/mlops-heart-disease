# src/model.py

import matplotlib.pyplot as plt
import os
from src.utils.config_loader import load_config
import pickle
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc
import matplotlib
matplotlib.use("Agg")  # Headless backend


rf = None


def train_random_forest_pipeline(
    processed_file_path: str = None,
    save_scaler_path: str = "data/processed/standard_scaler.pkl",
    save_plots_dir: str = "screenshots",
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

    # -----------------------------
    # Generate ROC Curve
    # -----------------------------
    os.makedirs(save_plots_dir, exist_ok=True)

    # Split for ROC plot
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    probs = rf.predict_proba(X_test)[:, 1]  # Positive class probabilities
    fpr, tpr, _ = roc_curve(y_test, probs)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC AUC = {roc_auc:.2f}")
    plt.plot([0, 1], [0, 1], "--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve – Random Forest")
    plt.legend()
    plt.savefig(os.path.join(save_plots_dir, "roc_curve_random_forest.png"))
    plt.close()
    return rf, rf_metrics, scaler


# Optional main guard to run as script
if __name__ == "__main__":
    rf, rf_metrics, scaler = train_random_forest_pipeline()
