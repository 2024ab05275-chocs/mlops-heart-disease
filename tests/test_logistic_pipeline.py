"""
Unified test file for Logistic Regression pipeline
Covers:
- Scaler artifact creation
- Scaled dataset output
- Cross-validation metrics structure
- ROC computation sanity
- generate_model_comments utility
"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
from sklearn.metrics import roc_curve, auc

# Import function under test
from src.models.train_evaluate_logistic_regression import (
    generate_model_comments,
)


# --------------------------------------------------
# Test 1: StandardScaler artifact is saved correctly
# --------------------------------------------------
def test_standard_scaler_saved():
    scaler_path = Path("data/processed/standard_scaler.pkl")

    assert scaler_path.exists(), "StandardScaler pickle file not found"

    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    assert isinstance(scaler, StandardScaler)


# --------------------------------------------------
# Test 2: Scaled dataset CSV is created and valid
# --------------------------------------------------
def test_scaled_dataset_created():
    scaled_file = Path("data/processed/heart_disease_scaled.csv")

    assert scaled_file.exists(), "Scaled dataset CSV not found"

    df = pd.read_csv(scaled_file)

    assert not df.empty, "Scaled dataframe is empty"
    assert df.shape[1] > 5, "Unexpected number of features"


# --------------------------------------------------
# Test 3: Cross-validation metric keys exist
# --------------------------------------------------
def test_cross_validation_metrics_structure():
    X, y = make_classification(
        n_samples=100,
        n_features=10,
        random_state=42
    )

    model = LogisticRegression(max_iter=500)

    scores = cross_validate(
        model,
        X,
        y,
        cv=3,
        scoring=["accuracy", "precision", "recall", "roc_auc"]
    )

    expected_keys = {
        "test_accuracy",
        "test_precision",
        "test_recall",
        "test_roc_auc"
    }

    assert expected_keys.issubset(scores.keys())


# --------------------------------------------------
# Test 4: ROC curve computation sanity
# --------------------------------------------------
def test_roc_curve_computation():
    y_true = np.array([0, 0, 1, 1])
    y_scores = np.array([0.1, 0.4, 0.35, 0.8])

    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    assert 0.0 <= roc_auc <= 1.0


# --------------------------------------------------
# Test 5: generate_model_comments output validity
# --------------------------------------------------
def test_generate_model_comments_output():
    dummy_scores = {
        "test_accuracy": np.array([0.8, 0.82, 0.81]),
        "test_precision": np.array([0.78, 0.80, 0.79]),
        "test_recall": np.array([0.75, 0.77, 0.76]),
        "test_roc_auc": np.array([0.83, 0.85, 0.84]),
    }

    comments = generate_model_comments(
        "Logistic Regression",
        dummy_scores
    )

    assert isinstance(comments, str)
    assert "Logistic Regression" in comments
    assert len(comments) > 100
