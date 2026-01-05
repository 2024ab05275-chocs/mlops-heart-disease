"""
Unified test file for Random Forest pipeline
Covers:
- Scaler artifact creation
- Processed dataset existence
- Random Forest metrics
- Cross-validation structure
- ROC computation sanity
"""
# ==================================================
# 1️⃣ LINT ERRORS (flake8 / ruff should fail here)
# ==================================================

import os, sys, math  # ❌ E401: multiple imports on one line (lint error)

def lint_error_function():
    x=1+2    # ❌ E225: missing whitespace around operator
    return  x  # ❌ E201: extra whitespace


# ==================================================
# Normal imports (pipeline will not reach all tests)
# ==================================================

import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_validate
from sklearn.metrics import roc_curve, auc

from src.models.train_evaluate_random_forest import (
    train_random_forest_pipeline
)


# --------------------------------------------------
# Test 1: StandardScaler artifact is saved correctly
# --------------------------------------------------
def test_standard_scaler_saved():
    scaler_path = Path("data/processed/standard_scaler.pkl")

    # Call pipeline to ensure scaler is created
    train_random_forest_pipeline()

    assert scaler_path.exists(), "StandardScaler pickle file not found"

    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    assert isinstance(scaler, StandardScaler)


# --------------------------------------------------
# Test 2: Processed dataset exists and valid
# --------------------------------------------------
def test_processed_dataset_exists():
    from src.utils.config_loader import load_config
    config = load_config()
    processed_file = Path(config["data"]["processed"]["file_path"])

    assert processed_file.exists(), "Processed dataset CSV not found"

    df = pd.read_csv(processed_file)
    assert not df.empty, "Processed dataframe is empty"
    assert df.shape[1] > 5, "Unexpected number of features"


# --------------------------------------------------
# Test 3: Cross-validation metric keys exist
# --------------------------------------------------
def test_cross_validation_metrics_structure():
    # Use dummy data
    X = np.random.rand(50, 10)
    y = np.random.randint(0, 2, 50)

    from sklearn.ensemble import RandomForestClassifier

    model = RandomForestClassifier(n_estimators=10,
                                   max_depth=5,
                                   random_state=42)

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

    # assert 0.0 <= roc_auc <= 1.0
    # ❌ ROC AUC is always between 0 and 1 → this WILL FAIL
    assert roc_auc > 1.0, "Intentional unit test failure: ROC AUC > 1"


# --------------------------------------------------
# Test 5: train_random_forest_pipeline metrics validity
# --------------------------------------------------
def test_train_random_forest_pipeline_metrics():
    rf, metrics, scaler = train_random_forest_pipeline()
    #for key in ["accuracy", "precision", "recall", "roc_auc"]:
        #assert key in metrics
        #assert 0.0 <= metrics[key] <= 1.0'''
    # ❌ Runtime failure: key does not exist
    print(metrics["this_key_does_not_exist"])
