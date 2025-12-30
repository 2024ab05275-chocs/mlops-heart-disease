import os
import pickle
from src.models.train_evaluate_logistic_regression import log_reg
from src.models.train_evaluate_random_forest import (
    train_random_forest_pipeline
)

MODEL_DIR = "./models"
os.makedirs(MODEL_DIR, exist_ok=True)

# Logistic Regression
with open(os.path.join(MODEL_DIR, "logistic_regression_model.pkl"), "wb") as f:
    pickle.dump(log_reg, f)

# Random Forest
rf_model, rf_metrics, _ = train_random_forest_pipeline()

with open(os.path.join(MODEL_DIR, "random_forest_model.pkl"), "wb") as f:
    pickle.dump(rf_model, f)
