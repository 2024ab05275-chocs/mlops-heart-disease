import mlflow
import pandas as pd
import mlflow.sklearn
from pathlib import Path
from src.utils.config_loader import load_config
from sklearn.model_selection import cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

config = load_config()

DATA_URL = config["data"]["raw"]["url"]
RAW_BASE_PATH = Path(config["data"]["raw"]["base_path"])
RAW_FILE_PATH = Path(config["data"]["raw"]["file_path"])
PROCESSED_BASE_PATH = Path(config["data"]["processed"]["base_path"])
PROCESSED_FILE_PATH = Path(config["data"]["processed"]["file_path"])
OUTPUT_FILENAME = config["data"]["raw"]["file_name"]
COLUMNS = config["schema"]["columns"]

############################################################
# Load processed data
############################################################
df = pd.read_csv(PROCESSED_FILE_PATH)

X = df.drop("target", axis=1)
y = df["target"]

############################################################
# Feature groups
############################################################
categorical_features = [
    "sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"
]

numerical_features = [
    "age", "trestbps", "chol", "thalach", "oldpeak"
]

############################################################
# MLflow setup
############################################################
mlflow.set_experiment("Heart_Disease_Classification")

############################################################
# Logistic Regression (with pipeline)
############################################################
with mlflow.start_run(run_name="Logistic_Regression"):

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_features),
            ("cat", "passthrough", categorical_features)
        ]
    )

    lr_pipeline = Pipeline(
        steps=[
            ("preprocessing", preprocessor),
            ("model", LogisticRegression(C=1.0, penalty="l2", max_iter=1000))
        ]
    )

    scores = cross_validate(
        lr_pipeline,
        X,
        y,
        cv=5,
        scoring=["accuracy", "precision", "recall", "roc_auc"]
    )

    mlflow.log_param("model", "LogisticRegression")
    mlflow.log_param("C", 1.0)
    mlflow.log_param("penalty", "l2")
    mlflow.log_param("cv_folds", 5)

    mlflow.log_metric("accuracy", scores["test_accuracy"].mean())
    mlflow.log_metric("precision", scores["test_precision"].mean())
    mlflow.log_metric("recall", scores["test_recall"].mean())
    mlflow.log_metric("roc_auc", scores["test_roc_auc"].mean())

    # Fit pipeline on full data and log SAME object
    lr_pipeline.fit(X, y)
    mlflow.sklearn.log_model(lr_pipeline, name="logistic_regression_pipeline")

############################################################
# Random Forest
############################################################
with mlflow.start_run(run_name="Random_Forest"):

    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=5,
        random_state=42
    )

    scores = cross_validate(
        rf,
        X,
        y,
        cv=5,
        scoring=["accuracy", "precision", "recall", "roc_auc"]
    )

    mlflow.log_param("model", "RandomForest")
    mlflow.log_param("n_estimators", 200)
    mlflow.log_param("max_depth", 5)
    mlflow.log_param("cv_folds", 5)

    mlflow.log_metric("accuracy", scores["test_accuracy"].mean())
    mlflow.log_metric("precision", scores["test_precision"].mean())
    mlflow.log_metric("recall", scores["test_recall"].mean())
    mlflow.log_metric("roc_auc", scores["test_roc_auc"].mean())

    rf.fit(X, y)
    mlflow.sklearn.log_model(rf, name="random_forest_model")
