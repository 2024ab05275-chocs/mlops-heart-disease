from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np

app = FastAPI(title="Heart Disease Prediction API")

# Load artifacts
with open("data/processed/standard_scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("models/logistic_regression_model.pkl", "rb") as f:
    lr_model = pickle.load(f)

with open("models/random_forest_model.pkl", "rb") as f:
    rf_model = pickle.load(f)

FEATURES = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
]


class HeartDiseaseInput(BaseModel):
    age: int
    sex: int
    cp: int
    trestbps: int
    chol: int
    fbs: int
    restecg: int
    thalach: int
    exang: int
    oldpeak: float
    slope: int
    ca: int
    thal: int


def prepare_input(data: HeartDiseaseInput):
    X = np.array([[getattr(data, f) for f in FEATURES]])
    return scaler.transform(X)


# -----------------------------
# Logistic Regression Endpoint
# -----------------------------
@app.post("/predict/logistic")
def predict_logistic(data: HeartDiseaseInput):
    X_scaled = prepare_input(data)
    prediction = int(lr_model.predict(X_scaled)[0])
    confidence = float(lr_model.predict_proba(X_scaled)[0][prediction])

    return {
        "model": "Logistic Regression",
        "prediction": prediction,
        "confidence": round(confidence, 3),
    }


# -----------------------------
# Random Forest Endpoint
# -----------------------------
@app.post("/predict/random-forest")
def predict_random_forest(data: HeartDiseaseInput):
    X_scaled = prepare_input(data)
    prediction = int(rf_model.predict(X_scaled)[0])
    confidence = float(rf_model.predict_proba(X_scaled)[0][prediction])

    return {
        "model": "Random Forest",
        "prediction": prediction,
        "confidence": round(confidence, 3),
    }
