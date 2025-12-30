from fastapi import FastAPI, Request
from pydantic import BaseModel
import pickle
import numpy as np
import time
import logging
import sys

from prometheus_fastapi_instrumentator import Instrumentator

# -----------------------------
# Logging Setup
# -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

logger = logging.getLogger("heart-disease-api")

# -----------------------------
# FastAPI App
# -----------------------------
app = FastAPI(title="Heart Disease Prediction API")

# Enable Prometheus metrics
Instrumentator().instrument(app).expose(app)

# -----------------------------
# Load Artifacts
# -----------------------------
logger.info("Loading scaler and models...")

with open("data/processed/standard_scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("models/logistic_regression_model.pkl", "rb") as f:
    lr_model = pickle.load(f)

with open("models/random_forest_model.pkl", "rb") as f:
    rf_model = pickle.load(f)

logger.info("Models and scaler loaded successfully")

# -----------------------------
# Feature Schema
# -----------------------------
FEATURES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak",
    "slope", "ca", "thal",
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


# -----------------------------
# Request Logging Middleware
# -----------------------------
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    duration = round(time.time() - start_time, 4)

    logger.info(
        f"{request.method} {request.url.path} | "
        f"status={response.status_code} | "
        f"time={duration}s"
    )
    return response


# -----------------------------
# Input Preparation
# -----------------------------
def prepare_input(data: HeartDiseaseInput):
    X = np.array([[getattr(data, f) for f in FEATURES]])
    return scaler.transform(X)


# -----------------------------
# Logistic Regression Endpoint
# -----------------------------
@app.post("/predict/logistic")
def predict_logistic(data: HeartDiseaseInput):
    logger.info("Inference started | model=logistic-regression")

    X_scaled = prepare_input(data)
    prediction = int(lr_model.predict(X_scaled)[0])
    confidence = float(lr_model.predict_proba(X_scaled)[0][prediction])

    logger.info(
        f"Inference completed | model=logistic-regression | "
        f"prediction={prediction} | confidence={round(confidence, 3)}"
    )

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
    logger.info("Inference started | model=random-forest")

    X_scaled = prepare_input(data)
    prediction = int(rf_model.predict(X_scaled)[0])
    confidence = float(rf_model.predict_proba(X_scaled)[0][prediction])

    logger.info(
        f"Inference completed | model=random-forest | "
        f"prediction={prediction} | confidence={round(confidence, 3)}"
    )

    return {
        "model": "Random Forest",
        "prediction": prediction,
        "confidence": round(confidence, 3),
    }
