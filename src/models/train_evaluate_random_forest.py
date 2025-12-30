import pickle
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_validate
from sklearn.ensemble import RandomForestClassifier
from src.utils.config_loader import load_config

# =====================
# Configuration
# =====================

config = load_config()

DATA_URL = config["data"]["raw"]["url"]
RAW_BASE_PATH = Path(config["data"]["raw"]["base_path"])
RAW_FILE_PATH = Path(config["data"]["raw"]["file_path"])
PROCESSED_BASE_PATH = Path(config["data"]["processed"]["base_path"])
PROCESSED_FILE_PATH = Path(config["data"]["processed"]["file_path"])
OUTPUT_FILENAME = config["data"]["raw"]["file_name"]
COLUMNS = config["schema"]["columns"]


############################################################
# 4.1 Categorical vs Numerical Features
############################################################
#       - Categorical:
#            sex, cp, fbs, restecg, exang, slope, ca, thal
#       - Numerical:
#            age, trestbps, chol, thalach, oldpeak
############################################################

df = pd.read_csv(PROCESSED_FILE_PATH)

X = df.drop("target", axis=1)
y = df["target"]

############################################################
# 4.2 Scaling
############################################################
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save scaler
with open("data/processed/standard_scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

############################################################
# Random Forest
############################################################
rf = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42)
rf.fit(X, y)

scores_rf = cross_validate(
    rf, X, y, cv=5, scoring=["accuracy", "precision", "recall", "roc_auc"]
)

rf_metrics = {
    "accuracy": scores_rf["test_accuracy"].mean(),
    "precision": scores_rf["test_precision"].mean(),
    "recall": scores_rf["test_recall"].mean(),
    "roc_auc": scores_rf["test_roc_auc"].mean(),
}

print("\n-------------------------------------------------")
print("Random Forest Metrics:")
print("-------------------------------------------------\n")

for metric, value in rf_metrics.items():
    print(f"* {metric}: {value:.3f}")

print("-------------------------------------------------\n")
