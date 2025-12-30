import os
import pickle
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
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

X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

# Save scaler
with open("data/processed/standard_scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

X_scaled_df.to_csv(f"{PROCESSED_BASE_PATH}/heart_disease_scaled.csv",
                   index=False)

############################################################
# Logistic Regression
############################################################
log_reg = LogisticRegression(max_iter=1000)
scores_lr = cross_validate(
    log_reg, X_scaled, y, cv=5, scoring=["accuracy",
                                         "precision",
                                         "recall",
                                         "roc_auc"]
)


lr_metrics = {
    "accuracy": scores_lr["test_accuracy"].mean(),
    "precision": scores_lr["test_precision"].mean(),
    "recall": scores_lr["test_recall"].mean(),
    "roc_auc": scores_lr["test_roc_auc"].mean(),
}

print("\n-------------------------------------------------")
print("Logistic Regression Metrics:")
print("-------------------------------------------------\n")

for metric, value in lr_metrics.items():
    print(f"* {metric}: {value:.3f}")

print("-------------------------------------------------\n")

############################################################
# ROC Curve - Logistic Regression
############################################################
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)


log_reg.fit(X_train, y_train)
y_prob = log_reg.predict_proba(X_test)[:, 1]


fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)


# -----------------------------
# Ensure folder exists
# -----------------------------
output_dir = "screenshots"
os.makedirs(output_dir, exist_ok=True)  # Creates folder if it doesn't exist

# -----------------------------
# Plot ROC curve
# -----------------------------
plt.plot(fpr, tpr, label=f"ROC AUC = {roc_auc:.2f}")
plt.plot([0, 1], [0, 1], "--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve – Logistic Regression")
plt.legend()

# -----------------------------
# Save plot (overwrites if exists)
# -----------------------------
output_path = os.path.join(output_dir, "roc_curve_logistic.png")
plt.savefig(output_path)  # Will overwrite if file already exists
plt.close()

print(f"✅ ROC curve saved to {output_path}")


def generate_model_comments(model_name, metrics):
    """
    Generate dynamic, exam-ready comments based on evaluation metrics.

    Parameters:
    - model_name: str
    - metrics: dict with keys:
        'accuracy', 'precision', 'recall', 'roc_auc'
    """

    acc = metrics["test_accuracy"].mean()
    prec = metrics["test_precision"].mean()
    rec = metrics["test_recall"].mean()
    auc = metrics["test_roc_auc"].mean()

    comments = []

    # ROC-AUC interpretation
    if auc >= 0.9:
        comments.append(
            f"* The {model_name} model demonstrates discriminative\n"
            f"ability with an ROC-AUC of {auc:.2f}, indicating \n"
            f"strong separation between patients heart disease.\n"
        )
    elif auc >= 0.8:
        comments.append(
            f"* The {model_name} model shows good discrimination \n"
            f"(ROC-AUC = {auc:.2f}), acceptable for clinical systems.\n"
        )
    else:
        comments.append(
            f"* The ROC-AUC of {auc:.2f} suggests limited power,\n"
            f"indicating the model may require further tuning.\n"
        )

    # Recall (medical importance)
    if rec >= 0.85:
        comments.append(
            f"* The high recall score ({rec:.2f}) indicates that \n"
            f" the model is effective at identifying patients \n"
            f" with heart disease, minimizing false negatives.\n"
        )
    elif rec >= 0.7:
        comments.append(
            f"* The recall score ({rec:.2f}) is moderate, suggesting\n"
            f"some true heart disease cases may be missed.\n"
        )
    else:
        comments.append(
            f"* The low recall score ({rec:.2f}) raises concern in a medical\n"
            f"context, as false negatives can be clinically dangerous.\n"
        )

    # Precision
    if prec >= 0.8:
        comments.append(
            f"* A precision of {prec:.2f} indicates that most positive\n"
            f"predictions made by the model are correct,\n"
            f"reducing unnecessary medical follow-ups.\n"
        )
    else:
        comments.append(
            f"* The precision score ({prec:.2f}) suggests a higher false  \n"
            f"positive rate, leading to unnecessary diagnostic procedures.\n"
        )

    # Accuracy (contextualized)
    comments.append(
        f"* The overall accuracy of {acc:.2f} interpreted cautiously,\n"
        f"as accuracy alone does not capture class imbalance.\n"
    )

    # Final summary
    comments.append(
        f"Overall, the {model_name} model provides a\n"
        f"balanced trade-off between sensitivity and specificity, making it "
        f"{'well-suited' if auc >= 0.85 and rec >= 0.8 else 'less suitable'} "
        f"for heart disease screening applications."
    )

    return "\n\n".join(comments)


print("\n")
print(generate_model_comments("Logistic Regression", scores_lr))
