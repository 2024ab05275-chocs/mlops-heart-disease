import os
import pandas as pd
import seaborn as sns
from tabulate import tabulate
import matplotlib.pyplot as plt
from pathlib import Path
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
# Obtain the dataset (Already downloaded from source)
############################################################
df = pd.read_csv(RAW_FILE_PATH, names=COLUMNS)
print("\n")
print(tabulate(df.head(), headers="keys", tablefmt="psql", showindex=False))


############################################################
# Target variable meaning:
#    0 → No heart disease
#    1,2,3,4 → Presence of heart disease

# We convert this into a binary classification problem:
############################################################
df["target"] = (df["target"] > 0).astype(int)

############################################################
# Data Cleaning and preprocessing
# a. Missing Values
#   - In the original UCI data, missing values are marked with ?.
############################################################
df.replace("?", pd.NA, inplace=True)
df = df.apply(pd.to_numeric)

null_counts = df.isnull().sum().reset_index()
null_counts.columns = ["Column", "Missing_Values"]

print("\n")
print(tabulate(null_counts, headers="keys", tablefmt="psql", showindex=False))

############################################################
# b. Use median imputation (robust to outliers):
#       - Medical data often contains outliers
#       - Median preserves central tendency better than mean
############################################################
df.fillna(df.median(), inplace=True)


############################################################
# Save Preprocessed Dataset (Clean processed directory)
############################################################

# Ensure Data  path exists
os.makedirs(os.path.dirname(PROCESSED_BASE_PATH), exist_ok=True)

# Delete all existing files in processed directory
for filename in os.listdir(PROCESSED_BASE_PATH):
    file_path = os.path.join(PROCESSED_BASE_PATH, filename)
    if os.path.isfile(file_path):
        os.remove(file_path)

# Save the preprocessed dataset
df.to_csv(PROCESSED_FILE_PATH, index=False)
print(f"\nPreprocessed dataset saved to: {PROCESSED_FILE_PATH}")

# --------------------------------------------------------
# Ensure screenshots directory exists
# --------------------------------------------------------
SCREENSHOT_DIR = "screenshots"
os.makedirs(SCREENSHOT_DIR, exist_ok=True)

# --------------------------------------------------------
# c. Exploratory Data Analysis: Class Distribution
# --------------------------------------------------------
plt.figure(figsize=(6, 4))
sns.countplot(x="target", data=df)
plt.title("Class Distribution (Heart Dis    ease)")
plt.xlabel("Target")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(f"{SCREENSHOT_DIR}/class_distribution.png")
plt.close()

# --------------------------------------------------------
# d. Feature Distribution
# --------------------------------------------------------
num_features = ["age", "trestbps", "chol", "thalach"]

df[num_features].hist(bins=20, figsize=(10, 6))
plt.suptitle("Feature Distributions")
plt.tight_layout()
plt.savefig(f"{SCREENSHOT_DIR}/feature_distributions.png")
plt.close()

# --------------------------------------------------------
# e. Correlation Heatmap
# --------------------------------------------------------
plt.figure(figsize=(12, 8))
corr = df.select_dtypes(include="number").corr()

sns.heatmap(
    corr,
    cmap="coolwarm",
    annot=False,
    linewidths=0.5
)

plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig(f"{SCREENSHOT_DIR}/correlation_heatmap.png")
plt.close()
