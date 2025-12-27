"""
Download raw Heart Disease dataset from a remote URL.

Usage:
    python data/raw/download_data.py

The file will be saved in data/raw/
"""

import os
import ssl
import urllib.request
from pathlib import Path
from src.utils.config_loader import load_config

# =====================
# Configuration
# =====================

config = load_config()

DATA_URL = config["data"]["raw"]["url"]
RAW_BASE_PATH = Path(config["data"]["raw"]["base_path"])
RAW_FILE_PATH = Path(config["data"]["raw"]["file_path"])
OUTPUT_FILENAME = config["data"]["raw"]["file_name"]
COLUMNS = config["schema"]["columns"]

OUTPUT_DIR = os.path.dirname(__file__)
OUTPUT_PATH = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)

print(RAW_BASE_PATH)

def download_data():
    if os.path.exists(OUTPUT_PATH):
        print(f"\n[INFO] Dataset already exists at {OUTPUT_PATH}")
        return
    
    # Delete all CSV files except the expected raw file
    for filename in os.listdir(RAW_BASE_PATH):
        file_path = os.path.join(RAW_BASE_PATH, filename)
        if (
            filename.endswith(".csv")
            and filename != OUTPUT_FILENAME
            and os.path.isfile(file_path)
        ):
            os.remove(file_path)

    print("\n[INFO] Downloading dataset...")

    # Handle SSL certificate issues (common in some environments)
    ssl._create_default_https_context = ssl._create_unverified_context

    try:
        urllib.request.urlretrieve(DATA_URL, OUTPUT_PATH)
        print(f"\n[SUCCESS] Dataset downloaded to {OUTPUT_PATH}")
    except Exception as e:
        print(f"\n[ERROR] Failed to download dataset: {e}")
        raise


if __name__ == "__main__":
    download_data()
