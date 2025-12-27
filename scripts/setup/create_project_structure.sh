#!/bin/bash

# Project root name
PROJECT_NAME="mlops-heart-disease"

echo "Creating project structure: $PROJECT_NAME"

# Root directory
mkdir -p $PROJECT_NAME
cd $PROJECT_NAME || exit

# Root files
touch README.md .gitignore requirements.txt environment.yml

# Data directories
mkdir -p data/raw data/processed
touch data/raw/README.md
touch data/raw/download_data.py
touch data/data_dictionary.md
touch data/processed/heart_clean.csv

# Notebooks
mkdir -p notebooks
touch notebooks/01_eda.ipynb
touch notebooks/02_feature_engineering.ipynb
touch notebooks/03_model_training_lr.ipynb
touch notebooks/04_model_training_rf.ipynb
touch notebooks/05_inference_demo.ipynb

# Source code structure
mkdir -p src/{config,data,models,api,utils}
touch src/__init__.py
touch src/config/config.yaml
touch src/data/load_data.py
touch src/data/preprocess.py
touch src/models/{train_lr.py,train_rf.py,evaluate.py,predict.py}
touch src/api/{main.py,schemas.py,utils.py}
touch src/utils/logger.py

# Tests
mkdir -p tests
touch tests/__init__.py
touch tests/{test_data.py,test_models.py,test_api.py}

# Docker
mkdir -p docker
touch docker/Dockerfile
touch docker/docker-compose.yml

# GitHub Actions
mkdir -p .github/workflows
touch .github/workflows/ci-cd.yml

# Deployment
mkdir -p deployment/kubernetes
touch deployment/kubernetes/{deployment.yaml,service.yaml,ingress.yaml}

mkdir -p deployment/helm/heart-disease/templates
touch deployment/helm/heart-disease/Chart.yaml
touch deployment/helm/heart-disease/values.yaml
touch deployment/helm/heart-disease/templates/{deployment.yaml,service.yaml,ingress.yaml}

# Screenshots
mkdir -p screenshots
touch screenshots/{api_swagger.png,kubernetes_pods.png,github_actions_pipeline.png,model_metrics.png}

# Reports
mkdir -p reports/figures
touch reports/final_report.docx
touch reports/figures/roc_curve.png

# Scripts
mkdir -p scripts
touch scripts/{train.sh,build_docker.sh,deploy_k8s.sh}

echo "✅ Project structure created successfully!"
