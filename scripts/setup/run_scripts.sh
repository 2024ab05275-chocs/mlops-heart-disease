#!/bin/bash
set -e

#########################################################
# Move to project root (robust)
#########################################################

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)" 
PROJECT_ROOT="mlops-heart-disease" 
cd "$PROJECT_ROOT" 
echo "📁 Working directory: $(pwd)"

#########################################################
# Detect CI environment
#########################################################

IS_CI=false
if [[ -n "$GITHUB_ACTIONS" ]]; then
  IS_CI=true
  echo "⚙️ Running in GitHub Actions"
fi

#########################################################
# Python Setup (skip install in CI)
#########################################################

PYTHON=python3.10

if ! command -v $PYTHON &>/dev/null; then
  if [[ "$IS_CI" == true ]]; then
    echo "❌ Python 3.10 not found in CI"
    exit 1
  fi

  echo "⚠️ Installing Python 3.10 locally..."

  if [[ "$(uname)" == "Darwin" ]]; then
    brew install python@3.10
  else
    sudo apt update
    sudo apt install -y python3.10 python3-pip
  fi
fi

#########################################################
# Dependencies
#########################################################

$PYTHON -m pip install --upgrade pip
$PYTHON -m pip install -r requirements.txt

#########################################################
# 1. Data Pipeline
#########################################################

$PYTHON -m src.data.data_acquisition
$PYTHON -m src.data.preprocess

#########################################################
# 2. Model Training
#########################################################

$PYTHON -m src.models.train_evaluate_logistic_regression
$PYTHON -m src.models.train_evaluate_random_forest

#########################################################
# 3. Experiment Tracking (MLflow)
#########################################################

MLFLOW_BACKEND="./mlops-heart-disease/src/mlruns/"
MLFLOW_PORT=5000

echo "📊 MLflow backend: $MLFLOW_BACKEND"

# Start MLflow UI only in local (not CI)
if [[ "$IS_CI" == false ]]; then
  echo "🚀 Starting MLflow UI in background..."

  python -m mlflow ui \
    --backend-store-uri "$MLFLOW_BACKEND" \
    --host 127.0.0.1 \
    --port $MLFLOW_PORT \
    > mlflow_ui.log 2>&1 &

  MLFLOW_PID=$!
  echo "📌 MLflow PID: $MLFLOW_PID"

  # Give MLflow time to start
  sleep 5

  echo "🌐 MLflow UI running at http://127.0.0.1:$MLFLOW_PORT"
else
  echo "⚙️ CI detected – skipping MLflow UI startup"
fi

# Run experiment tracking logic (logging metrics, params, artifacts)
$PYTHON -m src.models.experiment_tracking

#########################################################
# 4. Model Saving
#########################################################

$PYTHON -m src.models.save_model

#########################################################
# 5. Testing & Lint
#########################################################

echo "Step 1: Linting"
chmod +x scripts/run_lint.sh
bash scripts/run_lint.sh

echo "Step 2: Running tests"
pytest tests/test_download_data.py -v
pytest tests/test_logistic_pipeline.py -v
pytest tests/test_random_forest.py -v

#########################################################
# 6. Docker Build & Run
#########################################################

docker build -t heart-disease-api .

docker run -d --rm \
  -p 8000:8000 \
  --name heart-disease-api-container \
  heart-disease-api

#########################################################
# 7. Wait for API
#########################################################

echo "⏳ Waiting for API to be ready..."
until curl -s http://localhost:8000/health > /dev/null; do
  sleep 2
done

#########################################################
# 8. API Tests
#########################################################

curl -X POST http://localhost:8000/predict/logistic \
-H "Content-Type: application/json" \
-d '{
  "age": 63,
  "sex": 1,
  "cp": 3,
  "trestbps": 145,
  "chol": 233,
  "fbs": 1,
  "restecg": 0,
  "thalach": 150,
  "exang": 0,
  "oldpeak": 2.3,
  "slope": 0,
  "ca": 0,
  "thal": 1
}'

curl -X POST http://localhost:8000/predict/random-forest \
-H "Content-Type: application/json" \
-d '{
  "age": 63,
  "sex": 1,
  "cp": 3,
  "trestbps": 145,
  "chol": 233,
  "fbs": 1,
  "restecg": 0,
  "thalach": 150,
  "exang": 0,
  "oldpeak": 2.3,
  "slope": 0,
  "ca": 0,
  "thal": 1
}'

#########################################################
#  Production Deployment 
#########################################################

chmod +x scripts/run_local_k8s_ingress.sh
./scripts/run_local_k8s_ingress.sh

chmod +x ./mlops-heart-disease/scripts/run_local_k8s_ingress.sh
./mlops-heart-disease/scripts/run_local_k8s_ingress.sh

#########################################################
# Cleanup
#########################################################

docker stop heart-disease-api-container

echo "✅ MLOps pipeline completed successfully"
