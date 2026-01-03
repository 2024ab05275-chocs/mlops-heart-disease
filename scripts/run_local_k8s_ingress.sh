#!/bin/bash
set -e

# -----------------------------
# Configuration
# -----------------------------
APP_NAME="heart-disease-api"
IMAGE_NAME="heart-disease-api"
CONTAINER_NAME="heart-disease-api-container"
PORT=8000
INGRESS_HOST="heart.local"

# -----------------------------
# 0️⃣ Prerequisite Checks
# -----------------------------
echo -e "\n-------------------------------------"
echo "🔹 Checking prerequisites..."
echo -e "-------------------------------------\n"

if ! command -v docker >/dev/null 2>&1; then
  echo "❌ Docker CLI not found. Install Docker Desktop."
  exit 1
fi

if ! docker info >/dev/null 2>&1; then
  echo "❌ Docker daemon not running. Start Docker Desktop."
  exit 1
fi

if ! command -v kubectl >/dev/null 2>&1; then
  echo "❌ kubectl CLI not found. Install kubectl."
  exit 1
fi

if ! kubectl get nodes >/dev/null 2>&1; then
  echo "❌ Kubernetes cluster is not running. Enable Kubernetes in Docker Desktop."
  exit 1
fi

if ! kubectl get pods -n ingress-nginx >/dev/null 2>&1; then
  echo "❌ NGINX Ingress Controller not found. Install it first."
  exit 1
fi

echo -e "\n-------------------------------------"
echo "✅ All prerequisites verified"
echo -e "-------------------------------------\n"
# -----------------------------
# 1️⃣ Docker cleanup
# -----------------------------
# Stop the container if running
RUNNING_ID=$(docker ps -q -f name=$CONTAINER_NAME)
if [ -n "$RUNNING_ID" ]; then
    echo "🔹 Stopping old container..."
    docker stop $RUNNING_ID
fi

# Remove the container if exists
EXISTING_ID=$(docker ps -a -q -f name=$CONTAINER_NAME)
if [ -n "$EXISTING_ID" ]; then
    echo "🔹 Removing old container..."
    docker rm $EXISTING_ID
fi

# -----------------------------
# Optional: dynamic container name
# Uncomment if you want unique names for each run
# CONTAINER_NAME="${CONTAINER_NAME}-$(date +%s)"
# -----------------------------

# -----------------------------
# 2️⃣ Build and run Docker image
# -----------------------------
echo -e "\n-------------------------------------"
echo "🐳 Building Docker image..."
echo -e "-------------------------------------\n"
docker build -t $IMAGE_NAME .

echo -e "\n-------------------------------------"
echo "🚀 Running Docker container..."
echo -e "-------------------------------------\n"
docker run -d --name $CONTAINER_NAME -p $PORT:8000 $IMAGE_NAME

# -----------------------------
# 3️⃣ Apply Kubernetes manifests
# -----------------------------
echo -e "\n-------------------------------------"
echo "☸️ Applying Kubernetes manifests..."
echo -e "-------------------------------------\n"
kubectl apply -f k8s/

# -----------------------------
# 4️⃣ Restart Kubernetes pods
# -----------------------------
echo -e "\n-------------------------------------"
echo "♻️ Restarting Kubernetes pods..."
echo -e "-------------------------------------\n"
kubectl delete pod -l app=$APP_NAME || true
kubectl wait --for=condition=Ready pod -l app=$APP_NAME --timeout=120s

# -----------------------------
# 5️⃣ Configure /etc/hosts
# -----------------------------
if ! grep -q "$INGRESS_HOST" /etc/hosts; then
  echo "🔧 Adding $INGRESS_HOST to /etc/hosts (requires sudo)"
  sudo sh -c "echo '127.0.0.1 $INGRESS_HOST' >> /etc/hosts"
else
  echo "✅ /etc/hosts already configured"
fi

# -----------------------------
# 6️⃣ Verify Ingress
# -----------------------------
echo -e "\n-------------------------------------"
echo "🌐 Verifying Ingress..."
echo -e "-------------------------------------\n"
kubectl get ingress

# -----------------------------
# 7️⃣ Test Health Endpoint
# -----------------------------
echo -e "\n-------------------------------------"
echo "🧪 Testing health endpoint..."
echo -e "-------------------------------------\n"
curl -f http://$INGRESS_HOST/health || echo "⚠️ /health failed"

# -----------------------------
# 8️⃣ Test Logistic Regression Endpoint
# -----------------------------
echo -e "\n-------------------------------------"
echo "🧪 Testing Logistic Regression endpoint..."
echo -e "-------------------------------------\n"
curl -f -X POST http://$INGRESS_HOST/predict/logistic \
  -H "Content-Type: application/json" \
  -d '{
    "age": 52,
    "sex": 1,
    "cp": 0,
    "trestbps": 125,
    "chol": 212,
    "fbs": 0,
    "restecg": 1,
    "thalach": 168,
    "exang": 0,
    "oldpeak": 1.0,
    "slope": 2,
    "ca": 0,
    "thal": 2
  }'

echo ""

# -----------------------------
# 9️⃣ Test Random Forest Endpoint
# -----------------------------
echo -e "\n-------------------------------------"
echo "🧪 Testing Random Forest endpoint..."
echo -e "-------------------------------------\n"
curl -f -X POST http://$INGRESS_HOST/predict/random-forest \
  -H "Content-Type: application/json" \
  -d '{
    "age": 52,
    "sex": 1,
    "cp": 0,
    "trestbps": 125,
    "chol": 212,
    "fbs": 0,
    "restecg": 1,
    "thalach": 168,
    "exang": 0,
    "oldpeak": 1.0,
    "slope": 2,
    "ca": 0,
    "thal": 2
  }'

echo -e "\n-------------------------------------"
echo ""
echo "✅ Local Kubernetes + Ingress deployment, Docker container restart, and endpoint verification completed successfully"
echo "📘 Swagger UI: http://$INGRESS_HOST/docs"
echo -e "-------------------------------------\n"
