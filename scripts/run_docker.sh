#!/bin/bash
set -e

PORT=8000
CONTAINER_NAME="heart-disease-api"
IMAGE_NAME="heart-disease-api"

if ! docker info >/dev/null 2>&1; then
  echo "❌ Docker daemon is not running. Start Docker Desktop."
  exit 1
fi

echo "🔍 Checking if port $PORT is in use..."

# Check if port is in use
PID=$(lsof -ti tcp:$PORT || true)

if [ -n "$PID" ]; then
    echo "⚠️ Port $PORT is in use by PID(s): $PID"
    echo "🛑 Killing process(es)..."
    kill -9 $PID
    sleep 2
else
    echo "✅ Port $PORT is free"
fi

# Stop existing container if running
if docker ps -q -f name=$CONTAINER_NAME | grep -q .; then
    echo "🛑 Stopping running container: $CONTAINER_NAME"
    docker stop $CONTAINER_NAME
fi

# Remove container if exists
if docker ps -aq -f name=$CONTAINER_NAME | grep -q .; then
    echo "🧹 Removing existing container: $CONTAINER_NAME"
    docker rm $CONTAINER_NAME
fi

# Build Docker image
echo "🐳 Building Docker image..."
docker build -t $IMAGE_NAME .

# Run Docker container
echo "🚀 Starting Docker container..."
docker run -d \
    --name $CONTAINER_NAME \
    -p $PORT:8000 \
    $IMAGE_NAME

echo "✅ API is running at http://localhost:$PORT"
echo "📄 Swagger UI: http://localhost:$PORT/docs"
