# ---------------------------------------------------------
# Use a lightweight official Python 3.10 image
# Slim reduces image size and attack surface
# ---------------------------------------------------------
FROM python:3.10-slim

# ---------------------------------------------------------
# Set the working directory inside the container
# All subsequent commands will run from /app
# ---------------------------------------------------------
WORKDIR /app

# ---------------------------------------------------------
# Copy only requirements.txt first
# This enables Docker layer caching:
# dependencies are reinstalled ONLY if requirements.txt changes
# ---------------------------------------------------------
COPY requirements.txt .

# ---------------------------------------------------------
# Install Python dependencies required for:
# - FastAPI (API serving)
# - Uvicorn (ASGI server)
# - ML libraries (sklearn, numpy, pandas)
# --no-cache-dir reduces image size
# ---------------------------------------------------------
RUN pip install --no-cache-dir -r requirements.txt

# ---------------------------------------------------------
# Copy application source code (FastAPI app)
# Contains /predict endpoints
# ---------------------------------------------------------
COPY src/serving ./src/serving

# ---------------------------------------------------------
# Copy trained ML model artifacts
# These .pkl files are created during training phase
# ---------------------------------------------------------
COPY models ./models

# ---------------------------------------------------------
# Copy preprocessing artifacts (scaler)
# Ensures the same preprocessing used during training
# is applied during inference
# ---------------------------------------------------------
COPY data/processed ./data/processed

# ---------------------------------------------------------
# Expose port 8000
# This is the port on which FastAPI/Uvicorn runs
# ---------------------------------------------------------
EXPOSE 8000

# ---------------------------------------------------------
# Start the FastAPI application using Uvicorn
# --host 0.0.0.0 allows access from outside the container
# --port 8000 matches the exposed port
# ---------------------------------------------------------
CMD ["uvicorn", "src.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
