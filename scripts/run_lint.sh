#!/bin/bash

echo "========================================"
echo " Running Python Linting (Flake8)"
echo "========================================"

# Exit immediately if any command fails
set -e

# Activate virtual environment if exists
if [ -d ".venv" ]; then
    echo "Activating virtual environment"
    source .venv/bin/activate
fi

echo "Installing lint dependencies (if needed)"
pip install --quiet flake8

echo "Running Flake8 on src directory"
flake8 src/

echo "Linting completed successfully ✅"
