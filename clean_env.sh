#!/bin/bash

# Fail on error
set -e
cd "$(dirname "$0")"

echo "Starting full environment cleanup..."

# Remove virtual environment if it exists
if [ -d "venv" ]; then
  echo "Removing virtual environment..."
  rm -rf venv
fi

# Remove Python caches
echo "Removing Python cache directories..."
find . -type d -name "__pycache__" -exec rm -rf {} +
find . -type d -name ".ipynb_checkpoints" -exec rm -rf {} +

# Remove output folders
echo "Removing generated outputs..."
rm -rf results logs tables models

# Optionally remove processed data
if [ -d "data/processed" ]; then
  echo "Removing processed data..."
  rm -rf data/processed
fi

echo "Environment cleanup complete."
