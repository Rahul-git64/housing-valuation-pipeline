#!/usr/bin/env sh
set -e

if [ ! -f "artifacts/models/model_bundle.joblib" ]; then
    echo "Model artifacts not found; running initial training pipeline..."
    python scripts/train.py
fi

exec uvicorn housing_valuation.api.main:app --host 0.0.0.0 --port "${PORT:-8000}"
