#!/bin/bash
set -e

echo "=== Activating Virtual Environment ==="
source .venv/bin/activate

echo "=== Single-Stage Spectral Denoising Pipeline ==="

echo "1. Running Baseline Correction..."
python scripts/baseline.py

echo "2. Making Pairs (.npy)..."
python scripts/make_pairs.py

echo "3. Training single-stage ResUNet1D..."
python scripts/train_resunet.py

echo "4. Evaluating single-stage ResUNet1D..."
python scripts/evaluate_resunet.py

echo "=== Pipeline Completed Successfully ==="
