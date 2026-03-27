#!/bin/bash
set -e

echo "=== Activating Virtual Environment ==="
source .venv/bin/activate

echo "=== Spectral Denoising Pipeline ==="

echo "1. Running Baseline Correction..."
python scripts/baseline.py

echo "2. Making Pairs (.npy)..."
python scripts/make_pairs.py

echo "3. Loading Dataset & QC (for downstream tasks)..."
python scripts/load_qc.py

echo "4. Training ResNetThreshold1D..."
python scripts/train_resnet_threshold.py

echo "5. Evaluating ResNetThreshold1D..."
python scripts/evaluate_resnet_threshold.py

echo "6. Running Downstream Evaluation..."
python scripts/downstream_evaluation.py

echo "=== Pipeline Completed Successfully ==="
