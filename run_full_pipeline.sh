#!/bin/bash
set -e

echo "=========================================="
echo "    Starting Full GeoFormer Pipeline      "
echo "=========================================="

echo "[1/3] Training Full Dataset..."
/usr/local/bin/python3 run_geoformer.py train \
  --city A \
  --epochs 2 \
  --model_size small \
  --batch_size 8 \
  --grad_accum 2

echo "[2/3] Generating Predictions for Full Dataset..."
/usr/local/bin/python3 run_geoformer.py generate \
  --checkpoint checkpoints/geoformer_cityD_small_best.pt \
  --city D \
  --topk 5 \
  --temperature 1.0 \
  --batch_size 128 --max_users 200

echo "[3/3] Evaluating Full Dataset..."
/usr/local/bin/python3 run_geoformer.py evaluate \
  --predictions predictions/predictions_cityD.csv \
  --city D  --max_users 200

echo "=========================================="
echo "          Pipeline Completed!             "
echo "=========================================="
