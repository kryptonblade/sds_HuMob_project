#!/bin/bash
# Quick test: 30 minutes to validate improvements

echo "🚀 Quick validation test (30 min)"
echo "=================================="

# 1. Inference parameter tuning on subset (10 min)
echo "\n1️⃣  Testing inference parameters on 500-user subset..."
for temp in 0.8 0.9 0.95 1.0; do
    echo "  Temperature: $temp"
    python run_geoformer.py generate \
      --checkpoint checkpoints/geoformer_cityA_small_best.pt \
      --city A --topk 5 --temperature $temp \
      --batch_size 256 --max_users 500 \
      --output_file predictions/test_temp_${temp}.csv
done

echo "\n2️⃣  Evaluating results..."
for temp in 0.8 0.9 0.95 1.0; do
    echo "  Temperature $temp:"
    python run_geoformer.py evaluate --city A \
      --predictions predictions/test_temp_${temp}.csv 2>/dev/null | grep "Mean\|Median"
done

echo "\n✅ Test complete! Check results above."
echo "Best temperature found? Use it for full training."
