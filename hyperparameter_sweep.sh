#!/bin/bash
# Hyperparameter sweep for City A
# Test different configurations and log results

mkdir -p sweep_results

echo "Starting City A hyperparameter sweep..."

# Test 1: Baseline + 1 extra epoch
echo "Test 1: Baseline + 1 epoch..."
python run_geoformer.py train --city A --epochs 3 --batch_size 8 --grad_accum 2 --lr 3e-4
python run_geoformer.py generate --city A --checkpoint checkpoints/geoformer_cityA_small_best.pt --topk 5 --temperature 1.0
python run_geoformer.py evaluate --city A --predictions predictions/predictions_cityA.csv > sweep_results/test1_baseline_3ep.txt

# Test 2: Lower learning rate
echo "Test 2: Lower learning rate (2e-4)..."
python run_geoformer.py train --city A --epochs 2 --batch_size 8 --grad_accum 2 --lr 2e-4
python run_geoformer.py generate --city A --checkpoint checkpoints/geoformer_cityA_small_best.pt --topk 5 --temperature 1.0
python run_geoformer.py evaluate --city A --predictions predictions/predictions_cityA.csv > sweep_results/test2_lr2e4.txt

# Test 3: Warmer initialization (transfer from B)
echo "Test 3: Transfer learning from City B..."
python run_geoformer.py train --city B --epochs 2 --batch_size 8 --grad_accum 2
python run_geoformer.py train --city A --resume checkpoints/geoformer_cityB_small_best.pt --epochs 2 --batch_size 8 --grad_accum 2 --lr 1e-4
python run_geoformer.py generate --city A --checkpoint checkpoints/geoformer_cityA_small_best.pt --topk 5 --temperature 1.0
python run_geoformer.py evaluate --city A --predictions predictions/predictions_cityA.csv > sweep_results/test3_transfer_B.txt

# Test 4: Temperature sweep (inference only, no retraining)
echo "Test 4: Temperature sweep..."
for temp in 0.8 0.9 0.95 1.0 1.1; do
    echo "  Testing temperature=$temp"
    python run_geoformer.py generate --city A \
      --checkpoint checkpoints/geoformer_cityA_small_best.pt \
      --topk 5 --temperature $temp --max_users 500
    python run_geoformer.py evaluate --city A --predictions predictions/predictions_cityA.csv \
      > sweep_results/test4_temp_${temp}.txt
done

echo "Sweep complete! Results in sweep_results/"
ls -lh sweep_results/
