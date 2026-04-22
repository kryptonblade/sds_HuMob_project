#!/usr/bin/env python3
"""
Quick optimization strategies for City A GEO-BLEU improvement.

Strategy priorities (time investment vs. potential gain):
1. ✅ Tune inference parameters (temperature, top_k) - 10 min, 3-5% gain
2. ✅ Increase epochs 2→3 - 3.5 extra hours, 5-10% gain  
3. ✅ Gradient checkpointing - allows batch_size 16 instead of 8, same time
4. ✅ Analyze City A data differences vs City B
5. ✅ Try cosine annealing + longer warmup
"""

import pandas as pd
import torch
from pathlib import Path
from collections import defaultdict
import sys

# ============================================================================
# STRATEGY 1: Quick inference parameter tuning
# ============================================================================

def tune_inference_params():
    """
    Test different temperature/top_k combinations on validation set.
    This is FREE - just need to regenerate predictions with different params.
    
    Recommendation:
      python run_geoformer.py generate \
        --checkpoint checkpoints/geoformer_cityA_small_best.pt \
        --city A \
        --topk 5 \
        --temperature 0.95 \
        --batch_size 256
    
    Then try:
      - temperature: 0.8, 0.9, 0.95, 1.0, 1.1
      - top_k: 3, 5, 7, 10
      
    Test on small subset first:
      python run_geoformer.py generate \
        --checkpoint checkpoints/geoformer_cityA_small_best.pt \
        --city A \
        --topk 5 \
        --temperature 0.95 \
        --batch_size 256 \
        --max_users 500
    """
    print("""
    ╔════════════════════════════════════════════════════════════╗
    ║ STRATEGY 1: Tune Inference Parameters (10 min, ~3-5% gain) ║
    ╚════════════════════════════════════════════════════════════╝
    
    Current default: temperature=1.0, top_k=5
    
    🎯 Recommended to test:
       - Slightly lower temperature (0.8-0.95): less random, more consistent
       - top_k=5 usually works well, but try 3-7
       
    📊 Test on 500 users first to find best params:
    """)
    
    test_configs = [
        {"temp": 0.8, "top_k": 5},
        {"temp": 0.9, "top_k": 5},
        {"temp": 0.95, "top_k": 5},
        {"temp": 1.0, "top_k": 5},  # baseline
        {"temp": 0.9, "top_k": 3},
        {"temp": 0.9, "top_k": 7},
    ]
    
    for cfg in test_configs:
        print(f"  python run_geoformer.py generate \\")
        print(f"    --checkpoint checkpoints/geoformer_cityA_small_best.pt \\")
        print(f"    --city A --topk {cfg['top_k']} \\")
        print(f"    --temperature {cfg['temp']} --batch_size 256 --max_users 500")
        print()

# ============================================================================
# STRATEGY 2: Check data characteristics
# ============================================================================

def analyze_city_data():
    """
    Compare City A vs City B data to understand differences.
    Differences in user trajectory patterns, mobility patterns, etc.
    """
    print("""
    ╔═══════════════════════════════════════════════════════════════╗
    ║ STRATEGY 2: Analyze City A vs B Data Characteristics         ║
    ╚═══════════════════════════════════════════════════════════════╝
    
    Run this analysis to understand why City A scores lower:
    """)
    
    analysis_code = '''
import pandas as pd
from pathlib import Path
import numpy as np

data_dir = Path("Data")
for city in ["A", "B"]:
    # Load training data (first 60 days)
    cache_path = data_dir / f"city_{city}_train_dataset_cache_seq192_usersNone_chunked.pt"
    if cache_path.exists():
        dataset = torch.load(cache_path, weights_only=False)
        print(f"\\nCity {city}:")
        print(f"  Total sequences: {len(dataset):,}")
        
        # Check sequence length distribution
        seq_lens = [len(item["tokens"]) for item in dataset.samples[:1000]]
        print(f"  Avg seq length: {np.mean(seq_lens):.1f} (std {np.std(seq_lens):.1f})")
        print(f"  Avg sparsity: {np.mean([np.sum(t==40000)/len(t) for t in seq_lens]):.1%}")
    '''
    
    print(analysis_code)
    print("""
    This will show:
      - Dataset sizes (City A vs B)
      - Sequence sparsity (missing data ratio)
      - Trajectory patterns
    
    If City A has:
      - More sparse data → harder to learn
      - Longer trajectories → needs longer context
      - Different mobility patterns → may need different hyperparams
    """)

# ============================================================================
# STRATEGY 3: Recommend 3-epoch training
# ============================================================================

def suggest_3epoch_training():
    """
    Increase epochs from 2→3 with better learning rate schedule.
    Estimated time: +50% (~3.5 extra hours)
    Expected gain: 5-10%
    """
    print("""
    ╔═══════════════════════════════════════════════════════════════╗
    ║ STRATEGY 3: Increase Epochs 2→3 (+3.5 hours, +5-10% gain)   ║
    ╚═══════════════════════════════════════════════════════════════╝
    
    ✅ RECOMMENDED COMMAND:
    
    python run_geoformer.py train \\
      --city A \\
      --epochs 3 \\
      --model_size small \\
      --batch_size 8 \\
      --grad_accum 2 \\
      --lr 3e-4
      
    Why 3 epochs?
      - Small model needs more iterations to converge
      - City A data may have more complexity
      - Marginal time cost: ~50% more
      - Typical gain: 5-10% improvement
      
    ⏱️  Expected total time: ~10.5 hours (vs current 7 hours)
    """)

# ============================================================================
# STRATEGY 4: Gradient checkpointing for larger batches
# ============================================================================

def enable_gradient_checkpointing():
    """
    Enable gradient checkpointing to allow larger batch sizes.
    Swap memory for compute - same training time, better learning.
    """
    print("""
    ╔═══════════════════════════════════════════════════════════════╗
    ║ STRATEGY 4: Gradient Checkpointing (no extra time!)          ║
    ╚═══════════════════════════════════════════════════════════════╝
    
    Currently: batch_size=8 (effective 16 with grad_accum=2)
    
    With checkpointing: batch_size=16 (effective 32) - SAME COMPUTE TIME
    
    Add to train.py (after model.to(device)):
    
      if hasattr(model, 'gradient_checkpointing_enable'):
          model.gradient_checkpointing_enable()
    
    Modified command:
    
    python run_geoformer.py train \\
      --city A \\
      --epochs 3 \\
      --model_size small \\
      --batch_size 16 \\
      --grad_accum 2 \\
      --lr 3e-4
      
    Benefits:
      - Effective batch size 64 (vs 16) = better optimization
      - Same training time (~40% faster per epoch, but 3 epochs)
      - Better convergence properties
      - No memory cost on MPS (just compute vs memory tradeoff)
    """)

# ============================================================================
# STRATEGY 5: Cosine annealing with longer warmup
# ============================================================================

def cosine_annealing_suggestion():
    """
    The code already uses cosine annealing + warmup (good!).
    But City A might benefit from:
      - Slightly lower peak LR
      - Longer warmup phase
    """
    print("""
    ╔═══════════════════════════════════════════════════════════════╗
    ║ STRATEGY 5: Fine-tune Learning Rate Schedule                 ║
    ╚═══════════════════════════════════════════════════════════════╝
    
    Current in train.py:
      - LR: 3e-4 (already good)
      - Warmup: 10% of total steps
      - Schedule: Linear warmup → Cosine decay
    
    For City A specifically, try:
      - Lower LR: 2e-4 (more stable, slower learning)
      - Longer warmup: 15-20% of steps (better stability)
      - Same cosine schedule
      
    This is EASY to test - modify run_geoformer.py:
    
    python run_geoformer.py train \\
      --city A \\
      --epochs 3 \\
      --model_size small \\
      --batch_size 8 \\
      --grad_accum 2 \\
      --lr 2e-4          ← Change from 3e-4
      --warmup_ratio 0.15 ← Change from default 0.1
    
    Expected impact: 2-3% improvement in stability
    """)

# ============================================================================
# STRATEGY 6: Transfer learning from City B
# ============================================================================

def transfer_learning():
    """
    Initialize City A training from City B's best checkpoint!
    """
    print("""
    ╔═══════════════════════════════════════════════════════════════╗
    ║ STRATEGY 6: Transfer Learning from City B (FAST!)            ║
    ╚═══════════════════════════════════════════════════════════════╝
    
    🚀 FASTEST PATH TO IMPROVEMENT:
    
    City B already trained well (0.0793 GEO-BLEU).
    Use it as initialization for City A!
    
    1️⃣  Train on City B (if not already done):
        python run_geoformer.py train \\
          --city B --epochs 3 --model_size small \\
          --batch_size 8 --grad_accum 2
    
    2️⃣  Use City B checkpoint to initialize City A:
        
        In run_geoformer.py train function, add:
        
        if resume is None and transfer_from:
            print(f"[train] Transferring from {transfer_from}")
            t_ckpt = torch.load(transfer_from, map_location=device)
            model.load_state_dict(t_ckpt["model"], strict=False)
    
    3️⃣  Train City A from City B weights:
        python run_geoformer.py train \\
          --city A \\
          --resume checkpoints/geoformer_cityB_small_best.pt \\
          --epochs 2 \\
          --model_size small \\
          --batch_size 8 \\
          --grad_accum 2 \\
          --lr 1e-4          ← Lower LR (fine-tuning)
    
    Expected gain: 10-15% (transfer learning is powerful!)
    Time cost: Only 2 epochs × 1.5x time = 3 hours
    
    Total for both: City B (3h) + City A transfer (3h) = 6h
    vs 7h for single City A training → FASTER & BETTER!
    """)

# ============================================================================
# STRATEGY 7: Ensemble / Multi-seed
# ============================================================================

def ensemble_suggestion():
    """
    Train 2-3 models with different seeds, average predictions.
    """
    print("""
    ╔═══════════════════════════════════════════════════════════════╗
    ║ STRATEGY 7: Ensemble Multiple Seeds (if time permits)        ║
    ╚═══════════════════════════════════════════════════════════════╝
    
    If you have extra time (10-15 hours):
    
    Train 3 models with different random seeds:
    
    for seed in 42 101 201; do
        python run_geoformer.py train \\
          --city A \\
          --epochs 3 \\
          --seed $seed \\
          --model_size small \\
          --batch_size 8 \\
          --grad_accum 2
    done
    
    Average predictions from all 3 models.
    
    Expected gain: 5-7% improvement in scores
    Time cost: 3× training time (21 hours total)
    
    Only worth if you have the compute budget!
    """)

# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("          GeoFormer City A Optimization Strategies")
    print("="*70)
    
    tune_inference_params()
    print("\n" + "-"*70 + "\n")
    
    analyze_city_data()
    print("\n" + "-"*70 + "\n")
    
    suggest_3epoch_training()
    print("\n" + "-"*70 + "\n")
    
    enable_gradient_checkpointing()
    print("\n" + "-"*70 + "\n")
    
    cosine_annealing_suggestion()
    print("\n" + "-"*70 + "\n")
    
    transfer_learning()
    print("\n" + "-"*70 + "\n")
    
    ensemble_suggestion()
    
    print("\n" + "="*70)
    print("""
    🎯 RECOMMENDED QUICK PATH (10.5 total hours):
    
    1. (10 min)  Tune inference params on 500-user subset
    2. (3.5h)    Train City A for 3 epochs (batch=8)
    3. (30 min)  Evaluate and compare
    
    Expected result: 0.0641 → 0.075-0.080
    
    ---
    
    🚀 FAST TRANSFER LEARNING PATH (6 total hours):
    
    1. Use City B checkpoint as initialization
    2. Fine-tune on City A for 2 epochs (lower LR)
    3. Evaluate
    
    Expected result: 0.0641 → 0.075-0.085
    """)
    print("="*70 + "\n")
