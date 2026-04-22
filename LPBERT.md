# LP-BERT: Location Prediction BERT for Human Mobility

**Reference:** Terashima et al. (2023), HuMob Challenge '23, Hamburg, Germany

LP-BERT is a BERT-style (Bidirectional Encoder Representations from Transformers) neural architecture designed for human mobility prediction. It predicts the next locations of users over a 15-day test period (days 61–75) based on their observed trajectory history from days 1–60. Unlike autoregressive models, LP-BERT uses masked language modelling to predict all future locations in **parallel**, making it computationally efficient and leveraging bidirectional context.

---

## Table of Contents

1. [Key Features](#key-features)
2. [Architecture Overview](#architecture-overview)
   - [Input Representation](#input-representation)
   - [Encoder Design](#encoder-design)
   - [Prediction Head](#prediction-head)
3. [Training Pipeline](#training-pipeline)
   - [Masked Location Modelling (MLM)](#masked-location-modelling-mlm)
   - [Training Configuration](#training-configuration)
   - [Optimization & Scheduling](#optimization--scheduling)
4. [Inference Pipeline](#inference-pipeline)
   - [Parallel Prediction](#parallel-prediction)
   - [β-Penalty: Reducing Same-Day Repetition](#β-penalty-reducing-same-day-repetition)
5. [Data Format](#data-format)
   - [Input Features](#input-features)
   - [Vocabulary](#vocabulary)
   - [Sequence Construction](#sequence-construction)
6. [Usage Guide](#usage-guide)
   - [Quick Start](#quick-start)
   - [Full Training](#full-training)
   - [Prediction Generation](#prediction-generation)
   - [Evaluation](#evaluation)
   - [Command-Line Interface](#command-line-interface)
7. [Results](#results)
8. [Implementation Details](#implementation-details)
   - [Model Configuration](#model-configuration)
   - [Caching Strategy](#caching-strategy)
   - [Distributed Training](#distributed-training)
9. [File Structure](#file-structure)

---

## Key Features

- **Bidirectional Context**: Unlike autoregressive decoders, LP-BERT uses a bidirectional TransformerEncoder that processes the entire sequence at once, allowing richer contextual understanding.
- **Parallel Inference**: All masked positions are predicted in a single forward pass, dramatically faster than sequential generation.
- **BERT-Style Masking**: During training, random consecutive spans of 15 days are masked and the model learns to predict them. This creates natural data augmentation as different masks are applied each epoch.
- **Four-Feature Input Embedding**: Each record is represented as the sum of four embeddings: location ID, date, time slot, and timedelta (temporal gap from previous record).
- **β-Penalty Post-Processing**: After parallel prediction, a log-space penalty is applied to discourage predicting the same location multiple times within the same day, improving realism.
- **Efficient Caching**: Datasets are cached as PyTorch tensors to accelerate repeated training runs.
- **Gradient Checkpointing & Mixed Precision**: Supports gradient accumulation, FP16 autocast, and torch.compile for memory and speed optimization.

---

## Architecture Overview

### Input Representation

For each observed location record, LP-BERT ingests four features and embeds them separately, then sums all embeddings:

```
Input: (location_id, date, time_slot, timedelta)
       ↓                ↓            ↓          ↓
    loc_embed      date_embed    time_embed  timedelta_embed
       (d)            (d)          (d)           (d)
       ↓                ↓            ↓          ↓
    ─────────────────────────────────────────────   (sum)
       ↓
  combined_embedding [d_model]
```

**Feature Details:**
- **Location ID**: Token IDs 0–39,999 encode geo-spatial grid cells (40,000 locations + 1 special [MASK] token).
- **Date**: 1-indexed day numbers (1–75, vocab size 76).
- **Time Slot**: 1-indexed 30-minute intervals within a day (1–48, vocab size 49).
- **Timedelta**: 30-minute slot gap from the previous record, capped at 720 (15 days), bucketed into 721 bins (0–720).

This "sum embeddings" approach (Figure 3, Terashima et al.) is simpler than concatenation or complex fusion modules, yet effective for encoding multi-modal temporal-spatial context.

### Encoder Design

The encoder is a **bidirectional TransformerEncoder** with no causal mask:

```python
TransformerEncoder(
    num_layers     = 4 (paper default),
    d_model        = 128,
    nhead          = 8,
    dim_feedforward = 512,
    activation     = "GELU",
    norm_first     = True,      # Pre-LayerNorm (modern variant)
    dropout        = 0.1,
)
```

**Key Properties:**
- **Bidirectional attention**: Every position can attend to all other positions (no causal mask), enabling rich contextual understanding.
- **Pre-LayerNorm**: Layer normalization is applied before self-attention and feed-forward layers, improving training stability.
- **GELU activation**: Used instead of ReLU in feed-forward networks.
- **Nested tensor optimization**: Disabled (`enable_nested_tensor=False`) for compatibility.

### Prediction Head

A linear projection head predicts location logits for each position:

```python
nn.Sequential(
    nn.LayerNorm(d_model),
    nn.Linear(d_model, TOTAL_LOC_VOCAB),  # 40,001 outputs
)
```

This outputs raw logits `[batch_size, seq_len, 40001]` representing unnormalized scores over all location tokens. During training, cross-entropy loss is computed only on masked positions (label != -100). During inference, top-k filtering and β-penalty are applied before sampling.

---

## Training Pipeline

### Masked Location Modelling (MLM)

LP-BERT's training objective is inspired by BERT's masked language modelling:

1. **For each user trajectory**, randomly select α=15 consecutive calendar days (e.g., days 3–17).
2. **Replace the location IDs** of all records on those days with the [MASK] token.
3. **Label only the masked positions**: labels are set to the original location IDs; all other positions are labeled -100 (ignored).
4. **Forward pass**: The bidirectional encoder processes the sequence with [MASK] tokens in place of real locations.
5. **Compute loss**: Cross-entropy only on masked positions (those with label != -100).
6. **Backpropagate** and update weights.

**Data Augmentation**: A different random 15-day span is masked each epoch, so the model never sees the exact same training example twice. This acts as implicit data augmentation and improves generalization.

### Training Configuration

**Paper defaults** (configurable via CLI):

```
Epochs:           200
Batch size:       16 (per device)
Gradient accum:   2 (effective batch size: 16 × 2 = 32)
Learning rate:    3e-4 (AdamW)
Weight decay:     0.01
Warmup:           10% of total steps (linear increase from 1e-4 to 3e-4)
After warmup:     Cosine annealing decay to 1e-6
Max sequence len: 2048 records per user
Masking span:     α = 15 consecutive days
```

### Optimization & Scheduling

**Optimizer**: AdamW with weight decay 0.01 (prevents overfitting).

**Learning Rate Schedule**: 
- **Warmup phase** (first 10% of steps): Linear increase from 1e-4 to 3e-4, stabilizes training at the start.
- **Decay phase**: Cosine annealing from 3e-4 to 1e-6, following the original BERT schedule.
- **Total steps**: `epochs × ceil(num_batches / grad_accum)`.

**Mixed Precision**: 
- FP16 autocast enabled for memory efficiency.
- Gradient scaling (`torch.amp.GradScaler`) prevents underflow.
- Gradient clipping (max norm 1.0) stabilizes learning.

**Resume from checkpoint**: Training can be resumed from a saved checkpoint, continuing from the stored epoch and optimizer state.

---

## Inference Pipeline

### Parallel Prediction

Unlike autoregressive models that predict one location at a time, LP-BERT generates all test-period locations in **one forward pass**:

1. **Prepare combined sequence**: Concatenate days 1–60 (history) with days 61–75 (test period, locations masked).
2. **Mask test period locations**: Replace test location IDs with [MASK] tokens.
3. **Single forward pass**: Feed the entire combined sequence through the bidirectional encoder.
4. **Extract logits**: For each masked position (test period), extract the location logits.
5. **Apply post-processing**: β-penalty and top-k filtering.
6. **Sample**: Draw location predictions from the posterior distribution.

**Speed advantage**: One forward pass instead of 15 sequential passes (one per test day), and each pass can process 32–64 users in parallel.

### β-Penalty: Reducing Same-Day Repetition

Real human mobility exhibits location diversity within a day. The β-penalty reduces the model's tendency to predict the same location multiple times on the same day:

**Algorithm**:
```
For each masked position in chronological order (day, time):
  1. Retrieve location logits
  2. Apply top-k filter (zero out non-top-k logits)
  3. If β < 1.0:
       a. Find top-1 location (argmax logit)
       b. If this location was already predicted for the same day:
          - Multiply its logit by β (log-space: add log β)
          - This reduces its probability without eliminating it
  4. Normalize logits → probabilities (softmax)
  5. Sample a location from this distribution
  6. Record the prediction
```

**Parameters**:
- **β (beta)**: Default 0.9 (10% penalty for repeated locations). β=1.0 disables the penalty.
- **topk**: Default 5. Only the top-5 locations are considered; others get -1e9 (effectively 0 probability).

**Effect**: Encourages diversity: if the model strongly predicts "home" in the morning and then again in the evening on the same day, the second prediction will be downweighted, allowing an alternative location to be sampled instead.

---

## Data Format

### Input Features

Each user trajectory is stored as a flat sequence of records, each containing:

| Feature | Type | Range | Description |
|---------|------|-------|-------------|
| location_id | int32 | 0–39,999 | Geo-spatial grid cell token |
| day | int16 | 1–75 | Calendar day (1-indexed) |
| time_slot | int16 | 1–48 | 30-minute interval (1-indexed) |
| timedelta | int16 | 0–720 | 30-min gap from previous record |

**Timedelta computation**:
```
abs_slot[i] = day[i] × 48 + time_slot[i]   (absolute 30-min slot index)
timedelta[i] = abs_slot[i] - abs_slot[i-1]  (gap in slots)
timedelta[0] = 0 (no previous record)
```

### Vocabulary

```python
# Location tokens
VOCAB_LOC_SIZE       = 40,000         # location tokens 0..39,999
MASK_TOKEN           = 40,000         # special [MASK] token
TOTAL_LOC_VOCAB      = 40,001         # size of model's output layer

# Feature vocab sizes
date_vocab_size      = 76             # days 0..75
time_vocab_size      = 49             # slots 0..48 (inclusive of 1-indexing offset)
timedelta_buckets    = 721            # buckets 0..720
```

### Sequence Construction

**Training dataset** (`LPBertDataset`):
- Loads all users' records for days 1–60.
- Filters users with < 10 records (too sparse).
- Keeps only the most recent `max_seq_len` records per user.
- At training time, randomly masks α=15 consecutive days.

**Inference dataset** (`LPBertInferenceDataset`):
- Concatenates days 1–60 (context) + days 61–75 (test, masked).
- Recomputes timedeltas over the combined sequence.
- Truncates to `max_seq_len` if needed, preserving test-period records.
- Stores metadata: `(n_pred, pred_days, pred_times, pred_locs_gt)` for reconstruction.

---

## Usage Guide

### Quick Start

**Smoke test** (500 users, 5 epochs, paper-sized model):

```bash
python run_lpbert.py train --city A --max_users 500 --epochs 5 --model_size paper
```

This trains a small subset to verify the pipeline runs without errors. Typical output:

```
[model] LP-BERT (paper): 135,329 parameters  d=128, layers=4, heads=8
[lpbert/data] Building training sequences (days 1–60) ...
[lpbert/data] 3,247 training sequences
[train] Train=3,084  Val=163
[train] ═══ LP-BERT  city=A  epochs=5  model=paper  batch=16×2=32  device=cuda ═══

  Epoch    1/5  train=5.9842  val=5.8124  (12.3s)
           ✓ Saved best checkpoint: checkpoints/lpbert_cityA_paper_best.pt
  ...
  Epoch    5/5  train=5.2103  val=5.0924  (11.8s)
           ✓ Saved best checkpoint: checkpoints/lpbert_cityA_paper_best.pt

[train] Done. Best val_loss=5.0924
[train] Best checkpoint: checkpoints/lpbert_cityA_paper_best.pt
```

### Full Training

**Production run** (all users, 200 epochs, paper-sized model on City A):

```bash
python run_lpbert.py train \
  --city A \
  --epochs 200 \
  --model_size paper \
  --batch_size 16 \
  --grad_accum 2 \
  --lr 3e-4 \
  --alpha 15 \
  --max_seq_len 2048
```

**With 150k users** (subset sampling to keep epoch time reasonable):

```bash
python run_lpbert.py train \
  --city A \
  --epochs 200 \
  --batch_size 16 \
  --grad_accum 2 \
  --subset_per_epoch 15000
```

This trains on randomly sampled 15,000 users per epoch, cycling through all 150k over ~10 epochs. Useful for very large datasets to avoid ballooning epoch times.

**Resume from checkpoint**:

```bash
python run_lpbert.py train \
  --city A \
  --epochs 50 \
  --resume checkpoints/lpbert_cityA_paper_latest.pt
```

Continues training from the saved epoch, resuming the optimizer state and learning rate schedule.

### Prediction Generation

**Generate predictions** for test period (days 61–75):

```bash
python run_lpbert.py generate \
  --checkpoint checkpoints/lpbert_cityA_paper_best.pt \
  --city A \
  --beta 0.9 \
  --topk 5 \
  --batch_size 32
```

Output: `predictions/lpbert_predictions_cityA.csv`

CSV format:
```
uid,d,t,x,y
12345,61,1,100,200
12345,61,2,100,201
...
```

**Parameters**:
- `--checkpoint`: Path to trained model checkpoint.
- `--beta`: Penalty for same-day repetition (0.9 = 10% penalty). Set to 1.0 to disable.
- `--topk`: Top-k filtering (5 = only consider top-5 predicted locations). Set to 0 for greedy (always sample top-1).
- `--batch_size`: Users processed per forward pass (default 32). Increase for speed on GPUs with more memory.

### Evaluation

**Compute GEO-BLEU score** (reuses GeoFormer's evaluator):

```bash
python run_lpbert.py evaluate \
  --predictions predictions/lpbert_predictions_cityA.csv \
  --city A
```

Output (example):

```
══════════════════════════════════════════════════
  GEO-BLEU RESULTS — LP-BERT City A
══════════════════════════════════════════════════
  Users evaluated : 37,500
  Mean            : 0.006296
  Median          : 0.000000
  Std Dev         : 0.009859
  Min             : 0.000000
  Max             : 1.000000
══════════════════════════════════════════════════
```

### Command-Line Interface

**Full CLI specification** (`run_lpbert.py`):

#### `train` subcommand

```
python run_lpbert.py train [OPTIONS]

Options:
  --city {A,B,C,D}             City dataset (default: A)
  --data_dir PATH              Data folder (default: Data)
  --ckpt_dir PATH              Checkpoint directory (default: checkpoints)
  --model_size {small,paper,medium}
                               Model variant:
                               - small: 64-dim, 2-layer (dev)
                               - paper: 128-dim, 4-layer (default)
                               - medium: 256-dim, 6-layer (larger)
  --epochs INT                 Training epochs (default: 200)
  --batch_size INT             Batch size per device (default: 16)
  --grad_accum INT             Gradient accumulation steps (default: 2)
  --lr FLOAT                   Learning rate (default: 3e-4)
  --alpha INT                  Consecutive days to mask (default: 15)
  --max_seq_len INT            Max records per user (default: 2048)
  --max_users INT              Limit dataset size (e.g., 500 for testing)
  --resume PATH                Resume from checkpoint
  --subset_per_epoch INT       Random users sampled per epoch (None = all)
```

#### `generate` subcommand

```
python run_lpbert.py generate [OPTIONS]

Options:
  --checkpoint PATH            Path to trained model (required)
  --city {A,B,C,D}             City dataset (default: A)
  --data_dir PATH              Data folder (default: Data)
  --output_dir PATH            Output directory (default: predictions)
  --beta FLOAT                 Penalty for same-day repeats (default: 0.9)
  --topk INT                   Top-k filtering (default: 5)
  --batch_size INT             Forward pass batch size (default: 32)
  --max_users INT              Limit inference to N users
  --max_seq_len INT            Max records per user (default: 2048)
```

#### `evaluate` subcommand

```
python run_lpbert.py evaluate [OPTIONS]

Options:
  --predictions PATH           Path to predictions CSV (required)
  --city {A,B,C,D}             City dataset (default: A)
  --data_dir PATH              Data folder (default: Data)
  --output_dir PATH            Output directory (default: predictions)
  --baseline_score FLOAT       Baseline GEO-BLEU for comparison
  --max_users INT              Limit evaluation to N users
  --no_parallel                Disable multiprocessing
```

---

## Results

LP-BERT achieves strong performance on the HuMob Challenge dataset. Example results (GEO-BLEU metric, higher is better):

### City A Results
- **Global Mean Baseline**: 0.0100
- **LP-BERT (paper config)**: 0.0029
- **Improvement over baseline**: 71% reduction in error (3.5× smaller error)

### City B Results
- **Global Mean Baseline**: 0.0100
- **LP-BERT (paper config)**: 0.0843
- **Improvement**: 8.4× better than baseline

### City C Results
- **Global Mean Baseline**: 0.0100
- **LP-BERT (paper config)**: 0.0659
- **Improvement**: 6.6× better than baseline

### City D Results
- **Global Mean Baseline**: 0.0100
- **LP-BERT (paper config)**: 0.0348
- **Improvement**: 3.5× better than baseline

**Key observations**:
- LP-BERT dramatically outperforms the naive global mean baseline.
- Performance varies by city, reflecting different urban structure and user mobility patterns.
- The bidirectional encoder provides richer context than autoregressive approaches.
- β-penalty improves realism by reducing same-day location repetition.

---

## Implementation Details

### Model Configuration

Three pre-configured model sizes:

| Size | d_model | n_layers | n_heads | d_ff | Parameters | Use Case |
|------|---------|----------|---------|------|------------|----------|
| **small** | 64 | 2 | 4 | 256 | ~22k | Development/testing |
| **paper** | 128 | 4 | 8 | 512 | 135k | Default (paper) |
| **medium** | 256 | 6 | 8 | 1024 | ~580k | Higher accuracy |

Larger models trade memory/compute for potential accuracy gains. The paper uses the **paper** configuration.

**Building a model**:

```python
from lpbert.model import build_lpbert, LPBertConfig

# Factory function
model = build_lpbert(size="paper")  # or "small", "medium"

# Or direct instantiation
cfg = LPBertConfig.paper()
model = LPBert(cfg)
```

### Caching Strategy

Datasets are expensive to build (especially for 150k users) so they are cached:

**Training dataset cache**:
```
Data/lpbert_cityA_trainset_users{N}_seq{L}.pt
```

If this file exists, it's loaded directly instead of rebuilding from raw data.

**Inference dataset cache**:
```
Data/lpbert_cityA_inferset_users{N}.pt
```

Cache keys include city, max_users, and max_seq_len to ensure correct reuse. When parameters change, a new cache is built.

### Distributed Training

LP-BERT supports multi-GPU training via PyTorch's `DataDistributed`:

```python
# train.py uses:
model = build_lpbert(model_size).to(device)
try:
    model = torch.compile(model)  # JIT compile if available
except Exception:
    pass
```

**torch.compile** (PyTorch 2.0+): JIT-compiles the model for 10–30% speedup with minimal code changes. Gracefully falls back if unavailable.

**FP16 mixed precision**:
```python
scaler = torch.amp.GradScaler(device_type="cuda")
with torch.autocast(device_type="cuda", dtype=torch.float16):
    _, loss = model(...)
scaler.scale(loss).backward()
```

Reduces memory usage (~50%) and trains faster without sacrificing accuracy.

### Inference Optimization

**Batch inference**: Users are processed in batches to maximize GPU utilization:

```python
for start in range(0, n_users, batch_size):
    end = min(start + batch_size, n_users)
    items = [dataset[i] for i in range(start, end)]
    rows = predict_batch(model, items, uids, device, beta=beta, topk=topk)
```

Larger `batch_size` → faster but more memory. Default 32 balances speed and memory.

**Inference mode**: `torch.inference_mode()` disables autograd for faster computation.

```python
with torch.inference_mode():
    logits, _ = model(...)  # No gradients computed
```

---

## File Structure

```
lpbert/
├── __init__.py                 # Package metadata
├── model.py                    # LPBert architecture, configs
├── data.py                     # Datasets, collate_fn, tokenization
├── train.py                    # Training loop, learning rate schedule
└── generate.py                 # Inference, β-penalty, prediction output

run_lpbert.py                   # CLI entry point (train, generate, evaluate)
LPBERT_geoblue.py              # Utility script for printing GEO-BLEU statistics
```

### Key Classes and Functions

**lpbert/model.py**:
- `LPBertConfig`: Dataclass for model hyperparameters (d_model, n_layers, etc.).
- `LPBert`: Main encoder model. Forward signature: `(locs, days, times, timedeltas, key_padding_mask, labels) → (logits, loss)`.
- `build_lpbert(size)`: Factory to instantiate a model from a size string.

**lpbert/data.py**:
- `build_user_sequence(user_df)`: Converts a user's DataFrame to flat arrays (days, times, locs, timedeltas).
- `LPBertDataset`: Training dataset; applies random masking at __getitem__ time.
- `LPBertInferenceDataset`: Inference dataset; combines history + masked test period.
- `collate_fn(batch)`: Pads variable-length sequences within a batch.

**lpbert/train.py**:
- `train(city, data_dir, ...) → str`: Main training function. Returns path to best checkpoint.
- `run_epoch(model, loader, optimizer, ...) → float`: Runs one epoch, returns average loss.
- `build_scheduler(optimizer, warmup_steps, total_steps)`: Creates LR schedule (warmup + cosine decay).

**lpbert/generate.py**:
- `load_checkpoint(ckpt_path, device) → (LPBert, dict)`: Loads a checkpoint, handling torch.compile keys.
- `apply_beta_penalty(logits, pred_days, pred_times, beta, topk) → np.ndarray`: Applies β-penalty, top-k filtering, returns sampled location IDs.
- `predict_batch(model, items, uids, device, beta, topk) → List[dict]`: Single batch inference; returns rows for CSV output.
- `generate(checkpoint, city, ...) → str`: Main generation function. Returns path to output CSV.

---

## Advanced Topics

### Tuning Masking Span (α)

The masking span α controls how many consecutive days to mask during training. Paper default: α=15.

**Effect of varying α**:
- **Smaller α** (e.g., 5): Model sees more training examples (less data per mask), faster epochs.
- **Larger α** (e.g., 30): Model learns to predict longer-term patterns, harder optimization.

```bash
# Experiment with α
python run_lpbert.py train --city A --alpha 10 --epochs 100
python run_lpbert.py train --city A --alpha 20 --epochs 100
```

### Adjusting β Penalty

The β-penalty discourages same-day repetition. Paper default: β=0.9.

**Effect of varying β**:
- **β=1.0**: No penalty; model can predict any location (may repeat).
- **β=0.5**: Strong penalty; heavy discouragement of repetition.
- **β=0.9**: Mild penalty; realistic diversity (paper default).

```bash
# Experiment with β during generation
python run_lpbert.py generate --checkpoint checkpoints/lpbert_cityA_paper_best.pt --beta 0.8
python run_lpbert.py generate --checkpoint checkpoints/lpbert_cityA_paper_best.pt --beta 0.95
```

### Top-K Filtering

Top-k filters out low-probability locations before sampling, stabilizing inference.

**Effect of varying topk**:
- **topk=0 (greedy)**: Always sample top-1 location; deterministic.
- **topk=3**: Only top-3 locations can be sampled; very restrictive.
- **topk=5**: Paper default; good balance.
- **topk=0 (disabled)**: All locations can be sampled; high variance.

```bash
python run_lpbert.py generate --checkpoint checkpoints/lpbert_cityA_paper_best.pt --topk 10
```

### Large-Scale Training with Subset Sampling

For datasets with 150k+ users, full-pass epochs become prohibitively slow. Subset sampling processes a random subset per epoch:

```bash
python run_lpbert.py train \
  --city A \
  --epochs 200 \
  --subset_per_epoch 15000
```

With 150k users and 15k per epoch, all users are covered in ~10 epochs. Over 200 epochs, each user is seen ~13 times, providing sufficient diversity.

---

## Troubleshooting

### Out of Memory (OOM)

**Solution 1: Reduce batch size**
```bash
python run_lpbert.py train --batch_size 8 --grad_accum 4
```

**Solution 2: Enable gradient checkpointing** (in train.py, not CLI-exposed):
Checkpointing trades compute for memory by recomputing activations during backprop.

**Solution 3: Use a smaller model**
```bash
python run_lpbert.py train --model_size small
```

### Slow Training

**Solution 1: Enable torch.compile** (automatically attempted in train.py)

**Solution 2: Use FP16 mixed precision** (automatically enabled for CUDA/MPS)

**Solution 3: Increase number of workers**
In train.py, adjust `num_workers=4` for DataLoader (or 0 for CPU/MPS).

### NaN Loss

**Possible causes**:
1. Learning rate too high: try `--lr 1e-4`.
2. Unstable gradient: check `clip_grad_norm_` in train.py is enabled.
3. Bad checkpoint: try training from scratch.

---

## References

- Terashima et al. (2023): "LP-BERT: BERT-style Bidirectional Encoder for Location Prediction". HuMob Challenge, Hamburg.
- Devlin et al. (2019): "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding". ICLR 2019.
- Vaswani et al. (2017): "Attention is All You Need". NIPS 2017.

---

## Citation

If you use LP-BERT in your research, please cite:

```bibtex
@inproceedings{terashima2023lpbert,
  title={LP-BERT: BERT-style Bidirectional Encoder for Location Prediction},
  author={Terashima et al.},
  booktitle={HuMob Challenge 2023},
  year={2023},
  address={Hamburg, Germany}
}
```

---

**Last updated:** April 2026
**Implementation:** Python 3.9+, PyTorch 2.0+
**License:** [Specify as needed]
