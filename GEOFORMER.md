# GeoFormer: GPT-Style Transformer for Human Mobility Prediction

## Overview

**GeoFormer** is a decoder-only GPT-style transformer architecture designed to predict human mobility trajectories. Given 60 days of historical GPS-based location data, it predicts the next 15 days (days 61–75) of user movements at 30-minute resolution (48 slots per day = 720 predictions per user).

The model operates over a **200×200 spatial grid** and uses a tokenization scheme where each `(x, y)` coordinate is mapped to a unique integer token. Predictions are evaluated using **GEO-BLEU**, a metric that measures both spatial accuracy and temporal sequence similarity against ground truth trajectories.

---

## High-Level Architecture

GeoFormer is a **causal language model** that processes tokenized mobility sequences:

```
Input:  [day 1 observations] → [day 2 observations] → ... → [day 60 observations]
                                      ↓
Output: Predict [day 61 observations] → [day 62 observations] → ... → [day 75 observations]
```

The architecture consists of:

1. **Input Embedding Layer**: Combines location tokens with temporal context (time-of-day, day-of-week, city)
2. **N Causal Transformer Blocks**: Stack of self-attention + feedforward layers with causal masking
3. **Language Model Head**: Linear projection to vocabulary logits (40,003 tokens)

---

## Tokenization Scheme

### Location Token Encoding

Each `(x, y)` grid coordinate is encoded as a single integer token using a row-major mapping:

```
token = (x - 1) * GRID_SIZE + (y - 1)
       = (x - 1) * 200 + (y - 1)
```

**Example**: Coordinate `(1, 1)` → token `0`, Coordinate `(200, 200)` → token `39,999`

### Token Vocabulary

| Token Type | Range | Count | Description |
|------------|-------|-------|-------------|
| Location tokens | 0–39,999 | 40,000 | Spatial coordinates on 200×200 grid |
| PAD token | 40,000 | 1 | Padding for sequence alignment |
| BOS token | 40,001 | 1 | Beginning-of-sequence marker |
| EOS token | 40,002 | 1 | End-of-sequence marker |
| **Total Vocabulary** | — | **40,003** | — |

### Inverse Mapping

To convert a token back to spatial coordinates:

```python
def token_to_xy(token: int) -> Tuple[int, int]:
    x, y = divmod(token, GRID_SIZE)  # divmod gives (quotient, remainder)
    return x + 1, y + 1             # convert back to 1-indexed
```

---

## Input Embeddings and Positional Encoding

### SpatioTemporalEmbedding

At each time step, four independent embeddings are summed together and then projected:

| Component | Embedding Dimension | Range | Purpose |
|-----------|---------------------|-------|---------|
| **Location** | `[vocab_size=40003, d_model]` | 0–40,002 | Encodes the spatial token |
| **Time-of-Day (ToD)** | `[48, d_model]` | 0–47 | 30-minute slot within a day (0=00:00, 1=00:30, ..., 47=23:30) |
| **Day-of-Week (DoW)** | `[7, d_model]` | 0–6 | Monday=0, Tuesday=1, ..., Sunday=6 |
| **City ID** | `[4, d_model]` | 0–3 | A=0, B=1, C=2, D=3 (optional, learnable) |

### Embedding Computation

For each position `i` in the sequence:

```python
embedding[i] = emb_loc(token[i]) 
              + emb_tod(tod[i]) 
              + emb_dow(dow[i]) 
              + emb_city(city_id) 
              + pos_embed(position)
```

Then layer normalization and dropout are applied:

```python
x[i] = dropout(LayerNorm(embedding[i]))
```

### Learnable Positional Embeddings

Unlike sinusoidal positional encodings in standard Transformers, GeoFormer uses **learnable positional embeddings**:
- Range: `[max_seq_len, d_model]` where `max_seq_len` depends on model config (default 256)
- Positions are continuous throughout training/inference (supports any sequence length up to `max_seq_len`)

---

## Transformer Architecture

### TransformerBlock: Pre-LayerNorm GPT Design

Each of the `n_layers` blocks follows the **pre-normalization** pattern (GPT-2 style):

```
┌─────────────────┐
│ LayerNorm(x)    │
└────────┬────────┘
         │
         ▼
┌──────────────────────────────┐
│ Multi-Head Causal Self-Attn  │
└────────┬─────────────────────┘
         │
    x = x + attn_output
         │
         ▼
┌─────────────────┐
│ LayerNorm(x)    │
└────────┬────────┘
         │
         ▼
┌──────────────────────────────┐
│ FFN(GELU) + Dropout          │
└────────┬─────────────────────┘
         │
    x = x + ffn_output
         │
         ▼
       return x
```

### Causal Self-Attention

- Uses `nn.MultiheadAttention` with `batch_first=True`
- **Causal mask**: Upper-triangular mask prevents attending to future positions
- Mask shape: `[seq_len, seq_len]` where `True` positions are ignored
- Formula: attention only considers positions `i ≤ j` (token at position `i` can attend to itself and all previous positions)

### Feedforward (FFN)

Two linear layers with GELU activation and dropout:

```python
FFN(x) = Dropout(Linear(GELU(Linear(x, d_model) * 4), d_model))
```

Expansion factor: typically 4× (e.g., if `d_model=256`, intermediate dim is `1024`)

### KV-Caching Optimization

For efficient autoregressive generation, the implementation supports **Key-Value (KV) caching**:
- On first forward pass: compute Q, K, V from full sequence
- On subsequent passes: compute Q only for new token, concatenate previous K, V
- This reduces attention computation from O(n²) to O(n) per step during inference

---

## Model Configurations

Four preset model sizes are available via `GeoFormerConfig`:

| Config | d_model | n_heads | n_layers | d_ff | max_seq_len | ~Parameters | Use Case |
|--------|---------|---------|----------|------|-------------|------------|----------|
| **tiny** | 64 | 2 | 1 | 256 | 192 | ~0.5M | Extreme smoke tests on CPU |
| **small** | 128 | 4 | 2 | 512 | 192 | ~2.5M | Development, CPU/MPS friendly |
| **medium** | 256 | 8 | 4 | 1024 | 256 | ~12M | Balanced default, GPU friendly |
| **full** | 768 | 12 | 12 | 3072 | 384 | ~150M+ | Paper config, needs A100 for city A |

All configurations support dynamic sequence lengths up to `max_seq_len`.

---

## Data Pipeline

### Step 1: Data Loading

**Input**: Raw parquet or CSV.gz files with columns: `uid`, `d`, `t`, `x`, `y`
- `uid`: User ID (int32)
- `d`: Day (int16, 1-indexed)
- `t`: Time slot (int16, 0–47)
- `x`, `y`: Grid coordinates (int16, 1–200)

**Supported Formats**:
- Parquet (fast, preferred)
- CSV.gzip (slower but widely supported)

Auto-detection: checks for `.parquet` suffix first, falls back to CSV.gz.

### Step 2: Per-User Trajectory Organization

Rows are grouped by `uid` and sorted by `(d, t)`:

```python
trajectories = {}
for uid, group in df.groupby("uid"):
    trajectories[uid] = group.sort_values(["d", "t"]).reset_index(drop=True)
```

This creates a dict mapping `uid → sorted DataFrame` for fast per-user access.

### Step 3: Sliding-Window Sample Construction

For training (days 1–60), a **strided chunking strategy** is used:

- **Context window**: 8 consecutive days (CONTEXT_DAYS)
- **Target window**: 1 day immediately after context
- **Stride**: Exactly CONTEXT_DAYS (no overlap)

For example:
```
User's 60-day history:
[Day 1–8] → [Day 9]     ← Sample 1
[Day 9–16] → [Day 17]   ← Sample 2
[Day 17–24] → [Day 25]  ← Sample 3
[Day 25–32] → [Day 33]  ← Sample 4
[Day 33–40] → [Day 41]  ← Sample 5
[Day 41–48] → [Day 49]  ← Sample 6
[Day 49–56] → [Day 57]  ← Sample 7
```

**Advantages**:
- Trains on 100% of user's data with **zero redundancy**
- Reduces dataset size drastically compared to daily sliding windows
- Maintains causality: prediction always occurs after context

### Step 4: Sequence Construction

For each window, three parallel token sequences are built:

| Sequence | Content | Example |
|----------|---------|---------|
| **tokens** | Location tokens for all observations | `[BOS, loc_tok_1, ..., loc_tok_N]` |
| **tod** | Time-of-day for each observation | `[0, 5, 12, 23, ..., 15]` |
| **dow** | Day-of-week for each observation | `[0, 0, 0, 1, ..., 2]` |

If context or target has sparse data (missing time slots), only non-zero observations are included.

### Step 5: Right-Truncation and Padding

- **Max sequence length**: `max_seq_len` (default 256 for medium config)
- If sequence exceeds max length: **keep most recent tokens** (right-truncate, preserve recency)
- During collation: **left-pad** with PAD_TOKEN to batch length

This ensures the model sees the most recent context, matching causal LM convention.

### Step 6: Labels for Language Modeling

For each sample, a label tensor is created with shape `[seq_len]`:

```
labels = [-100, -100, ..., -100, token_N+1, ..., token_N+48]
         └─────────────────┘     └──────────────────────┘
            Context (ignore)         Target (predict)
```

- **-100** positions (context): ignored by cross-entropy loss (via `ignore_index=-100`)
- **Actual tokens** (target day): must be predicted

This implements **causal LM training**: predict token `i+1` from position `i`.

### Dataset Caching

To avoid recomputation, compiled datasets are cached to disk:
```
Data/city_{city}_train_dataset_cache_seq{max_seq_len}_users{max_users}_chunked.pt
```

Subsequent runs load from cache, saving ~minutes of preprocessing.

---

## Training Pipeline

### Optimizer and Learning Rate Scheduling

**Optimizer**: AdamW with weight decay

```python
optimizer = AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
```

**Learning Rate Schedule**: Linear warmup → Cosine annealing

```
LR Curve:
    │
 1× │         ╱──────╲
    │        ╱        ╲
    │       ╱          ╲
0.01×├─────╱────────────╲───→ (minimal LR)
    │    /              \
    └────────────────────────────→ steps
    ↑warmup             total_steps
```

- Warmup steps: 10% of total training steps (default)
- Warmup start factor: `1e-4` (starts very small, increases linearly to 1.0)
- Cosine decay: from end of warmup to `eta_min=1e-6`

### Gradient Accumulation

Effective batch size = `batch_size × grad_accum_steps`

- Default: `batch_size=16, grad_accum=2` → effective batch = 32
- Gradients are scaled: `scaled_loss = loss / grad_accum`
- Accumulated over `grad_accum` steps, then optimizer step is taken

**Benefit**: Simulates larger batches without requiring more GPU memory.

### Mixed Precision Training

- Uses `torch.autocast` (float32 → float16 on GPU/MPS)
- GradScaler automatically handles underflow/overflow
- Approximately **1.5–2× speedup** with negligible accuracy loss

### Gradient Clipping

After backward pass: `nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)`

Prevents exploding gradients (important for transformers).

### Train-Validation Split

- **Split ratio**: 95% train, 5% validation (default)
- Train and validation loaders: both use `collate_fn` for left-padding batching
- Validation set: used to select best checkpoint (saved when `val_loss` is minimized)

### Training Loop

```python
for epoch in range(num_epochs):
    for batch in train_loader:
        # Forward pass
        logits, loss, _ = model(
            tokens=batch["tokens"],
            tod=batch["tod"],
            dow=batch["dow"],
            city_id=city_ids,
            key_padding_mask=batch["attn_mask"],  # masks padding
            labels=batch["labels"],
        )
        
        # Backward pass with gradient accumulation
        scaled_loss = loss / grad_accum
        scaled_loss.backward()
        
        # Optimizer step every grad_accum steps
        if (step + 1) % grad_accum == 0:
            clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
    
    # Validation after each epoch
    val_loss = run_epoch(model, val_loader, train=False)
    
    # Save best checkpoint
    if val_loss < best_val_loss:
        save_checkpoint(model, optimizer, epoch, val_loss)
```

### Checkpoint Format

Saved state_dict includes:

```python
checkpoint = {
    "epoch":      epoch_num,
    "model":      model.state_dict(),        # model weights
    "optimizer":  optimizer.state_dict(),    # optimizer state (for resume)
    "val_loss":   validation_loss,
    "train_loss": training_loss,
    "model_size": "medium",
    "city":       "A",
    "city_id":    0,
    "cfg":        model.cfg,                 # config for reconstruction
}
```

Allows **exact resumption** of training from any epoch.

### Loss Computation

Cross-entropy loss on shifted positions:

```python
# Shift: predict token[i+1] from position[i]
shift_logits = logits[:, :-1, :].contiguous()  # [B, T-1, V]
shift_labels = labels[:, 1:].contiguous()      # [B, T-1]

loss = F.cross_entropy(
    shift_logits.view(-1, vocab_size),
    shift_labels.view(-1),
    ignore_index=-100  # only optimize on target positions
)
```

---

## Knowledge Distillation

GeoFormer supports **knowledge distillation** to transfer knowledge from a large teacher model to a smaller student model.

### Distillation Setup

**Teacher**: Larger model (e.g., `medium` or `full`) trained to convergence
**Student**: Smaller model (e.g., `small` or `tiny`) being trained from scratch

### Distillation Loss

Combines standard cross-entropy with KL-divergence on softened logits:

```python
# Student and teacher both forward through same data
logits_student, loss_ce_student, _ = model_student(...)
logits_teacher, _, _ = model_teacher(...)  # with torch.no_grad()

# Soften outputs by temperature T
student_soft = log_softmax(logits_student / T)
teacher_soft = softmax(logits_teacher / T)

# KL divergence on valid positions only
loss_kd = F.kl_div(student_soft, teacher_soft, reduction='batchmean')

# Final loss: weighted combination
loss = alpha * loss_ce + (1 - alpha) * loss_kd * (T ** 2)
```

### Hyperparameters

- **alpha** (default 0.5): Weight for cross-entropy vs. KD loss
  - `alpha=1.0` → pure cross-entropy (no distillation)
  - `alpha=0.0` → pure distillation (no supervised signal)
- **temperature** (default 2.0): Softening factor for logits
  - Higher T → softer probability distributions (easier for student to mimic)
  - Multiplied by √T² to account for KL magnitude

### Training Similar to Standard Training

- Same optimizer (AdamW), scheduler, gradient accumulation
- Validation on student model only
- Best student checkpoint saved when student's validation loss is minimized

---

## Inference (Generation Pipeline)

### Context Preparation

For each user, build an **8-day context** from days 53–60 (last week of training):

```python
context_days = [53, 54, 55, 56, 57, 58, 59, 60]
ctx_tok, ctx_tod, ctx_dow = build_day_sequence(user_df, context_days)
```

If any day is missing data, only available observations are used. If all missing, use one seed location from user's history.

Context is **prepended with BOS token**:
```
[BOS] + context_tokens
```

### Autoregressive Generation

For each prediction slot in days 61–75, generate **one token at a time**:

```python
for d in range(61, 76):  # days 61–75
    for t in range(48):  # time slots 0–47
        # Pass full context (including all previously generated tokens)
        logits, _, _ = model(context, tod, dow, city_id)
        
        # Get logits for last position (the next token to predict)
        next_logits = logits[:, -1, :]  # [batch_size, vocab_size]
        
        # Apply constraints and sampling
        next_token = sample(next_logits)
        
        # Append to context for next step
        context = context + [next_token]
```

**Key point**: Each generated token becomes part of the context for the next prediction. This is **true autoregressive generation** (not feeding dummy PAD tokens).

### Constrained Decoding: History Constraint

Before sampling, mask any location the user has **never visited** in their training history:

```python
# history_counts[i] = number of times user visited location i
history_mask = (history_counts > 0)  # [vocab_size]

# Mask impossible locations to -inf (log-probability = 0)
logits[~history_mask] = float("-inf")
```

After masking, softmax is applied. Only valid locations have non-zero probability.

**Motivation**: Without this constraint, the model would spread predictions across the full 40,000-token vocabulary. Users typically visit 50–500 locations across 60 days; masking eliminates ~39,500 impossible locations, dramatically improving GEO-BLEU scores.

**Critical importance**: Empirically, this single constraint is the **largest factor** in improving prediction accuracy.

### Top-k Sampling

After history masking, apply **top-k sampling** for diversity:

```python
topk = 5
k_vals, k_indices = torch.topk(logits, k=topk)
kth_val = k_vals[:, -1].unsqueeze(-1)

# Zero out all logits below top-k threshold
logits[logits < kth_val] = -inf

# Compute probabilities
probs = softmax(logits)  # only top-5 have non-zero prob
next_token = multinomial(probs)  # sample from top-5
```

**Alternatives**:
- `topk=0`: Greedy sampling (always pick argmax)
- `topk=5`: Top-5 sampling (default)
- `temperature > 1.0`: Soften probabilities for more exploration

### Temperature Scaling

To adjust randomness:

```python
logits /= temperature  # before softmax

# temperature < 1.0: more peaked (less random)
# temperature = 1.0: original (neutral)
# temperature > 1.0: more uniform (more random)
```

### Fallback for Empty Histories

Rare edge case: if a user's history is empty (all locations masked):

```python
if (probs.sum() == 0):  # all masked
    # Uniform fallback over all locations
    probs = torch.ones(vocab_size) / vocab_size
```

### Batch Generation Optimization

To maximize GPU utilization, multiple users are generated **simultaneously**:

```python
def generate_batch(model, items, batch_size=16, topk=5, temperature=1.0):
    # items = [user_1, user_2, ..., user_16]
    # Pre-allocate GPU tensors for full generation (all 720 steps)
    tok_t = torch.full((batch_size, max_len), PAD_TOKEN)
    tod_t = torch.zeros((batch_size, max_len))
    dow_t = torch.zeros((batch_size, max_len))
    
    # Generate all 720 steps
    for step in range(720):
        # Vectorized forward pass: all 16 users at once
        logits = model(tok_t, tod_t, dow_t)  # [batch, 16, vocab]
        
        # Apply constraints and sample in parallel
        next_tokens = sample_batch(logits, histories)  # [16]
        
        # Append to all users' contexts
        tok_t[:, step] = next_tokens
        tod_t[:, step] = tod_value
        dow_t[:, step] = dow_value
```

This achieves **~2–16× speedup** vs. per-user generation.

### KV-Caching During Generation

The implementation **does not currently use KV-caching** during generation (context is passed fully each time). Future optimization could cache intermediate K, V values to reduce computation from O(n²) to O(n) per step.

### Output Format

Predictions saved as CSV: `uid, d, t, x, y`

```csv
uid,d,t,x,y
12345,61,0,45,67
12345,61,1,45,68
...
```

- One row per time slot (48 × 15 = 720 rows per user)
- Columns are int types for efficient storage

### Resumable Generation

Predictions are saved every 200 batches. If interrupted:

```python
# On re-run with same checkpoint and output file:
if prediction_csv.exists():
    processed_uids = load_already_predicted_users(prediction_csv)
    remaining_users = [u for u in all_users if u not in processed_uids]
    # Generate only remaining users, append to CSV
```

---

## Evaluation: GEO-BLEU Metric

**GEO-BLEU** measures both spatial and temporal similarity between predicted and ground-truth trajectories.

### GEO-BLEU Computation

GEO-BLEU is computed per user for days 61–75:

```python
predicted_trajectory = [(d, t, x, y), ...]  # 720 points
reference_trajectory = [(d, t, x, y), ...]  # ground truth

# Only compare time slots present in BOTH predictions and ground truth
geobleu_score = calc_geobleu_single(predicted_trajectory, reference_trajectory)
```

**Output**: Single float in [0, 1]
- Score ≈ 0: Poor predictions (far from ground truth)
- Score ≈ 1: Excellent predictions (match ground truth closely)

### GEO-BLEU Properties

- **Spatial component**: Evaluates distance between predicted and actual locations
- **Temporal component**: Evaluates sequence similarity (BLEU-like n-gram matching)
- **Symmetric**: Independent of prediction order (only (d, t) slots matter)

### Evaluation Pipeline

```python
def evaluate(predictions_csv, ground_truth_data, output_dir):
    # 1. Load predictions and ground truth
    pred_df = pd.read_csv(predictions_csv)
    gt_df = load_ground_truth(ground_truth_data, days=61-75)
    
    # 2. Group by user
    pred_grouped = {uid: grp for uid, grp in pred_df.groupby("uid")}
    gt_grouped = {uid: grp for uid, grp in gt_df.groupby("uid")}
    
    common_uids = set(pred_grouped) & set(gt_grouped)
    
    # 3. Parallel GEO-BLEU computation
    with Pool(cpu_count()) as pool:
        results = pool.map(
            _compute_geobleu_worker,
            [(uid, pred_grouped[uid], gt_grouped[uid]) for uid in common_uids]
        )
    
    # 4. Aggregate scores
    scores = {uid: score for uid, score in results if score is not None}
    
    # 5. Summary statistics
    mean   = np.mean(list(scores.values()))
    median = np.median(list(scores.values()))
    std    = np.std(list(scores.values()))
    min    = np.min(list(scores.values()))
    max    = np.max(list(scores.values()))
    
    return scores, (mean, median, std, min, max)
```

### Score Caching

Computed GEO-BLEU scores are cached to disk:

```
predictions/geobleu_scores_cache_{model_name}_city{city}.csv
```

On re-evaluation, already-computed scores are loaded from cache, only new users are scored.

### Parallel Computation

Uses Python's multiprocessing to compute GEO-BLEU in parallel:
- Default: `min(cpu_count(), 8)` processes
- Can disable with `--no_parallel` flag for debugging

### Comparison Visualization

A bar chart is generated comparing model performance to baseline:

```
┌────────────────────────────────┐
│ GEO-BLEU Comparison — City A   │
├────────────────────────────────┤
│                                │
│  ░░░            ████████████   │
│  ░░░            ████████████   │
│  ░░░            ████████████   │
│ ├────────────────────────────┤ │
│ │Global Mean (Baseline) 0.0100│ │
│ │GeoFormer (Ours)       0.0659│ │
│ └────────────────────────────┘ │
│                                │
└────────────────────────────────┘
```

---

## Usage: Complete Workflow

### 1. Training

```bash
# Quick smoke test (500 users, 2 epochs, small model)
python run_geoformer.py train --city B --max_users 500 --epochs 2 --model_size small

# Full training (City B, 5 epochs, medium model, gradient accumulation)
python run_geoformer.py train \
    --city B \
    --epochs 5 \
    --batch_size 16 \
    --grad_accum 2 \
    --model_size medium \
    --lr 3e-4
```

**Output**:
```
checkpoints/geoformer_cityB_medium_best.pt     # best checkpoint
checkpoints/geoformer_cityB_medium_latest.pt   # latest (even if not best)
checkpoints/train_log_cityB.csv                # epoch metrics (loss, time)
```

### 2. Knowledge Distillation (Optional)

```bash
# Distill medium teacher → small student
python run_geoformer.py distill \
    --teacher checkpoints/geoformer_cityB_medium_best.pt \
    --student_size small \
    --city B \
    --alpha 0.5 \
    --temp 2.0 \
    --epochs 5
```

**Output**:
```
checkpoints/geoformer_cityB_small_distilled_best.pt
```

### 3. Generation (Inference)

```bash
# Generate predictions for days 61–75
python run_geoformer.py generate \
    --checkpoint checkpoints/geoformer_cityB_medium_best.pt \
    --city B \
    --topk 5 \
    --temperature 1.0
```

**Output**:
```
predictions/predictions_cityB.csv    # 720 rows per user (uid, d, t, x, y)
```

### 4. Evaluation

```bash
# Compute GEO-BLEU scores
python run_geoformer.py evaluate \
    --predictions predictions/predictions_cityB.csv \
    --city B \
    --baseline_score 0.01
```

**Output**:
```
predictions/geobleu_scores_cityB.csv          # per-user scores
predictions/geobleu_scores_cache_GeoFormer_cityB.csv  # cached
predictions/geobleu_comparison_cityB.png      # bar chart
```

Console output:
```
==================================================
  GEO-BLEU RESULTS — GeoFormer City B
==================================================
  Users evaluated : 7,500
  Mean            : 0.073864
  Median          : 0.031503
  Std Dev         : 0.107747
  Min             : 0.000000
  Max             : 0.880176
==================================================
```

### Full Pipeline Script

```bash
#!/bin/bash
# Complete workflow for one city

CITY=B
DATA_DIR=Data
CKPT_DIR=checkpoints
PRED_DIR=predictions

# 1. Train
python run_geoformer.py train \
    --city $CITY \
    --data_dir $DATA_DIR \
    --ckpt_dir $CKPT_DIR \
    --model_size medium \
    --epochs 5

# 2. Generate
python run_geoformer.py generate \
    --checkpoint $CKPT_DIR/geoformer_city${CITY}_medium_best.pt \
    --city $CITY \
    --data_dir $DATA_DIR \
    --output_dir $PRED_DIR

# 3. Evaluate
python run_geoformer.py evaluate \
    --predictions $PRED_DIR/predictions_city${CITY}.csv \
    --city $CITY \
    --data_dir $DATA_DIR \
    --output_dir $PRED_DIR
```

---

## Output Directory Structure

```
project_root/
│
├── checkpoints/
│   ├── geoformer_cityA_medium_best.pt        # best validation loss
│   ├── geoformer_cityA_medium_latest.pt      # latest epoch
│   └── train_log_cityA.csv                   # epoch metrics
│
├── predictions/
│   ├── predictions_cityA.csv                 # generated predictions (720/user)
│   ├── geobleu_scores_cityA.csv              # per-user GEO-BLEU scores
│   ├── geobleu_scores_cache_GeoFormer_cityA.csv  # cached scores
│   └── geobleu_comparison_cityA.png          # comparison bar chart
│
├── Data/
│   ├── city_A_alldata.parquet                # input data (parquet preferred)
│   ├── city_A_alldata.csv.gz                 # input data (fallback)
│   └── city_A_train_dataset_cache_seq256_users*_chunked.pt  # cached dataset
│
└── geoformer/
    ├── __init__.py
    ├── model.py                              # GeoFormerConfig, GeoFormer, TransformerBlock
    ├── data.py                               # tokenization, datasets
    ├── train.py                              # training & distillation logic
    ├── generate.py                           # inference pipeline
    └── evaluate.py                           # GEO-BLEU evaluation
```

---

## Key Design Decisions and Rationale

| Decision | Why |
|----------|-----|
| **Causal masking** | Ensures the model can only attend to past and current positions, matching the autoregressive generation setup |
| **Right-truncation for long sequences** | Recent context is more predictive than distant history; preserves recency in fixed-length buffers |
| **Left-padding during collation** | Aligns with causal LM convention; padding masks ensure ignored gradients on PAD tokens |
| **Strided chunking (no sliding window)** | Eliminates massive redundancy; trains on 100% of data without overlap |
| **History constraint at inference** | Prevents model from predicting impossible locations; single largest factor in GEO-BLEU improvement |
| **True autoregressive generation** | Each prediction conditions on previous *real* predictions, not dummy PAD tokens |
| **Learnable positional embeddings** | More flexible than sinusoidal; supports variable-length sequences naturally |
| **KV-caching support** | Enables efficient incremental generation without recomputing full attention |
| **Mixed precision (float16)** | ~1.5–2× speedup with negligible accuracy loss on GPU/MPS |
| **Gradient accumulation** | Simulates larger effective batch sizes within memory constraints |
| **Score caching during evaluation** | Avoid redundant GEO-BLEU computation on rerun; parallel compute saves time |
| **Dataset caching** | Pre-compiled datasets are saved to disk; reduces training startup time from ~minutes to ~seconds |
| **Knowledge distillation support** | Enables deployment on edge devices without sacrificing too much accuracy |

---

## Performance Results

Based on evaluation across four cities (A, B, C, D) with the medium model size:

### GEO-BLEU Scores

| City | Users Evaluated | Mean | Median | Std Dev | Min | Max |
|------|-----------------|------|--------|---------|-----|-----|
| **A** | 37,500 | 0.0630 | 0.0216 | 0.0989 | 0.0 | 1.0 |
| **B** | 7,500 | 0.0739 | 0.0315 | 0.1077 | 0.0 | 0.8802 |
| **C** | 6,250 | 0.0653 | 0.0205 | 0.1062 | 0.0 | 0.9882 |
| **D** | 5,000 | 0.0741 | 0.0299 | 0.1104 | 0.0 | 1.0 |

### Comparison to Baseline

Global mean baseline: **0.0100** (predicting median location for each user)

| City | GeoFormer Mean | vs. Baseline | Improvement |
|------|----------------|-------------|------------|
| **A** | 0.0630 | 6.3× | +530% |
| **B** | 0.0739 | 7.4× | +639% |
| **C** | 0.0653 | 6.5× | +553% |
| **D** | 0.0741 | 7.4× | +641% |

---

## Technical Implementation Details

### Device Support

- **CUDA**: Automatically detected if available
- **MPS (Apple Metal)**: Supported on M-series Macs
- **CPU**: Falls back if GPU unavailable (slower)

### Memory Optimization

1. **Mixed precision**: ~50% less memory with float16 embeddings/activations
2. **Gradient accumulation**: Reduces effective batch requirement
3. **Sequence truncation**: Limits max_seq_len to fit in GPU memory
4. **Batch size tuning**: Adjust to available GPU VRAM

### Numerical Stability

- **Gradient clipping**: Prevents exploding gradients (max_norm=1.0)
- **LayerNorm**: Stabilizes activations between layers
- **Softmax numerical tricks**: `torch.nan_to_num` for fallback sampling

### Code Structure

```
geoformer/
  model.py      — 320 lines — GeoFormer class + config presets
  data.py       — 350 lines — tokenization, datasets, data loading
  train.py      — 590 lines — training loop, distillation, optimizer setup
  generate.py   — 310 lines — inference, batched generation, constraints
  evaluate.py   — 320 lines — GEO-BLEU computation, caching, visualization
  
Total: ~1900 lines (excluding comments/docstrings)
```

---

## Debugging and Common Issues

### Issue: OOM (Out of Memory)

**Solution**:
- Reduce `batch_size` (e.g., 16 → 8)
- Increase `grad_accum` (e.g., 2 → 4) to maintain effective batch
- Use smaller model (`small` instead of `medium`)
- Set `max_seq_len` lower in config

### Issue: Predictions Are All the Same Location

**Likely cause**: History constraint is too restrictive or empty

```python
# Debug: print history_counts sum
print(f"User {uid} visited {history_counts.sum()} locations")
if history_counts.sum() == 0:
    print("  → Empty history, using fallback")
```

### Issue: NaN Losses During Training

**Likely cause**: Learning rate too high or gradient explosion

**Solution**:
- Lower learning rate (e.g., 3e-4 → 1e-4)
- Check data for invalid values (NaN, inf in x, y)
- Ensure labels don't contain invalid indices > vocab_size

### Issue: Slow Generation

**Optimization**:
- Increase `batch_size` (e.g., 1 → 16) to utilize GPU better
- Use smaller model for inference
- Enable KV-caching (future optimization)

---

## Future Improvements

1. **KV-Caching During Generation**: Currently computes full attention each step. KV-caching would reduce from O(n²) to O(n).
2. **Mixture-of-Experts (MoE)**: Different experts for commuting vs. leisure patterns (partially explored in `moe_framework/`)
3. **Multi-City Transfer Learning**: Pre-train on all cities, fine-tune per-city
4. **Uncertainty Quantification**: Predict confidence intervals alongside points
5. **Spatial Attention Bias**: Incorporate distance decay (nearby locations should be more probable)
6. **Real-Time Streaming**: Support continuous prediction as new data arrives

---

## Citation and Related Work

GeoFormer follows the architecture of GPT-2 and adapts it for mobility prediction. The constrained decoding strategy is inspired by:
- **Transformer Language Models** (Vaswani et al., 2017; Radford et al., 2019)
- **Knowledge Distillation** (Hinton et al., 2015)
- **GEO-BLEU metric** (See project references)

---

## References

- **run_geoformer.py**: CLI entry point with subcommands (train, distill, generate, evaluate)
- **geoformer/model.py**: Core transformer architecture
- **geoformer/data.py**: Data loading and tokenization
- **geoformer/train.py**: Training loops and optimization
- **geoformer/generate.py**: Inference and constrained decoding
- **geoformer/evaluate.py**: GEO-BLEU evaluation and visualization
