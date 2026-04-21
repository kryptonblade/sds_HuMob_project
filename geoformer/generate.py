"""
geoformer/generate.py
=====================
Constrained autoregressive inference for GeoFormer.

Key design (from the paper):
 1. Load 8-day context (days 53–60) per user as prompt
 2. Autoregressively generate tokens for each time slot of days 61–75
 3. Constraint: if a location has never appeared in the user's history,
    its logit is set to -inf (zero probability) → prevents impossible moves
 4. Top-k sampling (k=5, temperature configurable) for diversity

Output CSV: uid, d, t, x, y — matching the ground-truth format.
"""

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

from geoformer.data import (
    InferenceDataset, load_data, build_user_trajectories,
    xy_to_token, token_to_xy,
    BOS_TOKEN, EOS_TOKEN, PAD_TOKEN,
    VOCAB_LOC_SIZE, TOTAL_VOCAB,
    SLOTS_PER_DAY, TEST_DAYS, TRAIN_DAYS,
    day_of_week,
)
from geoformer.model import GeoFormer, GeoFormerConfig
from geoformer.train import CITY_IDS, city_data_path, get_device


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────
def load_checkpoint(ckpt_path: str, device: torch.device) -> Tuple[GeoFormer, dict]:
    print(f"[generate] Loading checkpoint: {ckpt_path}")
    # Register our dataclass as safe for torch.load (PyTorch >= 2.6 requires explicit allowlisting)
    try:
        import torch.serialization
        torch.serialization.add_safe_globals([GeoFormerConfig])
        ckpt = torch.load(ckpt_path, map_location=device)
    except Exception:
        # Fallback: weights_only=False (trustworthy since we generated it ourselves)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    cfg = ckpt.get("cfg", GeoFormerConfig())
    model = GeoFormer(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    print(f"[generate] Checkpoint epoch={ckpt.get('epoch','?')}, "
          f"train_loss={ckpt.get('train_loss', float('nan')):.4f}")
    return model, ckpt


# ─────────────────────────────────────────────────────────────────────────────
# Batched Per-user generation
# ─────────────────────────────────────────────────────────────────────────────
def generate_batch(
    model:           GeoFormer,
    items:           List[dict],
    city_id:         int,
    device:          torch.device,
    test_days:       Tuple[int, int] = TEST_DAYS,
    topk:            int   = 5,
    temperature:     float = 1.0,
    max_ctx_tokens:  int   = 384,
) -> List[dict]:
    """
    Generate day-61..75 trajectory for a batch of users simultaneously.
    Returns list of dicts: {uid, d, t, x, y}
    """
    batch_size = len(items)
    uids = [item["uid"] for item in items]
    
    # [batch, vocab]
    history_counts_list = [torch.from_numpy(item["history_counts"]) for item in items]
    history_counts = torch.stack(history_counts_list).to(device)

    ctx_tok_list = []
    ctx_tod_list = []
    ctx_dow_list = []

    for item in items:
        c_tok, c_tod, c_dow = list(item["ctx_tok"]), list(item["ctx_tod"]), list(item["ctx_dow"])
        
        if not c_tok:
            if item["history_counts"].sum() > 0:
                seed = int(np.argmax(item["history_counts"]))
            else:
                seed = int(np.random.randint(0, VOCAB_LOC_SIZE))
            c_tok, c_tod, c_dow = [seed], [0], [0]

        c_tok = [BOS_TOKEN] + c_tok
        c_tod = [0] + c_tod
        c_dow = [0] + c_dow

        if len(c_tok) >= max_ctx_tokens:
            c_tok = c_tok[-max_ctx_tokens+1:]
            c_tod = c_tod[-max_ctx_tokens+1:]
            c_dow = c_dow[-max_ctx_tokens+1:]
            
        ctx_tok_list.append(c_tok)
        ctx_tod_list.append(c_tod)
        ctx_dow_list.append(c_dow)

    # ─────────────────────────────────────────────────────────
    # Massive GPU Optimization: Pre-allocate vector buffers
    # ─────────────────────────────────────────────────────────
    total_steps = (test_days[1] - test_days[0] + 1) * SLOTS_PER_DAY
    full_len = max_ctx_tokens + total_steps
    
    tok_np = np.full((batch_size, full_len), PAD_TOKEN, dtype=np.int64)
    tod_np = np.zeros((batch_size, full_len), dtype=np.int64)
    dow_np = np.zeros((batch_size, full_len), dtype=np.int64)
    key_pad_np = np.ones((batch_size, full_len), dtype=bool)

    for i in range(batch_size):
        L = len(ctx_tok_list[i])
        start_idx = max_ctx_tokens - L
        tok_np[i, start_idx:max_ctx_tokens] = ctx_tok_list[i]
        tod_np[i, start_idx:max_ctx_tokens] = ctx_tod_list[i]
        dow_np[i, start_idx:max_ctx_tokens] = ctx_dow_list[i]
        key_pad_np[i, start_idx:max_ctx_tokens] = False

    # Upload single unified tensor blocks to GPU
    tok_t_full = torch.from_numpy(tok_np).to(device)
    tod_t_full = torch.from_numpy(tod_np).to(device)
    dow_t_full = torch.from_numpy(dow_np).to(device)
    key_pad_full = torch.from_numpy(key_pad_np).to(device)
    cid_t = torch.tensor([city_id]*batch_size, dtype=torch.long, device=device)

    predictions = []
    generated_toks_list = []
    step_idx = 0
    
    inf_mask = (history_counts == 0)
    next_logits = torch.zeros((batch_size, TOTAL_VOCAB), dtype=torch.float32, device=device)
    fallback_probs = torch.zeros((batch_size, TOTAL_VOCAB), dtype=torch.float32, device=device)
    fallback_probs[:, :VOCAB_LOC_SIZE] = 1.0 / VOCAB_LOC_SIZE

    # Enable Mixed Precision and pre-allocations for ~2x scaling
    with torch.autocast('mps', dtype=torch.float16):
        for d in range(test_days[0], test_days[1] + 1):
            dow = day_of_week(d)
            for t in range(SLOTS_PER_DAY):
                
                window_tok_t = tok_t_full[:, step_idx : step_idx + max_ctx_tokens]
                window_tod_t = tod_t_full[:, step_idx : step_idx + max_ctx_tokens]
                window_dow_t = dow_t_full[:, step_idx : step_idx + max_ctx_tokens]
                window_key_pad = key_pad_full[:, step_idx : step_idx + max_ctx_tokens]
                
                with torch.inference_mode():
                    logits, _, _ = model(window_tok_t, window_tod_t, window_dow_t, city_id=cid_t, key_padding_mask=window_key_pad, return_last_logit_only=True)
                    
                    # Zero-allocation in-place copy
                    next_logits.copy_(logits[:, -1, :].float())

                    # Global constraints
                    next_logits[:, PAD_TOKEN] = float("-inf")
                    next_logits[:, BOS_TOKEN] = float("-inf")
                    next_logits[:, EOS_TOKEN] = float("-inf")
                    next_logits[:, :VOCAB_LOC_SIZE].masked_fill_(inf_mask, float("-inf"))
                    
                    if temperature != 1.0:
                        next_logits /= temperature
                        
                    # Vectorized TopK over Batch to prevent CPU blocking
                    if topk > 0:
                        calc_k = min(topk, VOCAB_LOC_SIZE)
                        k_vals, _ = torch.topk(next_logits, calc_k, dim=-1)
                        kth_val = k_vals[:, -1].unsqueeze(1)
                        next_logits.masked_fill_(next_logits < kth_val, float("-inf"))
                    
                    probs = F.softmax(next_logits, dim=-1)
                    probs = torch.nan_to_num_(probs, nan=0.0)
                    
                    empty_mask = (probs.sum(dim=-1, keepdim=True) == 0)
                    probs = torch.where(empty_mask, fallback_probs, probs)
                        
                next_toks = torch.multinomial(probs, num_samples=1).squeeze(-1)
                generated_toks_list.append(next_toks)
                
                # --- In-Place Rolling Window Slicing (Zero Copy) ---
                fill_idx = step_idx + max_ctx_tokens
                tok_t_full[:, fill_idx] = next_toks
                tod_t_full[:, fill_idx] = t
                dow_t_full[:, fill_idx] = dow
                key_pad_full[:, fill_idx] = False
                
                step_idx += 1
            
    # ── Reconstruct Predictions Sequentially ──
    all_toks = torch.stack(generated_toks_list, dim=1).cpu().numpy()
    step = 0
    for d in range(test_days[0], test_days[1] + 1):
        for t in range(SLOTS_PER_DAY):
            for i in range(batch_size):
                ntok = int(all_toks[i, step])
                x, y = token_to_xy(ntok)
                predictions.append({"uid": uids[i], "d": d, "t": t, "x": x, "y": y})
            step += 1

    return predictions


# ─────────────────────────────────────────────────────────────────────────────
# Main generation function
# ─────────────────────────────────────────────────────────────────────────────
def generate(
    checkpoint:   str,
    city:         str   = "B",
    data_dir:     Path  = Path("Data"),
    output_dir:   Path  = Path("predictions"),
    topk:         int   = 5,
    temperature:  float = 1.0,
    max_users:    int   = None,
    batch_size:   int   = 1,   # per-user generation, keeps it simple
):
    city = city.upper()
    city_id = CITY_IDS[city]
    device  = get_device()

    # ── Load model ──
    model, ckpt = load_checkpoint(checkpoint, device)

    # ── Build or Load inference dataset ──
    dataset_cache_path = data_dir / f"city_{city}_inference_dataset_cache_users{max_users}.pt"
    
    if dataset_cache_path.exists():
        print(f"[generate] Loading pre-processed inference dataset from cache: {dataset_cache_path}")
        inf_ds = torch.load(dataset_cache_path, weights_only=False)
    else:
        data_path = city_data_path(data_dir, city)
        df = load_data(data_path, max_users=max_users)
        trajectories = build_user_trajectories(df)
        inf_ds = InferenceDataset(trajectories, city_id=city_id)
        print(f"[generate] Saving inference dataset to cache: {dataset_cache_path}")
        torch.save(inf_ds, dataset_cache_path)
    from torch.utils.data import DataLoader
    
    # Simple custom collate to just return list of dicts directly
    def list_collate(batch): return batch
    
    loader = DataLoader(inf_ds, batch_size=batch_size, shuffle=False,
                        collate_fn=list_collate, num_workers=0)

    # ── Generate ──
    all_preds = []
    print(f"[generate] Generating predictions for {len(inf_ds):,} users in batches of {batch_size} ...")
    print(f"           top-k={topk}, temperature={temperature}")

    for batch_items in tqdm(loader, desc="Generating"):
        user_preds = generate_batch(
            model, batch_items, city_id, device,
            topk=topk, temperature=temperature,
            max_ctx_tokens=48,  # Slashed completely down to 1 day context (extremely fast local attention matrix computation)
        )
        all_preds.extend(user_preds)

    # ── Save ──
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"predictions_city{city}.csv"

    pred_df = pd.DataFrame(all_preds, columns=["uid", "d", "t", "x", "y"])
    pred_df = pred_df.astype({"uid": "int32", "d": "int16", "t": "int16",
                               "x": "int16", "y": "int16"})
    pred_df.to_csv(out_path, index=False)

    print(f"[generate] Saved {len(pred_df):,} rows → {out_path}")
    return str(out_path)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Generate GeoFormer predictions")
    p.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint")
    p.add_argument("--city",       default="B",   choices=list(CITY_IDS))
    p.add_argument("--data_dir",   default="Data")
    p.add_argument("--output_dir", default="predictions")
    p.add_argument("--topk",       type=int,   default=5)
    p.add_argument("--temperature",type=float, default=1.0)
    p.add_argument("--max_users",  type=int,   default=None)
    p.add_argument("--batch_size", type=int,   default=16, help="Users generated concurrently")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    generate(
        checkpoint=args.checkpoint,
        city=args.city,
        data_dir=Path(args.data_dir),
        output_dir=Path(args.output_dir),
        topk=args.topk,
        temperature=args.temperature,
        max_users=args.max_users,
        batch_size=args.batch_size,
    )
