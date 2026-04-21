"""
lpbert/generate.py
==================
Parallel inference for LP-BERT.

Key difference from GeoFormer (autoregressive):
  All masked positions are predicted in ONE forward pass per user-batch.

β-penalty (Terashima et al. §4):
  After the parallel prediction, for each user we do a sequential scan over
  the predicted day-slots. If the top-1 predicted location was already used
  on the same day, we multiply that location's logit by β (=0.9) and resample.
  This reduces the tendency to repeat the same location within a day.

Output CSV: uid, d, t, x, y — same format as GeoFormer.
"""

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

from geoformer.data import (
    token_to_xy, VOCAB_LOC_SIZE,
    load_data, build_user_trajectories,
    TEST_DAYS, TRAIN_DAYS,
)
from geoformer.train import get_device, city_data_path

from lpbert.data import (
    LPBertInferenceDataset,
    MASK_TOKEN, TOTAL_LOC_VOCAB,
)
from lpbert.model import LPBert, LPBertConfig


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint loading
# ─────────────────────────────────────────────────────────────────────────────
def load_checkpoint(ckpt_path: str, device: torch.device) -> Tuple[LPBert, dict]:
    print(f"[generate] Loading checkpoint: {ckpt_path}")
    try:
        import torch.serialization
        torch.serialization.add_safe_globals([LPBertConfig])
        ckpt = torch.load(ckpt_path, map_location=device)
    except Exception:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    cfg   = ckpt.get("cfg", LPBertConfig.paper())
    model = LPBert(cfg).to(device)

    # torch.compile saves keys as "_orig_mod.<name>" — strip that prefix
    state = ckpt["model"]
    if any(k.startswith("_orig_mod.") for k in state):
        state = {k.replace("_orig_mod.", "", 1): v for k, v in state.items()}
    model.load_state_dict(state)
    model.eval()

    print(f"[generate] Checkpoint epoch={ckpt.get('epoch','?')}  "
          f"val_loss={ckpt.get('val_loss', float('nan')):.4f}")
    return model, ckpt


# ─────────────────────────────────────────────────────────────────────────────
# β-penalty: reduce repeated same-day location predictions
# ─────────────────────────────────────────────────────────────────────────────
def apply_beta_penalty(
    logits:    np.ndarray,   # [n_masked, TOTAL_LOC_VOCAB]
    pred_days: np.ndarray,   # [n_masked] day index per masked position
    pred_times: np.ndarray,  # [n_masked] time slot per masked position
    beta:      float = 0.9,
    topk:      int   = 5,
) -> np.ndarray:
    """
    Returns sampled location token IDs [n_masked].

    Algorithm:
      For each masked position in chronological order:
        1. Apply top-k filter.
        2. If the top-1 prediction was already predicted for the same day,
           multiply its logit by β (log-space: add log β).
        3. Sample from softmax over surviving logits.
    """
    n = len(pred_days)
    preds      = np.zeros(n, dtype=np.int32)
    day_used   = {}   # day → set of predicted location IDs

    # Sort positions by (day, time) — they should already be sorted, but be safe
    order = np.lexsort((pred_times, pred_days))

    for rank, idx in enumerate(order):
        log_probs = logits[idx].copy()

        # Top-k filter (zero out non-top-k)
        if topk > 0:
            threshold = np.partition(log_probs, -topk)[-topk]
            log_probs[log_probs < threshold] = -1e9

        # β penalty: if top-1 already used on same day, scale it down
        if beta < 1.0:
            d = int(pred_days[idx])
            top1 = int(np.argmax(log_probs))
            if d in day_used and top1 in day_used[d]:
                log_probs[top1] += np.log(beta)   # log-space multiply

        # Softmax + sample
        log_probs -= log_probs.max()
        probs = np.exp(log_probs)
        probs_sum = probs.sum()
        if probs_sum <= 0:
            probs = np.ones(VOCAB_LOC_SIZE) / VOCAB_LOC_SIZE
            probs = np.concatenate([probs, [0.0]])   # mask token gets 0
        else:
            probs /= probs_sum

        tok = int(np.random.choice(len(probs), p=probs))
        if tok >= VOCAB_LOC_SIZE:
            tok = int(np.random.randint(0, VOCAB_LOC_SIZE))
        preds[idx] = tok

        # Register prediction for this day
        d = int(pred_days[idx])
        if d not in day_used:
            day_used[d] = set()
        day_used[d].add(tok)

    return preds


# ─────────────────────────────────────────────────────────────────────────────
# Single-batch inference
# ─────────────────────────────────────────────────────────────────────────────
def predict_batch(
    model:   LPBert,
    items:   List[dict],
    uids:    List[int],
    device:  torch.device,
    beta:    float = 0.9,
    topk:    int   = 5,
) -> List[dict]:
    """
    Run one forward pass for a batch of users, return prediction rows.
    Each item has pre-built combined sequence with test locs masked.
    """
    B = len(items)

    # ── Pad batch ────────────────────────────────────────────────────────────
    max_len = max(len(it["locs"]) for it in items)

    locs_b   = torch.full((B, max_len), MASK_TOKEN, dtype=torch.long)
    days_b   = torch.zeros((B, max_len), dtype=torch.long)
    times_b  = torch.zeros((B, max_len), dtype=torch.long)
    deltas_b = torch.zeros((B, max_len), dtype=torch.long)
    kpm_b    = torch.ones((B, max_len), dtype=torch.bool)  # True=padding

    for i, it in enumerate(items):
        L = len(it["locs"])
        locs_b[i,   :L] = torch.from_numpy(it["locs"].astype(np.int64))
        days_b[i,   :L] = torch.from_numpy(it["days"].astype(np.int64))
        times_b[i,  :L] = torch.from_numpy(it["times"].astype(np.int64))
        deltas_b[i, :L] = torch.from_numpy(it["timedeltas"].astype(np.int64))
        kpm_b[i,    :L] = False

    locs_b   = locs_b.to(device)
    days_b   = days_b.to(device)
    times_b  = times_b.to(device)
    deltas_b = deltas_b.to(device)
    kpm_b    = kpm_b.to(device)

    # ── Forward pass ─────────────────────────────────────────────────────────
    with torch.inference_mode():
        with torch.autocast(device_type=device.type, dtype=torch.float16):
            logits, _ = model(locs_b, days_b, times_b, deltas_b,
                              key_padding_mask=kpm_b)

    logits = logits.float().cpu().numpy()   # [B, max_len, TOTAL_LOC_VOCAB]

    # ── Extract masked positions and apply β-penalty ──────────────────────
    rows = []
    for i, (it, uid) in enumerate(zip(items, uids)):
        L      = len(it["locs"])
        n_pred = it["n_pred"]
        # If test period > max_seq_len, the sequence was truncated from the front
        # and only the last n_in_seq pred records are actually present.
        n_in_seq   = min(n_pred, L)
        mask_start = L - n_in_seq

        masked_logits  = logits[i, mask_start:L, :VOCAB_LOC_SIZE]
        pred_days_arr  = it["pred_days"][-n_in_seq:].astype(np.int32)
        pred_times_arr = it["pred_times"][-n_in_seq:].astype(np.int32)

        pred_locs = apply_beta_penalty(
            masked_logits, pred_days_arr, pred_times_arr,
            beta=beta, topk=topk,
        )

        for j in range(n_in_seq):
            x, y = token_to_xy(int(pred_locs[j]))
            rows.append({
                "uid": uid,
                "d":   int(pred_days_arr[j]),
                "t":   int(pred_times_arr[j]),
                "x":   x,
                "y":   y,
            })

    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Main generate function
# ─────────────────────────────────────────────────────────────────────────────
def generate(
    checkpoint:  str,
    city:        str   = "A",
    data_dir:    Path  = Path("Data"),
    output_dir:  Path  = Path("predictions"),
    beta:        float = 0.9,
    topk:        int   = 5,
    max_users:   Optional[int] = None,
    batch_size:  int   = 32,
    max_seq_len: int   = 2048,
):
    city   = city.upper()
    device = get_device()

    model, _ = load_checkpoint(checkpoint, device)

    # ── Build inference dataset (cached) ──
    cache_path = data_dir / f"lpbert_city{city}_inferset_users{max_users}.pt"

    if cache_path.exists():
        print(f"[generate] Loading cached inference dataset: {cache_path}")
        inf_ds = torch.load(cache_path, weights_only=False)
    else:
        data_path = city_data_path(data_dir, city)
        df        = load_data(data_path, max_users=max_users)
        trajs     = build_user_trajectories(df)
        inf_ds    = LPBertInferenceDataset(trajs, max_seq_len=max_seq_len)
        print(f"[generate] Caching inference dataset: {cache_path}")
        torch.save(inf_ds, cache_path)

    print(f"[generate] {len(inf_ds):,} users  |  "
          f"beta={beta}  topk={topk}  batch={batch_size}")

    all_rows: List[dict] = []
    n = len(inf_ds)

    for start in tqdm(range(0, n, batch_size), desc="Generating"):
        end   = min(start + batch_size, n)
        items = [inf_ds[i] for i in range(start, end)]
        uids  = [inf_ds.uids[i] for i in range(start, end)]
        rows  = predict_batch(model, items, uids, device, beta=beta, topk=topk)
        all_rows.extend(rows)

    # ── Save ──
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"lpbert_predictions_city{city}.csv"

    pred_df = pd.DataFrame(all_rows, columns=["uid", "d", "t", "x", "y"])
    pred_df = pred_df.astype({
        "uid": "int32", "d": "int16", "t": "int16",
        "x": "int16",  "y": "int16",
    })
    pred_df.to_csv(out_path, index=False)

    print(f"[generate] Saved {len(pred_df):,} rows → {out_path}")
    return str(out_path)
