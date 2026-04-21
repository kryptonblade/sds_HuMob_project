"""
lpbert/data.py
==============
Data loading and Dataset classes for LP-BERT.

Key differences from GeoFormer:
  - Sequence: flat list of observed records (day, time, loc, timedelta)
  - Timedelta: 30-min slot gap from the previous record, capped at 720 (15 days)
  - Training: randomly mask α=15 consecutive days of location IDs (BERT-style MLM)
  - Inference: history (days 1–60) + test period (days 61–75) with locs masked
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from geoformer.data import (
    xy_to_token, token_to_xy,
    VOCAB_LOC_SIZE, SLOTS_PER_DAY,
    TRAIN_DAYS, TEST_DAYS,
    load_data, build_user_trajectories,
)

# ─── LP-BERT vocabulary extension ─────────────────────────────────────────────
# Location tokens 0..39999 are shared with GeoFormer.
# We add one new special token for masked positions.
MASK_TOKEN = VOCAB_LOC_SIZE          # 40000  ← [MASK] replaces location ID
TOTAL_LOC_VOCAB = VOCAB_LOC_SIZE + 1 # 40001  ← model's output head size

# ─── Feature dimensions ───────────────────────────────────────────────────────
MAX_DATE   = 76             # days 0..75 (paper uses 0-indexed, data uses 1-indexed)
TIMEDELTA_MAX     = 720     # 15 days × 48 slots — cap on timedelta
TIMEDELTA_BUCKETS = TIMEDELTA_MAX + 1  # 721 embedding rows (0..720)

ALPHA = 15                  # consecutive days to mask during training


# ─────────────────────────────────────────────────────────────────────────────
# Sequence builder
# ─────────────────────────────────────────────────────────────────────────────
def build_user_sequence(user_df: pd.DataFrame) -> Dict[str, np.ndarray]:
    """
    Convert a sorted user DataFrame into flat arrays: days, times, locs, timedeltas.
    Timedelta[0] = 0; timedelta[i] = (day*48 + time) gap from record i-1, capped at 720.
    """
    user_df = user_df.sort_values(["d", "t"])

    days  = user_df["d"].values.astype(np.int16)
    times = user_df["t"].values.astype(np.int16)
    locs  = np.array([xy_to_token(r.x, r.y) for r in user_df.itertuples()], dtype=np.int32)

    abs_slots        = days.astype(np.int32) * SLOTS_PER_DAY + times.astype(np.int32)
    deltas           = np.zeros(len(abs_slots), dtype=np.int32)
    deltas[1:]       = abs_slots[1:] - abs_slots[:-1]
    deltas           = np.clip(deltas, 0, TIMEDELTA_MAX).astype(np.int16)

    return {"days": days, "times": times, "locs": locs, "timedeltas": deltas}


# ─────────────────────────────────────────────────────────────────────────────
# Training dataset
# ─────────────────────────────────────────────────────────────────────────────
class LPBertDataset(Dataset):
    """
    Training dataset for LP-BERT.

    For each user: builds the full record sequence over days 1–60.
    At __getitem__ time, randomly selects α=15 consecutive days and replaces
    their location IDs with MASK_TOKEN (labels = original IDs, elsewhere -100).

    Different random masks on each epoch give natural data augmentation.
    """

    def __init__(
        self,
        trajectories: Dict[int, pd.DataFrame],
        day_start:   int  = TRAIN_DAYS[0],
        day_end:     int  = TRAIN_DAYS[1],
        alpha:       int  = ALPHA,
        max_seq_len: int  = 2048,
    ):
        self.alpha       = alpha
        self.max_seq_len = max_seq_len
        self.sequences: List[Dict[str, np.ndarray]] = []

        print(f"[lpbert/data] Building training sequences (days {day_start}–{day_end}) ...")
        for uid, user_df in trajectories.items():
            sub = user_df[(user_df["d"] >= day_start) & (user_df["d"] <= day_end)]
            if len(sub) < 10:
                continue
            seq = build_user_sequence(sub)
            # Keep most-recent max_seq_len records
            if len(seq["locs"]) > max_seq_len:
                for k in seq:
                    seq[k] = seq[k][-max_seq_len:]
            self.sequences.append(seq)

        print(f"[lpbert/data] {len(self.sequences):,} training sequences")

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int):
        seq   = self.sequences[idx]
        days  = seq["days"]
        times = seq["times"]
        locs  = seq["locs"].copy()
        deltas = seq["timedeltas"]

        unique_days = np.unique(days)
        n_unique    = len(unique_days)

        # Randomly pick α consecutive days to mask
        if n_unique <= self.alpha:
            mask_days = set(unique_days.tolist())
        else:
            start_idx = np.random.randint(0, n_unique - self.alpha + 1)
            mask_days = set(unique_days[start_idx : start_idx + self.alpha].tolist())

        labels    = np.full(len(locs), -100, dtype=np.int64)
        mask_flag = np.isin(days, list(mask_days))
        labels[mask_flag] = locs[mask_flag]
        locs[mask_flag]   = MASK_TOKEN

        return {
            "locs":      torch.tensor(locs,   dtype=torch.long),
            "days":      torch.tensor(days.astype(np.int32),   dtype=torch.long),
            "times":     torch.tensor(times.astype(np.int32),  dtype=torch.long),
            "timedeltas":torch.tensor(deltas.astype(np.int32), dtype=torch.long),
            "labels":    torch.tensor(labels, dtype=torch.long),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Inference dataset
# ─────────────────────────────────────────────────────────────────────────────
class LPBertInferenceDataset(Dataset):
    """
    Inference dataset: context (days 1–60) + masked test period (days 61–75).

    Stores per-user: full combined sequence with test locs replaced by MASK_TOKEN,
    plus metadata (pred_days, pred_times) to reconstruct output CSV rows.
    """

    def __init__(
        self,
        trajectories:  Dict[int, pd.DataFrame],
        max_seq_len:   int = 2048,
    ):
        self.items: List[dict] = []
        self.uids:  List[int]  = []

        print("[lpbert/data] Building inference sequences ...")
        for uid, user_df in trajectories.items():
            train_df = user_df[
                (user_df["d"] >= TRAIN_DAYS[0]) & (user_df["d"] <= TRAIN_DAYS[1])
            ]
            pred_df  = user_df[
                (user_df["d"] >= TEST_DAYS[0])  & (user_df["d"] <= TEST_DAYS[1])
            ]

            if len(train_df) == 0 or len(pred_df) == 0:
                continue

            train_seq = build_user_sequence(train_df)
            pred_seq  = build_user_sequence(pred_df)

            # Combine: train records + pred records (locations masked)
            comb_days   = np.concatenate([train_seq["days"],  pred_seq["days"]])
            comb_times  = np.concatenate([train_seq["times"], pred_seq["times"]])
            comb_locs   = np.concatenate([
                train_seq["locs"],
                np.full(len(pred_seq["locs"]), MASK_TOKEN, dtype=np.int32),
            ])

            # Recompute timedeltas over the combined sequence
            abs_slots   = comb_days.astype(np.int32) * SLOTS_PER_DAY + comb_times.astype(np.int32)
            comb_deltas = np.zeros(len(abs_slots), dtype=np.int32)
            comb_deltas[1:] = abs_slots[1:] - abs_slots[:-1]
            comb_deltas = np.clip(comb_deltas, 0, TIMEDELTA_MAX).astype(np.int16)

            n_pred     = len(pred_seq["locs"])
            p_days     = pred_seq["days"]
            p_times    = pred_seq["times"]
            p_locs     = pred_seq["locs"]

            # Truncate combined sequence; if test period > max_seq_len,
            # trim the oldest pred records to match what's in the sequence.
            if len(comb_locs) > max_seq_len:
                keep_start  = len(comb_locs) - max_seq_len
                train_len   = len(train_seq["locs"])
                pred_cutoff = max(0, keep_start - train_len)
                comb_days   = comb_days[keep_start:]
                comb_times  = comb_times[keep_start:]
                comb_locs   = comb_locs[keep_start:]
                comb_deltas = comb_deltas[keep_start:]
                if pred_cutoff > 0:
                    p_days  = p_days[pred_cutoff:]
                    p_times = p_times[pred_cutoff:]
                    p_locs  = p_locs[pred_cutoff:]
                    n_pred  = len(p_days)

            self.items.append({
                "days":       comb_days,
                "times":      comb_times,
                "locs":       comb_locs,
                "timedeltas": comb_deltas,
                "n_pred":     n_pred,
                "pred_days":  p_days,
                "pred_times": p_times,
                "pred_locs":  p_locs,   # ground-truth (for evaluation)
            })
            self.uids.append(int(uid))

        print(f"[lpbert/data] {len(self.items):,} users prepared for inference")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        return self.items[idx]   # raw dict; generate.py handles it directly


# ─────────────────────────────────────────────────────────────────────────────
# collate_fn — right-pad variable-length sequences
# ─────────────────────────────────────────────────────────────────────────────
def collate_fn(batch):
    """
    Pad sequences to the same length within a batch (right-padding).
    key_padding_mask: True = position is padding (ignored by TransformerEncoder).
    """
    max_len = max(b["locs"].size(0) for b in batch)
    B = len(batch)

    locs_out   = torch.full((B, max_len), MASK_TOKEN, dtype=torch.long)
    days_out   = torch.zeros((B, max_len), dtype=torch.long)
    times_out  = torch.zeros((B, max_len), dtype=torch.long)
    deltas_out = torch.zeros((B, max_len), dtype=torch.long)
    labels_out = torch.full((B, max_len), -100, dtype=torch.long)
    kpm        = torch.ones((B, max_len), dtype=torch.bool)   # True = ignore

    for i, b in enumerate(batch):
        L = b["locs"].size(0)
        locs_out[i,   :L] = b["locs"]
        days_out[i,   :L] = b["days"]
        times_out[i,  :L] = b["times"]
        deltas_out[i, :L] = b["timedeltas"]
        if "labels" in b:
            labels_out[i, :L] = b["labels"]
        kpm[i, :L] = False   # valid positions

    out = {
        "locs":             locs_out,
        "days":             days_out,
        "times":            times_out,
        "timedeltas":       deltas_out,
        "key_padding_mask": kpm,
    }
    if any("labels" in b for b in batch):
        out["labels"] = labels_out
    return out
