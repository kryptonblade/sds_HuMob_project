"""
geoformer/data.py
=================
Data loading, tokenization, and PyTorch Dataset construction
for the HuMob GeoFormer model.

Tokenization scheme (following GeoFormer paper):
  - Each (x, y) grid coordinate → single integer token: x * GRID_SIZE + y
  - GRID_SIZE = 200  →  vocab space = 0..39999
  - Special tokens:  PAD=40000, BOS=40001, EOS=40002
  - Time-of-day (tod):   t in 0..47   (48 slots/day)
  - Day-of-week (dow):   (d-1) % 7    (0=Mon..6=Sun)

Sequence design (following GeoFormer paper, adapted for 2025 data):
  - Context window : 8 days  (8 × 48 = 384 time slots max)
  - Prediction window: 1 day (48 slots)
  - Training: sliding windows over days 1..60
  - Inference: use days 53..60 as context → predict days 61..75
"""

import gzip
import io
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
GRID_SIZE       = 200          # 200×200 grid
VOCAB_LOC_SIZE  = GRID_SIZE * GRID_SIZE   # 40 000 location tokens
PAD_TOKEN       = VOCAB_LOC_SIZE          # 40 000
BOS_TOKEN       = VOCAB_LOC_SIZE + 1      # 40 001
EOS_TOKEN       = VOCAB_LOC_SIZE + 2      # 40 002
TOTAL_VOCAB     = VOCAB_LOC_SIZE + 3      # 40 003

SLOTS_PER_DAY   = 48           # 30-minute intervals
DOW_COUNT       = 7
TOD_COUNT       = SLOTS_PER_DAY

CONTEXT_DAYS    = 8            # 8-day sliding window
TARGET_DAYS     = 1            # predict 1 day ahead

TRAIN_DAYS      = (1, 60)
TEST_DAYS       = (61, 75)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def xy_to_token(x: int, y: int) -> int:
    """Encode (x, y) grid coordinate as a single integer token. 1-indexed to 0-indexed."""
    return (int(x) - 1) * GRID_SIZE + (int(y) - 1)


def token_to_xy(token: int) -> Tuple[int, int]:
    """Decode a location token back to (x, y). 0-indexed back to 1-indexed."""
    x, y = divmod(token, GRID_SIZE)
    return x + 1, y + 1


def day_of_week(d: int) -> int:
    """Convert 1-indexed day to 0-indexed day-of-week (0=Mon)."""
    return (d - 1) % DOW_COUNT


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────
def load_csv_gz(path: Path, max_users: Optional[int] = None) -> pd.DataFrame:
    """
    Load a city_X_alldata.csv.gz file.
    Columns: uid, d, t, x, y
    Optionally limit to the first `max_users` unique user IDs.
    """
    print(f"[data] Loading {path.name} ...")
    with gzip.open(path, "rt", encoding="utf-8") as f:
        df = pd.read_csv(f)

    df = df.astype({"uid": "int32", "d": "int16", "t": "int16",
                    "x": "int16", "y": "int16"})

    if max_users is not None:
        uids = sorted(df["uid"].unique())[:max_users]
        df = df[df["uid"].isin(uids)].reset_index(drop=True)
        print(f"[data] Truncated to {max_users} users → {len(df):,} rows")
    else:
        print(f"[data] Loaded {len(df):,} rows, {df['uid'].nunique():,} users")

    return df


def load_parquet(path: Path, max_users: Optional[int] = None) -> pd.DataFrame:
    """Load from cached parquet file (faster than CSV.gz)."""
    print(f"[data] Loading parquet {path.name} ...")
    df = pd.read_parquet(path)
    df = df.astype({"uid": "int32", "d": "int16", "t": "int16",
                    "x": "int16", "y": "int16"})
    if max_users is not None:
        uids = sorted(df["uid"].unique())[:max_users]
        df = df[df["uid"].isin(uids)].reset_index(drop=True)
        print(f"[data] Truncated to {max_users} users → {len(df):,} rows")
    else:
        print(f"[data] Loaded {len(df):,} rows, {df['uid'].nunique():,} users")
    return df


def load_data(data_path: Path, max_users: Optional[int] = None) -> pd.DataFrame:
    """Auto-detect parquet vs csv.gz and load accordingly."""
    if data_path.suffix == ".parquet":
        return load_parquet(data_path, max_users)
    else:
        return load_csv_gz(data_path, max_users)


# ─────────────────────────────────────────────────────────────────────────────
# Per-user trajectory building
# ─────────────────────────────────────────────────────────────────────────────
def build_user_trajectories(df: pd.DataFrame) -> Dict[int, pd.DataFrame]:
    """
    Group rows by user, sort by (d, t).
    Returns dict: uid → sorted DataFrame
    """
    print("[data] Building per-user trajectories ...")
    trajectories = {}
    for uid, group in df.groupby("uid", sort=False):
        trajectories[int(uid)] = group.sort_values(["d", "t"]).reset_index(drop=True)
    print(f"[data] {len(trajectories):,} users indexed")
    return trajectories


def user_history_token_counts(user_df: pd.DataFrame) -> np.ndarray:
    """
    Count how many times each location token appears in a user's training history.
    Returns array of shape [VOCAB_LOC_SIZE] with counts.
    """
    counts = np.zeros(VOCAB_LOC_SIZE, dtype=np.int32)
    for _, row in user_df.iterrows():
        tok = xy_to_token(row["x"], row["y"])
        counts[tok] += 1
    return counts


# ─────────────────────────────────────────────────────────────────────────────
# Sequence construction
# ─────────────────────────────────────────────────────────────────────────────
def build_day_sequence(user_df: pd.DataFrame,
                       days: List[int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build token/tod/dow arrays for a specific list of days, in order.
    Only includes time slots that appear in the data (sparse trajectories).

    Returns:
        tokens  shape [N]
        tod     shape [N]  (0..47)
        dow     shape [N]  (0..6)
    """
    rows = user_df[user_df["d"].isin(days)].sort_values(["d", "t"])
    if len(rows) == 0:
        return np.array([], dtype=np.int32), np.array([], dtype=np.int32), np.array([], dtype=np.int32)

    tokens = np.array([xy_to_token(r.x, r.y) for r in rows.itertuples()], dtype=np.int32)
    tod    = rows["t"].values.astype(np.int32)
    dow    = np.array([day_of_week(d) for d in rows["d"].values], dtype=np.int32)

    return tokens, tod, dow


# ─────────────────────────────────────────────────────────────────────────────
# PyTorch Dataset for training (sliding 8-day → 1-day windows)
# ─────────────────────────────────────────────────────────────────────────────
class MobilityDataset(Dataset):
    """
    PyTorch Dataset for GeoFormer training.

    For each user, produces sliding-window samples:
      - context: days [d, d+1, ..., d+CONTEXT_DAYS-1]  (8 days)
      - target:  days [d+CONTEXT_DAYS]                   (1 day)

    Each sample is a fixed-length sequence of up to max_seq_len tokens
    with causal LM labels. Padded with PAD_TOKEN on the left.

    The causal LM loss is computed only on target-day positions.
    """

    DEFAULT_MAX_SEQ_LEN = CONTEXT_DAYS * SLOTS_PER_DAY  # 384

    def __init__(self, trajectories: Dict[int, pd.DataFrame],
                 day_start: int = 1,
                 day_end: int = 60,
                 city_id: int = 0,
                 max_seq_len: int = None,
                 max_windows_per_user: int = 5):
        """
        Args:
            trajectories: dict uid → sorted user DataFrame
            day_start, day_end: inclusive day range to build windows from
            city_id: integer city identifier (0=A, 1=B, 2=C, 3=D)
        """
        self.city_id = city_id
        self.samples = []
        seq_limit = max_seq_len if max_seq_len is not None else self.DEFAULT_MAX_SEQ_LEN

        print(f"[data] Building training samples (days {day_start}–{day_end}) ...")

        for uid, user_df in trajectories.items():
            user_train = user_df[(user_df["d"] >= day_start) &
                                 (user_df["d"] <= day_end)]
            if len(user_train) == 0:
                continue

            # ── OPTIMIZATION: Contiguous Causal Chunking ──
            # Instead of a sliding window that moves by 1 day (generating massive overlapping redundancy),
            # we stride the window forward by exactly CONTEXT_DAYS.
            # This trains on 100% of the user's data with zero overlap, drastically slashing dataset size.
            
            max_context_start = day_end - CONTEXT_DAYS
            # Stride by CONTEXT_DAYS to cleanly "chunk" their 60-day history.
            valid_starts = list(range(day_start, max_context_start + 1, CONTEXT_DAYS))

            for ctx_start in valid_starts:
                ctx_days = list(range(ctx_start, ctx_start + CONTEXT_DAYS))
                tgt_day  = ctx_start + CONTEXT_DAYS

                # Build context tokens
                ctx_tok, ctx_tod, ctx_dow = build_day_sequence(user_train, ctx_days)
                # Build target tokens
                tgt_tok, tgt_tod, tgt_dow = build_day_sequence(
                    user_df[(user_df["d"] == tgt_day)], [tgt_day])

                if len(ctx_tok) == 0 or len(tgt_tok) == 0:
                    continue

                all_tok = np.concatenate([[BOS_TOKEN], ctx_tok, tgt_tok])
                all_tod = np.concatenate([[0],         ctx_tod, tgt_tod])
                all_dow = np.concatenate([[0],         ctx_dow, tgt_dow])

                # Truncate to seq_limit from the right (keep recent context)
                if len(all_tok) > seq_limit:
                    all_tok = all_tok[-seq_limit:]
                    all_tod = all_tod[-seq_limit:]
                    all_dow = all_dow[-seq_limit:]

                # Labels: -100 for context positions, actual tokens for target
                ctx_len = min(len(ctx_tok) + 1, len(all_tok) - len(tgt_tok))
                labels = np.full(len(all_tok), -100, dtype=np.int64)
                labels[ctx_len:] = all_tok[ctx_len:]

                self.samples.append({
                    "tokens": all_tok,
                    "tod":    all_tod,
                    "dow":    all_dow,
                    "labels": labels,
                })

        print(f"[data] {len(self.samples):,} training samples built")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return {
            "tokens": torch.tensor(s["tokens"], dtype=torch.long),
            "tod":    torch.tensor(s["tod"],    dtype=torch.long),
            "dow":    torch.tensor(s["dow"],    dtype=torch.long),
            "labels": torch.tensor(s["labels"], dtype=torch.long),
        }


def collate_fn(batch):
    """
    Pad sequences in a batch to the same length.
    Padding is on the LEFT (past positions), so the model sees the most
    recent tokens at the end — matching causal LM convention.
    """
    max_len = max(b["tokens"].shape[0] for b in batch)

    out_tokens = torch.full((len(batch), max_len), PAD_TOKEN, dtype=torch.long)
    out_tod    = torch.zeros((len(batch), max_len), dtype=torch.long)
    out_dow    = torch.zeros((len(batch), max_len), dtype=torch.long)
    out_labels = torch.full((len(batch), max_len), -100,      dtype=torch.long)
    attn_mask  = torch.zeros((len(batch), max_len), dtype=torch.bool)  # True = padding

    for i, b in enumerate(batch):
        L = b["tokens"].shape[0]
        out_tokens[i, -L:] = b["tokens"]
        out_tod[i,    -L:] = b["tod"]
        out_dow[i,    -L:] = b["dow"]
        out_labels[i, -L:] = b["labels"]
        attn_mask[i,  :-L] = True   # mask left-padding positions

    return {
        "tokens":      out_tokens,
        "tod":         out_tod,
        "dow":         out_dow,
        "labels":      out_labels,
        "attn_mask":   attn_mask,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Inference dataset: build 8-day context for generation
# ─────────────────────────────────────────────────────────────────────────────
class InferenceDataset(Dataset):
    """
    For each user: builds the 8-day context (days 53..60) for generation.
    Also stores the user's full training-day token counts for constrained generation.
    """
    CONTEXT_START = 53
    CONTEXT_END   = 60

    def __init__(self, trajectories: Dict[int, pd.DataFrame], city_id: int = 0):
        self.city_id = city_id
        self.items = []

        print("[data] Building inference context ...")
        ctx_days = list(range(self.CONTEXT_START, self.CONTEXT_END + 1))

        for uid, user_df in trajectories.items():
            train_df = user_df[(user_df["d"] >= TRAIN_DAYS[0]) &
                               (user_df["d"] <= TRAIN_DAYS[1])]
            if len(train_df) == 0:
                continue

            ctx_tok, ctx_tod, ctx_dow = build_day_sequence(train_df, ctx_days)
            history_counts = user_history_token_counts(train_df)

            self.items.append({
                "uid":            uid,
                "ctx_tok":        ctx_tok,
                "ctx_tod":        ctx_tod,
                "ctx_dow":        ctx_dow,
                "history_counts": history_counts,
            })

        print(f"[data] {len(self.items):,} users prepared for inference")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]   # raw dict, handled manually in generate.py
