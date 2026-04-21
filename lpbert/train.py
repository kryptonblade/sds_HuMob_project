"""
lpbert/train.py
===============
Training loop for LP-BERT on HuMob data.

Training objective: BERT-style masked location modelling (MLM).
  - For each user, α=15 consecutive days of location IDs are randomly masked.
  - The model predicts all masked locations in parallel (bidirectional attention).
  - New random masks on every epoch → natural data augmentation.

Usage (via run_lpbert.py):
    python run_lpbert.py train --city A --epochs 200 --model_size paper
"""

import math
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, RandomSampler, random_split
from tqdm import tqdm

from geoformer.data import load_data, build_user_trajectories, TRAIN_DAYS
from geoformer.train import get_device, city_data_path, CITY_IDS

from lpbert.data import LPBertDataset, collate_fn
from lpbert.model import LPBert, LPBertConfig, build_lpbert


# ─────────────────────────────────────────────────────────────────────────────
# LR schedule: linear warmup → cosine decay  (same as GeoFormer)
# ─────────────────────────────────────────────────────────────────────────────
def build_scheduler(optimizer, warmup_steps: int, total_steps: int):
    warmup = LinearLR(optimizer, start_factor=1e-4, end_factor=1.0,
                      total_iters=warmup_steps)
    cosine = CosineAnnealingLR(optimizer,
                                T_max=max(total_steps - warmup_steps, 1),
                                eta_min=1e-6)
    return SequentialLR(optimizer, [warmup, cosine], milestones=[warmup_steps])


# ─────────────────────────────────────────────────────────────────────────────
# Single epoch
# ─────────────────────────────────────────────────────────────────────────────
def run_epoch(
    model:      LPBert,
    loader:     DataLoader,
    optimizer,
    scheduler,
    device:     torch.device,
    train:      bool = True,
    grad_accum: int  = 1,
    scaler      = None,
) -> float:
    model.train(train)
    total_loss  = 0.0
    total_steps = 0

    if train:
        optimizer.zero_grad()

    pbar = tqdm(loader, desc="Train" if train else "Val", leave=False)

    for step, batch in enumerate(pbar):
        locs   = batch["locs"].to(device)
        days   = batch["days"].to(device)
        times  = batch["times"].to(device)
        deltas = batch["timedeltas"].to(device)
        labels = batch["labels"].to(device)
        kpm    = batch["key_padding_mask"].to(device)

        with torch.set_grad_enabled(train):
            with torch.autocast(device_type=device.type, dtype=torch.float16):
                _, loss = model(locs, days, times, deltas,
                                key_padding_mask=kpm, labels=labels)

        if loss is None or torch.isnan(loss):
            if train:
                optimizer.zero_grad()
            continue

        if train:
            scaled = loss / grad_accum
            if scaler is not None:
                scaler.scale(scaled).backward()
            else:
                scaled.backward()

            if (step + 1) % grad_accum == 0:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

        total_loss  += loss.item()
        total_steps += 1
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    return total_loss / max(total_steps, 1)


# ─────────────────────────────────────────────────────────────────────────────
# Main train function
# ─────────────────────────────────────────────────────────────────────────────
def train(
    city:              str   = "A",
    data_dir:          Path  = Path("Data"),
    checkpoint_dir:    Path  = Path("checkpoints"),
    model_size:        str   = "paper",
    epochs:            int   = 200,
    batch_size:        int   = 16,
    grad_accum:        int   = 2,
    lr:                float = 3e-4,
    warmup_ratio:      float = 0.1,
    val_split:         float = 0.05,
    max_users:         Optional[int] = None,
    resume:            Optional[str] = None,
    alpha:             int   = 15,
    max_seq_len:       int   = 2048,
    subset_per_epoch:  Optional[int] = None,
):
    city   = city.upper()
    device = get_device()

    # ── Model ──
    model = build_lpbert(model_size).to(device)
    try:
        model = torch.compile(model)
        print("[train] torch.compile enabled")
    except Exception as e:
        print(f"[train] torch.compile skipped ({e})")

    # ── Dataset (cache per city/users/seqlen) ──
    cache_path = (
        data_dir /
        f"lpbert_city{city}_trainset_users{max_users}_seq{max_seq_len}.pt"
    )

    if cache_path.exists():
        print(f"[train] Loading cached dataset: {cache_path}")
        dataset = torch.load(cache_path, weights_only=False)
    else:
        data_path = city_data_path(data_dir, city)
        df        = load_data(data_path, max_users=max_users)
        trajs     = build_user_trajectories(df)
        dataset   = LPBertDataset(
            trajs,
            day_start   = TRAIN_DAYS[0],
            day_end     = TRAIN_DAYS[1],
            alpha       = alpha,
            max_seq_len = max_seq_len,
        )
        print(f"[train] Saving dataset cache: {cache_path}")
        torch.save(dataset, cache_path)

    if len(dataset) == 0:
        raise RuntimeError("No training samples — check data and day ranges.")

    val_n   = max(1, int(len(dataset) * val_split))
    train_n = len(dataset) - val_n
    train_ds, val_ds = random_split(
        dataset, [train_n, val_n],
        generator=torch.Generator().manual_seed(42),
    )

    pin = device.type == "cuda"
    nw  = 4 if device.type == "cuda" else 0   # MPS/CPU: workers hurt more than help

    # Per-epoch subset sampler — covers all users over many epochs without
    # iterating the full dataset each time (critical for 150k-user runs).
    n_per_epoch = subset_per_epoch or len(train_ds)
    train_sampler = RandomSampler(
        train_ds, replacement=False,
        num_samples=min(n_per_epoch, len(train_ds)),
    )
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, sampler=train_sampler,
        collate_fn=collate_fn, num_workers=nw, pin_memory=pin,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=nw, pin_memory=pin,
    )
    subset_note = f"  subset={n_per_epoch:,}/epoch" if subset_per_epoch else ""
    print(f"[train] Train={len(train_ds):,}  Val={len(val_ds):,}{subset_note}")

    # ── Resume ──
    start_epoch = 0
    if resume is not None:
        ckpt = torch.load(resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        start_epoch = ckpt.get("epoch", 0)
        print(f"[train] Resumed from epoch {start_epoch}: {resume}")

    # ── Optimiser & scheduler ──
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    scaler = None
    if device.type in ("cuda", "mps"):
        try:
            scaler = torch.amp.GradScaler(device_type=device.type)
        except Exception:
            pass

    total_steps  = epochs * math.ceil(len(train_loader) / grad_accum)
    warmup_steps = max(1, int(total_steps * warmup_ratio))
    scheduler    = build_scheduler(optimizer, warmup_steps, total_steps)

    # ── Checkpoint dir ──
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_val     = float("inf")
    best_path    = checkpoint_dir / f"lpbert_city{city}_{model_size}_best.pt"
    log_path     = checkpoint_dir / f"lpbert_city{city}_train_log.csv"

    print(f"\n[train] ═══ LP-BERT  city={city}  epochs={epochs}  "
          f"model={model_size}  batch={batch_size}×{grad_accum}={batch_size*grad_accum}  "
          f"device={device} ═══\n")

    with open(log_path, "w") as f:
        f.write("epoch,train_loss,val_loss,elapsed_s\n")

    for epoch in range(start_epoch, start_epoch + epochs):
        t0 = time.time()

        tr_loss  = run_epoch(model, train_loader, optimizer, scheduler,
                             device, train=True,  grad_accum=grad_accum, scaler=scaler)
        val_loss = run_epoch(model, val_loader,   optimizer, scheduler,
                             device, train=False, scaler=scaler)

        elapsed  = time.time() - t0
        track    = val_loss if val_loss > 0 else tr_loss

        print(f"  Epoch {epoch+1:4d}/{start_epoch+epochs}"
              f"  train={tr_loss:.4f}  val={val_loss:.4f}"
              f"  ({elapsed:.1f}s)")

        with open(log_path, "a") as f:
            f.write(f"{epoch+1},{tr_loss:.6f},{val_loss:.6f},{elapsed:.1f}\n")

        if track < best_val:
            best_val = track
            torch.save({
                "epoch":      epoch + 1,
                "model":      model.state_dict(),
                "optimizer":  optimizer.state_dict(),
                "val_loss":   val_loss,
                "train_loss": tr_loss,
                "model_size": model_size,
                "city":       city,
                "cfg":        model.cfg,
                "alpha":      alpha,
                "max_seq_len": max_seq_len,
            }, best_path)
            print(f"           ✓ Saved best checkpoint: {best_path}")

        # Always save latest
        latest_path = checkpoint_dir / f"lpbert_city{city}_{model_size}_latest.pt"
        torch.save({
            "epoch":      epoch + 1,
            "model":      model.state_dict(),
            "val_loss":   val_loss,
            "train_loss": tr_loss,
            "model_size": model_size,
            "city":       city,
            "cfg":        model.cfg,
            "alpha":      alpha,
            "max_seq_len": max_seq_len,
        }, latest_path)

    print(f"\n[train] Done. Best val_loss={best_val:.4f}")
    print(f"[train] Best checkpoint: {best_path}")
    return str(best_path)
