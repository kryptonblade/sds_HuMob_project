"""
geoformer/train.py
==================
Training loop for GeoFormer on HuMob data.

Usage (via run_geoformer.py):
    python run_geoformer.py train --city B --epochs 5 --max_users 5000

Or directly:
    python -m geoformer.train --city B --epochs 2 --max_users 500 --model_size small
"""

import argparse
import math
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, random_split

from geoformer.data import (
    MobilityDataset, collate_fn, build_user_trajectories, load_data,
    TRAIN_DAYS,
)
from geoformer.model import build_model, GeoFormerConfig, GeoFormer


# ─────────────────────────────────────────────────────────────────────────────
# Device selection
# ─────────────────────────────────────────────────────────────────────────────
def get_device() -> torch.device:
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        print(f"[train] Using CUDA GPU: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        dev = torch.device("mps")
        print("[train] Using Apple Metal MPS backend")
    else:
        dev = torch.device("cpu")
        print("[train] Using CPU (consider using a GPU for faster training)")
    return dev


# ─────────────────────────────────────────────────────────────────────────────
# City helpers
# ─────────────────────────────────────────────────────────────────────────────
CITY_IDS = {"A": 0, "B": 1, "C": 2, "D": 3}


def city_data_path(data_dir: Path, city: str) -> Path:
    """Return path to city data file, preferring parquet over csv.gz."""
    parquet = data_dir / f"city_{city}_alldata.parquet"
    csv_gz  = data_dir / f"city_{city}_alldata.csv.gz"
    if parquet.exists():
        return parquet
    if csv_gz.exists():
        return csv_gz
    raise FileNotFoundError(
        f"No data file found for city {city} in {data_dir}. "
        f"Expected {parquet} or {csv_gz}."
    )


# ─────────────────────────────────────────────────────────────────────────────
# LR schedule: linear warmup → cosine decay
# ─────────────────────────────────────────────────────────────────────────────
def build_scheduler(optimizer, warmup_steps: int, total_steps: int):
    warmup = LinearLR(optimizer,
                      start_factor=1e-4,
                      end_factor=1.0,
                      total_iters=warmup_steps)
    cosine = CosineAnnealingLR(optimizer,
                                T_max=max(total_steps - warmup_steps, 1),
                                eta_min=1e-6)
    return SequentialLR(optimizer, [warmup, cosine], milestones=[warmup_steps])


# ─────────────────────────────────────────────────────────────────────────────
# Training epoch
# ─────────────────────────────────────────────────────────────────────────────
from tqdm import tqdm

def run_epoch(model: GeoFormer,
              loader: DataLoader,
              optimizer,
              scheduler,
              device: torch.device,
              city_id: int,
              train: bool = True,
              grad_accum: int = 1,
              scaler = None) -> float:
    model.train(train)
    total_loss  = 0.0
    total_steps = 0

    city_tensor = torch.tensor(city_id, dtype=torch.long, device=device)

    if train:
        optimizer.zero_grad()

    pbar = tqdm(loader, desc="Training" if train else "Validating", leave=False)

    for step, batch in enumerate(pbar):
        tokens    = batch["tokens"].to(device)
        tod       = batch["tod"].to(device)
        dow       = batch["dow"].to(device)
        labels    = batch["labels"].to(device)
        attn_mask = batch["attn_mask"].to(device)

        city_ids = city_tensor.expand(tokens.size(0))

        with torch.set_grad_enabled(train):
            with torch.autocast(device_type=device.type, dtype=torch.float16):
                _, loss, _ = model(
                    tokens=tokens,
                    tod=tod,
                    dow=dow,
                    city_id=city_ids,
                    key_padding_mask=attn_mask,
                    labels=labels,
                )

        if loss is None or torch.isnan(loss):
            if train:
                optimizer.zero_grad()
            continue

        if train:
            # Scale loss for gradient accumulation
            scaled_loss = loss / grad_accum
            
            if scaler is not None:
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()

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
# Distillation epoch
# ─────────────────────────────────────────────────────────────────────────────
def run_distill_epoch(model_student: GeoFormer,
                      model_teacher: GeoFormer,
                      loader: DataLoader,
                      optimizer,
                      scheduler,
                      device: torch.device,
                      city_id: int,
                      train: bool = True,
                      grad_accum: int = 1,
                      scaler = None,
                      alpha: float = 0.5,
                      temperature: float = 2.0) -> float:
    """
    Run one epoch of Knowledge Distillation.
    teacher is fixed (eval mode), student is trained.
    """
    model_student.train(train)
    model_teacher.eval()
    total_loss  = 0.0
    total_steps = 0

    city_tensor = torch.tensor(city_id, dtype=torch.long, device=device)

    if train:
        optimizer.zero_grad()

    pbar = tqdm(loader, desc="Distilling" if train else "Validating", leave=False)

    for step, batch in enumerate(pbar):
        tokens    = batch["tokens"].to(device)
        tod       = batch["tod"].to(device)
        dow       = batch["dow"].to(device)
        labels    = batch["labels"].to(device)
        attn_mask = batch["attn_mask"].to(device)

        city_ids = city_tensor.expand(tokens.size(0))

        with torch.set_grad_enabled(train):
            with torch.autocast(device_type=device.type, dtype=torch.float16):
                # Student forward
                logits_s, loss_ce, _ = model_student(
                    tokens=tokens, tod=tod, dow=dow, city_id=city_ids,
                    key_padding_mask=attn_mask, labels=labels
                )
                
                # Teacher forward (no grad)
                with torch.no_grad():
                    logits_t, _, _ = model_teacher(
                        tokens=tokens, tod=tod, dow=dow, city_id=city_ids,
                        key_padding_mask=attn_mask
                    )

                # ── Knowledge Distillation Loss ──
                # We distill on the same "shift" positions as cross-entropy
                # logits: [B, T, V] -> [B, T-1, V]
                shift_s      = logits_s[:, :-1, :].contiguous()
                shift_t      = logits_t[:, :-1, :].contiguous()
                shift_labels = labels[:, 1:].contiguous()
                
                valid_mask = (shift_labels != -100).view(-1)
                
                if valid_mask.any():
                    V = logits_s.size(-1)
                    # Soften student and teacher
                    s_soft = F.log_softmax(shift_s.view(-1, V) / temperature, dim=-1)[valid_mask]
                    t_soft = F.softmax(shift_t.view(-1, V) / temperature, dim=-1)[valid_mask]
                    
                    loss_kd = F.kl_div(s_soft, t_soft, reduction="sum") / valid_mask.sum()
                    # Rescale by T^2 as per original KD paper
                    loss = alpha * loss_ce + (1 - alpha) * loss_kd * (temperature ** 2)
                else:
                    loss = loss_ce

        if loss is None or torch.isnan(loss):
            if train:
                optimizer.zero_grad()
            continue

        if train:
            scaled_loss = loss / grad_accum
            if scaler is not None:
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()

            if (step + 1) % grad_accum == 0:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model_student.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    nn.utils.clip_grad_norm_(model_student.parameters(), max_norm=1.0)
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

        total_loss  += loss.item()
        total_steps += 1
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    return total_loss / max(total_steps, 1)


# ─────────────────────────────────────────────────────────────────────────────
# Main training function
# ─────────────────────────────────────────────────────────────────────────────
def train(
    city:          str   = "B",
    data_dir:      Path  = Path("Data"),
    checkpoint_dir: Path = Path("checkpoints"),
    model_size:    str   = "medium",
    epochs:        int   = 5,
    batch_size:    int   = 16,
    grad_accum:    int   = 2,
    lr:            float = 3e-4,
    warmup_ratio:  float = 0.1,
    val_split:     float = 0.05,
    max_users:     int   = None,
    resume:        str   = None,
    log_interval:  int   = 50,
):
    city = city.upper()
    city_id = CITY_IDS[city]
    device  = get_device()

    # ── Model (build first so we know max_seq_len for Dataset) ──
    model = build_model(model_size).to(device)

    dataset_cache_path = data_dir / f"city_{city}_train_dataset_cache_seq{model.cfg.max_seq_len}_users{max_users}_chunked.pt"

    if dataset_cache_path.exists():
        print(f"[train] Loading pre-processed training dataset from local cache: {dataset_cache_path}")
        dataset = torch.load(dataset_cache_path, weights_only=False)
    else:
        data_path = city_data_path(data_dir, city)
        df = load_data(data_path, max_users=max_users)
        trajectories = build_user_trajectories(df)

        dataset = MobilityDataset(
            trajectories,
            day_start=TRAIN_DAYS[0],
            day_end=TRAIN_DAYS[1],
            city_id=city_id,
            max_seq_len=model.cfg.max_seq_len,
        )
        print(f"[train] Saving compiled dataset to local cache for next time: {dataset_cache_path}")
        torch.save(dataset, dataset_cache_path)

    if len(dataset) == 0:
        raise RuntimeError("No training samples built — check data and day ranges.")

    val_size   = max(1, int(len(dataset) * val_split))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size],
                                    generator=torch.Generator().manual_seed(42))

    use_pin_memory = (device.type == "cuda")
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              collate_fn=collate_fn, num_workers=4, pin_memory=use_pin_memory)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              collate_fn=collate_fn, num_workers=4, pin_memory=use_pin_memory)

    print(f"[train] Train samples: {len(train_ds):,}  |  Val samples: {len(val_ds):,}")

    # ── Model (already built above) ──
    start_epoch = 0
    if resume is not None:
        ckpt = torch.load(resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        start_epoch = ckpt.get("epoch", 0)
        print(f"[train] Resumed from epoch {start_epoch}: {resume}")

    # ── Optimizer & Scheduler ──
    optimizer    = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    
    # ── Scaler for Mixed Precision ──
    scaler = None
    if device.type in ["cuda", "mps"]:
        try:
            scaler = torch.amp.GradScaler(device_type=device.type)
        except Exception:
            if device.type == "cuda":
                try:
                    scaler = torch.cuda.amp.GradScaler()
                except Exception:
                    pass

    # effective batch = batch_size * grad_accum
    eff_batch    = batch_size * grad_accum
    total_steps  = epochs * math.ceil(len(train_loader) / grad_accum)
    warmup_steps = max(1, int(total_steps * warmup_ratio))
    scheduler    = build_scheduler(optimizer, warmup_steps, total_steps)

    # ── Checkpoint dir ──
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")
    best_ckpt_path = checkpoint_dir / f"geoformer_city{city}_{model_size}_best.pt"
    log_path = checkpoint_dir / f"train_log_city{city}.csv"

    print(f"\n[train] ═══ Starting training: city={city}, epochs={epochs}, "
          f"model={model_size}, batch={batch_size}×accum{grad_accum}={batch_size*grad_accum}, "
          f"device={device} ═══\n")

    with open(log_path, "w") as f_log:
        f_log.write("epoch,train_loss,val_loss,elapsed_s\n")

    for epoch in range(start_epoch, start_epoch + epochs):
        t0 = time.time()

        train_loss = run_epoch(model, train_loader, optimizer, scheduler,
                               device, city_id, train=True,
                               grad_accum=grad_accum, scaler=scaler)
        val_loss   = run_epoch(model, val_loader,   optimizer, scheduler,
                               device, city_id, train=False, scaler=scaler)

        elapsed = time.time() - t0

        # If val set returned 0 (no valid label positions), use train_loss for tracking
        checkpoint_metric = val_loss if val_loss > 0 else train_loss

        print(f"  Epoch {epoch+1:3d}/{start_epoch+epochs}  "
              f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
              f"({elapsed:.1f}s)")

        with open(log_path, "a") as f_log:
            f_log.write(f"{epoch+1},{train_loss:.6f},{val_loss:.6f},{elapsed:.1f}\n")

        # Save best checkpoint
        if checkpoint_metric < best_val_loss:
            best_val_loss = checkpoint_metric
            torch.save({
                "epoch":      epoch + 1,
                "model":      model.state_dict(),
                "optimizer":  optimizer.state_dict(),
                "val_loss":   val_loss,
                "train_loss": train_loss,
                "model_size": model_size,
                "city":       city,
                "city_id":    city_id,
                "cfg":        model.cfg,
            }, best_ckpt_path)
            print(f"           ✓ Saved best checkpoint: {best_ckpt_path}")

        # Always save latest
        latest_path = checkpoint_dir / f"geoformer_city{city}_{model_size}_latest.pt"
        torch.save({
            "epoch":      epoch + 1,
            "model":      model.state_dict(),
            "optimizer":  optimizer.state_dict(),
            "val_loss":   val_loss,
            "train_loss": train_loss,
            "model_size": model_size,
            "city":       city,
            "city_id":    city_id,
            "cfg":        model.cfg,
        }, latest_path)

    print(f"\n[train] ═══ Done. Best val_loss={best_val_loss:.4f} ═══")
    print(f"[train] Best checkpoint: {best_ckpt_path}")
    print(f"[train] Training log:    {log_path}")
    return str(best_ckpt_path)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def distill(
    teacher_ckpt:  str,
    student_size:  str   = "small",
    city:          str   = "B",
    alpha:         float = 0.5,
    temperature:   float = 2.0,
    data_dir:      Path  = Path("Data"),
    checkpoint_dir: Path = Path("checkpoints"),
    epochs:        int   = 5,
    batch_size:    int   = 16,
    grad_accum:    int   = 2,
    lr:            float = 3e-4,
    warmup_ratio:  float = 0.1,
    val_split:     float = 0.05,
    max_users:     int   = None,
    resume:        str   = None,
):
    city = city.upper()
    city_id = CITY_IDS[city]
    device  = get_device()

    # ── 1. Load Teacher ──
    print(f"[distill] Loading teacher checkpoint: {teacher_ckpt}")
    t_ckpt = torch.load(teacher_ckpt, map_location=device, weights_only=False)
    if "cfg" in t_ckpt:
        t_model = GeoFormer(t_ckpt["cfg"]).to(device)
    else:
        # Fallback to model_size if cfg not found
        t_size = t_ckpt.get("model_size", "medium")
        t_model = build_model(t_size).to(device)
    
    t_model.load_state_dict(t_ckpt["model"])
    t_model.eval()
    print(f"[distill] Teacher loaded ({t_model.num_parameters():,} params)")

    # ── 2. Build Student ──
    s_model = build_model(student_size).to(device)
    if resume:
        print(f"[distill] Resuming student from: {resume}")
        s_ckpt = torch.load(resume, map_location=device)
        s_model.load_state_dict(s_ckpt["model"])

    # ── 3. Data ──
    dataset_cache_path = data_dir / f"city_{city}_train_dataset_cache_seq{s_model.cfg.max_seq_len}_users{max_users}_chunked.pt"
    if dataset_cache_path.exists():
        print(f"[distill] Loading pre-processed training dataset: {dataset_cache_path}")
        dataset = torch.load(dataset_cache_path, weights_only=False)
    else:
        data_path = city_data_path(data_dir, city)
        df = load_data(data_path, max_users=max_users)
        trajectories = build_user_trajectories(df)
        dataset = MobilityDataset(trajectories, day_start=TRAIN_DAYS[0], day_end=TRAIN_DAYS[1],
                                  city_id=city_id, max_seq_len=s_model.cfg.max_seq_len)
        torch.save(dataset, dataset_cache_path)

    val_size = max(1, int(len(dataset) * val_split))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))

    use_pin_memory = (device.type == "cuda")
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=4, pin_memory=use_pin_memory)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=4, pin_memory=use_pin_memory)

    # ── 4. Optimizer & Scaler ──
    optimizer = AdamW(s_model.parameters(), lr=lr, weight_decay=0.01)
    
    # ── Scaler for Mixed Precision ──
    scaler = None
    if device.type in ["cuda", "mps"]:
        try:
            scaler = torch.amp.GradScaler(device_type=device.type)
        except Exception:
            if device.type == "cuda":
                try:
                    scaler = torch.cuda.amp.GradScaler()
                except Exception:
                    pass

    total_steps  = epochs * math.ceil(len(train_loader) / grad_accum)
    warmup_steps = max(1, int(total_steps * warmup_ratio))
    scheduler    = build_scheduler(optimizer, warmup_steps, total_steps)

    # ── 5. Training Loop ──
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")
    best_ckpt_path = checkpoint_dir / f"geoformer_city{city}_{student_size}_distilled_best.pt"

    print(f"\n[distill] ═══ Distilling teacher -> {student_size} student (alpha={alpha}, T={temperature}) ═══\n")

    for epoch in range(epochs):
        t0 = time.time()
        train_loss = run_distill_epoch(s_model, t_model, train_loader, optimizer, scheduler, device, city_id,
                                        train=True, grad_accum=grad_accum, scaler=scaler, alpha=alpha, temperature=temperature)
        val_loss = run_distill_epoch(s_model, t_model, val_loader, optimizer, scheduler, device, city_id,
                                      train=False, scaler=scaler, alpha=alpha, temperature=temperature)
        elapsed = time.time() - t0

        print(f"  Epoch {epoch+1:2d}/{epochs}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  ({elapsed:.1f}s)")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "epoch": epoch + 1,
                "model": s_model.state_dict(),
                "val_loss": val_loss,
                "model_size": student_size,
                "city": city,
                "cfg": s_model.cfg,
            }, best_ckpt_path)
            print(f"           ✓ Saved best distilled: {best_ckpt_path}")

    return str(best_ckpt_path)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Train or Distill GeoFormer")
    sub = p.add_subparsers(dest="command")

    # train
    train_p = sub.add_parser("train")
    train_p.add_argument("--city",        default="B",       choices=list(CITY_IDS))
    train_p.add_argument("--data_dir",    default="Data")
    train_p.add_argument("--ckpt_dir",    default="checkpoints")
    train_p.add_argument("--model_size",  default="medium",  choices=["small","medium","full"])
    train_p.add_argument("--epochs",      type=int, default=5)
    train_p.add_argument("--batch_size",  type=int, default=32)
    train_p.add_argument("--lr",          type=float, default=3e-4)
    train_p.add_argument("--max_users",   type=int, default=None)
    train_p.add_argument("--resume",      default=None)

    # distill
    dist_p = sub.add_parser("distill")
    dist_p.add_argument("--teacher",     required=True)
    dist_p.add_argument("--student",     default="small")
    dist_p.add_argument("--alpha",       type=float, default=0.5)
    dist_p.add_argument("--temp",        type=float, default=2.0)
    dist_p.add_argument("--city",        default="A")
    dist_p.add_argument("--epochs",      type=int,   default=5)
    dist_p.add_argument("--batch_size",  type=int,   default=16)
    dist_p.add_argument("--max_users",   type=int,   default=None)

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.command == "train":
        train(city=args.city, data_dir=Path(args.data_dir), checkpoint_dir=Path(args.ckpt_dir),
              model_size=args.model_size, epochs=args.epochs, batch_size=args.batch_size,
              lr=args.lr, max_users=args.max_users, resume=args.resume)
    elif args.command == "distill":
        distill(teacher_ckpt=args.teacher, student_size=args.student, alpha=args.alpha,
                temperature=args.temp, city=args.city, epochs=args.epochs,
                batch_size=args.batch_size, max_users=args.max_users)
