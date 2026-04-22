#!/usr/bin/env python3
"""
run_lpbert.py
=============
Single-entry-point CLI for the LP-BERT mobility prediction pipeline.

Subcommands:
  train     — train LP-BERT on one city with continuous masked-location modelling
  generate  — predict days 61–75 for all users in one parallel forward pass
  evaluate  — compute GEO-BLEU vs ground truth (reuses GeoFormer evaluator)

Examples:
  # Quick smoke test (City A, 500 users, 5 epochs, paper-size model):
  python run_lpbert.py train --city A --max_users 500 --epochs 5 --model_size paper

  # Full training (City A, paper config: 200 epochs):
  python run_lpbert.py train --city A --epochs 200 --model_size paper --batch_size 16

  # Generate predictions:
  python run_lpbert.py generate --city A --checkpoint checkpoints/lpbert_cityA_paper_best.pt

  # Evaluate:
  python run_lpbert.py evaluate --city A --predictions predictions/lpbert_predictions_cityA.csv
"""

import argparse
import sys
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_lpbert",
        description="LP-BERT: BERT-based Human Mobility Prediction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ── train ─────────────────────────────────────────────────────────────────
    train_p = sub.add_parser("train", help="Train LP-BERT on a city dataset")
    train_p.add_argument("--city",       default="A",     choices=["A","B","C","D"])
    train_p.add_argument("--data_dir",   default="Data",  help="Data folder")
    train_p.add_argument("--ckpt_dir",   default="checkpoints")
    train_p.add_argument("--model_size", default="paper", choices=["small","paper","medium"],
                         help="Model size: small=dev, paper=128-dim (default), medium=256-dim")
    train_p.add_argument("--epochs",     type=int,   default=200,
                         help="Training epochs (paper uses 200)")
    train_p.add_argument("--batch_size", type=int,   default=16,
                         help="Per-device batch size (paper uses 16)")
    train_p.add_argument("--grad_accum", type=int,   default=2,
                         help="Gradient accumulation steps")
    train_p.add_argument("--lr",         type=float, default=3e-4)
    train_p.add_argument("--alpha",      type=int,   default=15,
                         help="Consecutive days to mask (paper uses α=15)")
    train_p.add_argument("--max_seq_len",type=int,   default=2048,
                         help="Maximum sequence length per user")
    train_p.add_argument("--max_users",  type=int,   default=None,
                         help="Limit users (e.g. 500 for smoke test)")
    train_p.add_argument("--resume",            default=None,
                         help="Resume from checkpoint path")
    train_p.add_argument("--subset_per_epoch", type=int, default=None,
                         help="Random users sampled per epoch (None=all). "
                              "Use e.g. 15000 for 150k-user runs to keep epoch time manageable.")

    # ── generate ──────────────────────────────────────────────────────────────
    gen_p = sub.add_parser("generate", help="Generate predictions for days 61–75")
    gen_p.add_argument("--checkpoint", required=True)
    gen_p.add_argument("--city",       default="A", choices=["A","B","C","D"])
    gen_p.add_argument("--data_dir",   default="Data")
    gen_p.add_argument("--output_dir", default="predictions")
    gen_p.add_argument("--beta",       type=float, default=0.9,
                       help="β penalty for consecutive same-day predictions (paper β=0.9)")
    gen_p.add_argument("--topk",       type=int,   default=5,
                       help="Top-k sampling (0=greedy)")
    gen_p.add_argument("--batch_size", type=int,   default=32,
                       help="Users processed per forward pass")
    gen_p.add_argument("--max_users",  type=int,   default=None)
    gen_p.add_argument("--max_seq_len",type=int,   default=2048)

    # ── evaluate ──────────────────────────────────────────────────────────────
    eval_p = sub.add_parser("evaluate", help="Evaluate predictions with GEO-BLEU")
    eval_p.add_argument("--predictions",    required=True)
    eval_p.add_argument("--city",           default="A", choices=["A","B","C","D"])
    eval_p.add_argument("--data_dir",       default="Data")
    eval_p.add_argument("--output_dir",     default="predictions")
    eval_p.add_argument("--baseline_score", type=float, default=None)
    eval_p.add_argument("--max_users",      type=int,   default=None)
    eval_p.add_argument("--no_parallel",    action="store_true")

    return parser


def main():
    parser = build_parser()
    args   = parser.parse_args()

    _check_torch()

    if args.command == "train":
        from lpbert.train import train
        train(
            city              = args.city,
            data_dir          = Path(args.data_dir),
            checkpoint_dir    = Path(args.ckpt_dir),
            model_size        = args.model_size,
            epochs            = args.epochs,
            batch_size        = args.batch_size,
            grad_accum        = args.grad_accum,
            lr                = args.lr,
            alpha             = args.alpha,
            max_seq_len       = args.max_seq_len,
            max_users         = args.max_users,
            resume            = args.resume,
            subset_per_epoch  = args.subset_per_epoch,
        )

    elif args.command == "generate":
        from lpbert.generate import generate
        generate(
            checkpoint  = args.checkpoint,
            city        = args.city,
            data_dir    = Path(args.data_dir),
            output_dir  = Path(args.output_dir),
            beta        = args.beta,
            topk        = args.topk,
            batch_size  = args.batch_size,
            max_users   = args.max_users,
            max_seq_len = args.max_seq_len,
        )

    elif args.command == "evaluate":
        from geoformer.evaluate import evaluate
        evaluate(
            predictions_path = args.predictions,
            city             = args.city,
            data_dir         = Path(args.data_dir),
            output_dir       = Path(args.output_dir),
            baseline_score   = args.baseline_score,
            max_users        = args.max_users,
            parallel         = not args.no_parallel,
            model_name       = "LP-BERT",
        )


def _check_torch():
    try:
        import torch
        print(f"[info] PyTorch {torch.__version__}")
    except ImportError:
        print("✗ PyTorch not installed.  pip install torch")
        sys.exit(1)


if __name__ == "__main__":
    main()
