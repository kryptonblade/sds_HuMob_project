#!/usr/bin/env python3
"""
run_geoformer.py
================
Single-entry-point CLI for the GeoFormer mobility prediction pipeline.

Subcommands:
  train     — train the GeoFormer model on one city
  distill   — perform knowledge distillation (teacher -> student)
  generate  — run constrained autoregressive generation
  evaluate  — compute GEO-BLEU vs ground truth

Examples:
  # Quick smoke test (City B, 500 users, 2 epochs, small model):
  python run_geoformer.py train --city B --max_users 500 --epochs 2 --model_size small

  # Full training on City B:
  python run_geoformer.py train --city B --epochs 5

  # Generate predictions:
  python run_geoformer.py generate --city B --checkpoint checkpoints/geoformer_cityB_medium_best.pt

  # Evaluate:
  python run_geoformer.py evaluate --city B --predictions predictions/predictions_cityB.csv
"""

import argparse
import sys
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_geoformer",
        description="GeoFormer: GPT-style Human Mobility Prediction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ── train ──────────────────────────────────────────────────────────────
    train_p = sub.add_parser("train", help="Train GeoFormer on a city dataset")
    train_p.add_argument("--city",       default="B",    choices=["A","B","C","D"],
                         help="Which city to train on (default: B)")
    train_p.add_argument("--data_dir",   default="Data", help="Data folder")
    train_p.add_argument("--ckpt_dir",   default="checkpoints", help="Checkpoint save dir")
    train_p.add_argument("--model_size", default="medium", choices=["tiny","small","medium","full"],
                         help="Model size preset (tiny=micro, small=CPU-friendly, medium=default, full=paper)")
    train_p.add_argument("--epochs",     type=int,   default=5)
    train_p.add_argument("--batch_size", type=int,   default=16)
    train_p.add_argument("--grad_accum", type=int,   default=2,
                         help="Gradient accumulation steps (effective_batch = batch_size * grad_accum)")
    train_p.add_argument("--lr",         type=float, default=3e-4)
    train_p.add_argument("--max_users",  type=int,   default=None,
                         help="Limit users (e.g. 500 for a quick smoke test)")
    train_p.add_argument("--resume",     default=None,
                         help="Resume from checkpoint path")

    # ── distill ────────────────────────────────────────────────────────────
    dist_p = sub.add_parser("distill", help="Knowledge Distillation (Teacher -> Student)")
    dist_p.add_argument("--teacher",      required=True, help="Path to teacher checkpoint")
    dist_p.add_argument("--student_size", default="small", choices=["tiny","small","medium"],
                        help="Student model size")
    dist_p.add_argument("--city",         default="A", choices=["A","B","C","D"])
    dist_p.add_argument("--epochs",       type=int,   default=5)
    dist_p.add_argument("--batch_size",   type=int,   default=16)
    dist_p.add_argument("--alpha",        type=float, default=0.5, help="Distillation weight")
    dist_p.add_argument("--temp",         type=float, default=2.0, help="Softening temperature")
    dist_p.add_argument("--max_users",    type=int,   default=None)
    dist_p.add_argument("--data_dir",     default="Data")
    dist_p.add_argument("--ckpt_dir",     default="checkpoints")

    # ── generate ────────────────────────────────────────────────────────────
    gen_p = sub.add_parser("generate", help="Generate predictions for days 61–75")
    gen_p.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint file")
    gen_p.add_argument("--city",       default="B",    choices=["A","B","C","D"])
    gen_p.add_argument("--data_dir",   default="Data")
    gen_p.add_argument("--output_dir", default="predictions")
    gen_p.add_argument("--topk",       type=int,   default=5,
                       help="Top-k sampling (0 = greedy)")
    gen_p.add_argument("--temperature",type=float, default=1.0)
    gen_p.add_argument("--max_users",  type=int,   default=None)
    gen_p.add_argument("--batch_size", type=int,   default=16,
                       help="Number of users to generate concurrently")

    # ── evaluate ────────────────────────────────────────────────────────────
    eval_p = sub.add_parser("evaluate", help="Evaluate predictions with GEO-BLEU")
    eval_p.add_argument("--predictions",    required=True, help="Predictions CSV path")
    eval_p.add_argument("--city",           default="B",   choices=["A","B","C","D"])
    eval_p.add_argument("--data_dir",       default="Data")
    eval_p.add_argument("--output_dir",     default="predictions")
    eval_p.add_argument("--baseline_score", type=float, default=None,
                        help="Baseline GEO-BLEU for chart comparison")
    eval_p.add_argument("--max_users",      type=int,   default=None)
    eval_p.add_argument("--no_parallel",    action="store_true")

    return parser


def main():
    parser = build_parser()
    args   = parser.parse_args()

    if args.command == "train":
        # Dependency check
        _check_torch()
        from geoformer.train import train
        train(
            city=args.city,
            data_dir=Path(args.data_dir),
            checkpoint_dir=Path(args.ckpt_dir),
            model_size=args.model_size,
            epochs=args.epochs,
            batch_size=args.batch_size,
            grad_accum=args.grad_accum,
            lr=args.lr,
            max_users=args.max_users,
            resume=args.resume,
        )

    elif args.command == "distill":
        _check_torch()
        from geoformer.train import distill
        distill(
            teacher_ckpt=args.teacher,
            student_size=args.student_size,
            city=args.city,
            alpha=args.alpha,
            temperature=args.temp,
            data_dir=Path(args.data_dir),
            checkpoint_dir=Path(args.ckpt_dir),
            epochs=args.epochs,
            batch_size=args.batch_size,
            max_users=args.max_users,
        )

    elif args.command == "generate":
        _check_torch()
        from geoformer.generate import generate
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

    elif args.command == "evaluate":
        from geoformer.evaluate import evaluate
        evaluate(
            predictions_path=args.predictions,
            city=args.city,
            data_dir=Path(args.data_dir),
            output_dir=Path(args.output_dir),
            baseline_score=args.baseline_score,
            max_users=args.max_users,
            parallel=not args.no_parallel,
        )


def _check_torch():
    try:
        import torch
        print(f"[info] PyTorch {torch.__version__} detected")
    except ImportError:
        print("❌ PyTorch is not installed.")
        print("   Install it with:")
        print("   python3 -m pip install torch")
        sys.exit(1)


if __name__ == "__main__":
    main()
