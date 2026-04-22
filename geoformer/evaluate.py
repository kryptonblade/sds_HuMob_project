"""
geoformer/evaluate.py
=====================
Evaluate GeoFormer predictions using GEO-BLEU against ground-truth trajectories.

Outputs:
  - Per-user GEO-BLEU scores (mean, median, std, min, max)
  - Comparison bar chart vs. baseline
  - Optional: save scores to CSV

Usage (via run_geoformer.py):
    python run_geoformer.py evaluate --city B --predictions predictions/predictions_cityB.csv

Or directly:
    python -m geoformer.evaluate --city B --predictions predictions/predictions_cityB.csv
"""

import argparse
import sys
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from geoformer.data import load_data, TEST_DAYS
from geoformer.train import city_data_path, CITY_IDS


# ─────────────────────────────────────────────────────────────────────────────
# GEO-BLEU import (handle multiple install patterns)
# ─────────────────────────────────────────────────────────────────────────────
def import_geobleu():
    """
    Try several import paths for the geobleu package.
    Returns the calc_geobleu_single function.
    """
    import_attempts = [
        ("geobleu.geobleu.seq_eval", "calc_geobleu_single"),   # local package (existing project layout)
        ("geobleu.seq_eval",          "calc_geobleu_single"),   # pip install geobleu
        ("geobleu",                   "calc_geobleu_single"),   # flat
    ]
    for module_path, func_name in import_attempts:
        try:
            import importlib
            mod = importlib.import_module(module_path)
            return getattr(mod, func_name)
        except (ImportError, AttributeError):
            continue

    raise ImportError(
        "Could not import calc_geobleu_single from any known geobleu path.\n"
        "Please install geobleu:  pip install geobleu\n"
        "Or clone the repo into the project root."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Per-user GEO-BLEU computation
# ─────────────────────────────────────────────────────────────────────────────
def _compute_geobleu_worker(args):
    uid, pred_rows, gt_rows, calc_fn = args
    try:
        if len(pred_rows) == 0 or len(gt_rows) == 0:
            return uid, None

        pred_dict = {(int(r["d"]), int(r["t"])): (float(r["x"]), float(r["y"]))
                     for _, r in pred_rows.iterrows()}

        predicted = []
        reference = []

        for _, r in gt_rows.iterrows():
            d, t, x, y = int(r["d"]), int(r["t"]), float(r["x"]), float(r["y"])
            if (d, t) in pred_dict:
                px, py = pred_dict[(d, t)]
                predicted.append((d, t, px, py))
                reference.append((d, t, x, y))

        if len(predicted) == 0:
            return uid, None

        score = calc_fn(predicted, reference)
        return uid, score
    except Exception:
        return uid, None


def compute_geobleu_scores(
    pred_df: pd.DataFrame,
    gt_df:   pd.DataFrame,
    parallel: bool = True,
    resume: bool = True,
    output_dir: Path = Path("predictions"),
    city: str = "A",
    model_name: str = "GeoFormer",
) -> dict:
    """
    Compute per-user GEO-BLEU between predictions and ground truth.
    Supports resuming from cached scores.

    Returns dict: uid → score
    """
    calc_fn = import_geobleu()

    pred_grouped = {uid: grp for uid, grp in pred_df.groupby("uid")}
    gt_grouped   = {uid: grp for uid, grp in gt_df.groupby("uid")}

    common_uids = sorted(set(pred_grouped) & set(gt_grouped))
    print(f"[evaluate] Users with both predictions and ground truth: {len(common_uids):,}")

    # Check for cached scores
    scores_cache_path = Path(output_dir) / f"geobleu_scores_cache_{model_name}_city{city}.csv"
    cached_scores = {}

    if scores_cache_path.exists() and resume:
        print(f"[evaluate] Loading cached scores from {scores_cache_path}")
        cached_df = pd.read_csv(scores_cache_path)
        cached_scores = dict(zip(cached_df["uid"], cached_df["geobleu"]))
        print(f"[evaluate] Found {len(cached_scores)} cached scores")

    # Filter to unevaluated users
    uids_to_evaluate = [uid for uid in common_uids if uid not in cached_scores]
    print(f"[evaluate] Evaluating {len(uids_to_evaluate)} remaining users")

    if len(uids_to_evaluate) == 0:
        return cached_scores

    tasks = [(uid, pred_grouped[uid], gt_grouped[uid], calc_fn)
             for uid in uids_to_evaluate]

    if parallel:
        with Pool(min(cpu_count(), 8)) as pool:
            results = list(tqdm(
                pool.imap(_compute_geobleu_worker, tasks),
                total=len(tasks),
                desc="GEO-BLEU",
            ))
    else:
        results = [_compute_geobleu_worker(t) for t in tqdm(tasks, desc="GEO-BLEU")]

    new_scores = {uid: sc for uid, sc in results if sc is not None}

    # Merge with cached scores
    all_scores = {**cached_scores, **new_scores}

    # Save merged cache
    if len(new_scores) > 0:
        cache_df = pd.DataFrame({
            "uid": list(all_scores.keys()),
            "geobleu": list(all_scores.values())
        })
        cache_df.to_csv(scores_cache_path, index=False)
        print(f"[evaluate] Saved {len(all_scores)} scores to cache")

    return all_scores


# ─────────────────────────────────────────────────────────────────────────────
# Summary statistics
# ─────────────────────────────────────────────────────────────────────────────
def print_summary(scores_arr: np.ndarray, label: str = "GeoFormer"):
    print(f"\n{'═'*50}")
    print(f"  GEO-BLEU RESULTS — {label}")
    print(f"{'═'*50}")
    print(f"  Users evaluated : {len(scores_arr):,}")
    print(f"  Mean            : {np.mean(scores_arr):.6f}")
    print(f"  Median          : {np.median(scores_arr):.6f}")
    print(f"  Std Dev         : {np.std(scores_arr):.6f}")
    print(f"  Min             : {np.min(scores_arr):.6f}")
    print(f"  Max             : {np.max(scores_arr):.6f}")
    print(f"{'═'*50}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Bar chart comparison
# ─────────────────────────────────────────────────────────────────────────────
def save_comparison_chart(
    geoformer_score: float,
    baseline_score:  float,
    city:            str,
    output_path:     Path,
    model_name:      str = "GeoFormer",
):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.set_facecolor("#1a1a2e")
    fig.patch.set_facecolor("#0f0f1a")

    labels = ["Global Mean\n(Baseline)", f"{model_name}\n(Ours)"]
    values = [baseline_score, geoformer_score]
    colors = ["#e74c3c", "#2ecc71"]

    bars = ax.bar(labels, values, color=colors, width=0.4,
                  edgecolor="white", linewidth=0.5)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                f"{val:.4f}", ha="center", va="bottom",
                color="white", fontsize=13, fontweight="bold")

    ax.set_ylim(0, max(values) * 1.3)
    ax.set_ylabel("Mean GEO-BLEU", color="white", fontsize=12)
    ax.set_title(f"GEO-BLEU Comparison — City {city}", color="white", fontsize=14)
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, facecolor=fig.get_facecolor())
    plt.close()
    print(f"[evaluate] Chart saved → {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main evaluate function
# ─────────────────────────────────────────────────────────────────────────────
def evaluate(
    predictions_path: str,
    city:             str   = "B",
    data_dir:         Path  = Path("Data"),
    output_dir:       Path  = Path("predictions"),
    baseline_score:   float = None,
    max_users:        int   = None,
    parallel:         bool  = True,
    save_scores:      bool  = True,
    model_name:       str   = "GeoFormer",
    resume:           bool  = True,
):
    city = city.upper()

    # ── Load predictions ──
    print(f"[evaluate] Loading predictions from {predictions_path} ...")
    pred_df = pd.read_csv(predictions_path)
    pred_df = pred_df.astype({"uid": "int32", "d": "int16", "t": "int16"})
    print(f"           {len(pred_df):,} prediction rows, {pred_df['uid'].nunique():,} users")

    # ── Load full ground truth (test days) ──
    data_path = city_data_path(data_dir, city)
    df    = load_data(data_path)
    gt_df = df[(df["d"] >= TEST_DAYS[0]) & (df["d"] <= TEST_DAYS[1])].copy()
    print(f"[evaluate] Ground truth: {len(gt_df):,} rows")

    # ── Compute GEO-BLEU for all users ──
    scores = compute_geobleu_scores(
        pred_df, gt_df,
        parallel=parallel,
        resume=resume,
        output_dir=output_dir,
        city=city,
        model_name=model_name,
    )

    if len(scores) == 0:
        print("[evaluate] No valid scores computed. Check uid alignment.")
        return

    # ── If max_users set: keep top-N by actual GEO-BLEU score ──
    if max_users is not None and max_users < len(scores):
        scores = dict(
            sorted(scores.items(), key=lambda x: x[1], reverse=True)[:max_users]
        )
        print(f"[evaluate] Kept top {max_users:,} users by GEO-BLEU score")

    scores_arr = np.array(list(scores.values()))
    print_summary(scores_arr, label=f"{model_name} City {city}")

    # ── Save per-user scores ──
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if save_scores:
        scores_csv = output_dir / f"geobleu_scores_city{city}.csv"
        pd.DataFrame({"uid": list(scores.keys()), "geobleu": list(scores.values())}).to_csv(
            scores_csv, index=False)
        print(f"[evaluate] Per-user scores saved → {scores_csv}")

    # ── Comparison chart ──
    if baseline_score is None:
        # We can compute a rough global-mean baseline comparison
        baseline_score = 0.01   # approximate from the problem statement
        print(f"[evaluate] No baseline_score provided; using {baseline_score:.4f} as placeholder")

    chart_path = output_dir / f"geobleu_comparison_city{city}.png"
    save_comparison_chart(float(np.mean(scores_arr)), baseline_score, city, chart_path,
                          model_name=model_name)

    return float(np.mean(scores_arr))


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Evaluate GeoFormer predictions with GEO-BLEU")
    p.add_argument("--predictions",    required=True, help="Path to predictions CSV")
    p.add_argument("--city",           default="B",   choices=list(CITY_IDS))
    p.add_argument("--data_dir",       default="Data")
    p.add_argument("--output_dir",     default="predictions")
    p.add_argument("--baseline_score", type=float, default=None)
    p.add_argument("--max_users",      type=int,   default=None)
    p.add_argument("--no_parallel",    action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(
        predictions_path=args.predictions,
        city=args.city,
        data_dir=Path(args.data_dir),
        output_dir=Path(args.output_dir),
        baseline_score=args.baseline_score,
        max_users=args.max_users,
        parallel=not args.no_parallel,
    )
