#!/usr/bin/env python3
"""
visualize_results.py
====================
Visualization and comparison of model predictions and GEO-BLEU scores.

Generates plots for:
- GEO-BLEU score comparisons across models
- Performance distribution per model
- Score statistics (mean, median, std dev)
- Model performance across cities
"""

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from tqdm import tqdm


# ─────────────────────────────────────────────────────────────────────────────
# GEO-BLEU loading and statistics
# ─────────────────────────────────────────────────────────────────────────────
def load_geobleu_scores(csv_path: str) -> Tuple[np.ndarray, Dict]:
    """
    Load GEO-BLEU scores from a CSV file.

    Expected columns: uid, geobleu (or uid, d, t, x, y with geobleu computed)

    Returns:
        scores: [n_users] array of GEO-BLEU scores
        stats: dict with mean, median, std, min, max
    """
    df = pd.read_csv(csv_path)

    if "geobleu" in df.columns:
        scores = df["geobleu"].values
    else:
        raise ValueError(f"CSV must contain 'geobleu' column. Found: {df.columns.tolist()}")

    # Compute statistics
    stats = {
        "mean": np.mean(scores),
        "median": np.median(scores),
        "std": np.std(scores),
        "min": np.min(scores),
        "max": np.max(scores),
        "n_users": len(scores),
    }

    return scores, stats


def compare_models_per_city(
    model_scores: Dict[str, str],  # {model_name: csv_path}
    city: str,
    output_dir: Path = Path("."),
):
    """
    Compare multiple models for a single city.
    Generates bar chart and distribution plots.

    Args:
        model_scores: dict mapping model name to CSV file path
        city: city identifier (A, B, C, D)
        output_dir: output directory
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    means = []
    stds = []
    model_names = []

    all_scores = {}

    for model_name, csv_path in model_scores.items():
        if not Path(csv_path).exists():
            print(f"  ⚠ Skipping {model_name}: {csv_path} not found")
            continue

        scores, stats = load_geobleu_scores(csv_path)
        means.append(stats["mean"])
        stds.append(stats["std"])
        model_names.append(model_name)
        all_scores[model_name] = scores

    if not means:
        print(f"  ✗ No valid scores for City {city}")
        return

    # Plot 1: Bar chart with error bars
    ax = axes[0]
    colors = plt.cm.Set2(np.linspace(0, 1, len(means)))
    bars = ax.bar(model_names, means, yerr=stds, capsize=5, color=colors, edgecolor='black', linewidth=1.5)

    # Add value labels
    for bar, mean in zip(bars, means):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{mean:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_ylabel("Mean GEO-BLEU", fontsize=11, fontweight='bold')
    ax.set_title(f"Mean GEO-BLEU Comparison - City {city}", fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Plot 2: Box plot (distribution)
    ax = axes[1]
    box_data = [all_scores[m] for m in model_names]
    bp = ax.boxplot(box_data, labels=model_names, patch_artist=True)

    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_ylabel("GEO-BLEU Score", fontsize=11, fontweight='bold')
    ax.set_title(f"GEO-BLEU Distribution - City {city}", fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    fig.suptitle(f"GEO-BLEU Comparison - City {city}", fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / f"geobleu_comparison_city{city}.png", dpi=300, facecolor='white', bbox_inches='tight')
    print(f"✓ Saved: {output_dir / f'geobleu_comparison_city{city}.png'}")
    plt.close()


def plot_geobleu_summary_table(
    model_scores: Dict[str, str],
    cities: List[str],
    output_dir: Path = Path("."),
):
    """
    Generate a summary table of GEO-BLEU scores for all models and cities.
    """
    summary_rows = []

    for model_name, csv_path in model_scores.items():
        row = {"Model": model_name}

        for city in cities:
            # Try to find city-specific CSV
            city_csv = csv_path.replace("City", f"City{city}").replace("city", f"city{city}")
            # Also try adding city to filename
            if not Path(city_csv).exists():
                city_csv = str(Path(csv_path).parent) + f"/geobleu_scores_city{city}.csv"

            if Path(city_csv).exists():
                _, stats = load_geobleu_scores(city_csv)
                row[f"City {city}"] = f"{stats['mean']:.6f}"
            else:
                row[f"City {city}"] = "N/A"

        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)

    print("\n" + "="*100)
    print("GEO-BLEU SCORE SUMMARY")
    print("="*100)
    print(summary_df.to_string(index=False))
    print("="*100 + "\n")

    summary_df.to_csv(output_dir / "geobleu_summary.csv", index=False)
    print(f"✓ Saved: {output_dir / 'geobleu_summary.csv'}")


def plot_model_comparison_across_cities(
    model_scores_dict: Dict[str, Dict[str, str]],  # {city: {model: csv_path}}
    output_dir: Path = Path("."),
):
    """
    Compare all models across all cities in a grid.

    Args:
        model_scores_dict: {city: {model_name: csv_path}}
    """
    cities = sorted(model_scores_dict.keys())
    models = set()
    for city_models in model_scores_dict.values():
        models.update(city_models.keys())
    models = sorted(models)

    if not models:
        print("✗ No models to compare")
        return

    # Build data matrix: models × cities
    means_matrix = np.zeros((len(models), len(cities)))

    for city_idx, city in enumerate(cities):
        for model_idx, model in enumerate(models):
            csv_path = model_scores_dict.get(city, {}).get(model)
            if csv_path and Path(csv_path).exists():
                _, stats = load_geobleu_scores(csv_path)
                means_matrix[model_idx, city_idx] = stats["mean"]
            else:
                means_matrix[model_idx, city_idx] = np.nan

    # Create heatmap
    fig, ax = plt.subplots(figsize=(10, 6))

    im = ax.imshow(means_matrix, cmap='RdYlGn', aspect='auto')

    # Set ticks and labels
    ax.set_xticks(np.arange(len(cities)))
    ax.set_yticks(np.arange(len(models)))
    ax.set_xticklabels(cities)
    ax.set_yticklabels(models)

    # Add text annotations
    for i in range(len(models)):
        for j in range(len(cities)):
            if not np.isnan(means_matrix[i, j]):
                text = ax.text(j, i, f'{means_matrix[i, j]:.4f}',
                              ha="center", va="center", color="black", fontsize=10, fontweight='bold')

    ax.set_xlabel("City", fontsize=12, fontweight='bold')
    ax.set_ylabel("Model", fontsize=12, fontweight='bold')
    ax.set_title("GEO-BLEU Score Heatmap: Models vs Cities", fontsize=14, fontweight='bold')

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("GEO-BLEU Score", fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_dir / "geobleu_heatmap.png", dpi=300, facecolor='white', bbox_inches='tight')
    print(f"✓ Saved: {output_dir / 'geobleu_heatmap.png'}")
    plt.close()


def plot_model_performance_curves(
    model_scores: Dict[str, str],
    city: str,
    output_dir: Path = Path("."),
):
    """
    Plot cumulative distribution of GEO-BLEU scores for each model.
    Shows what fraction of users achieve at least a given score.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    colors = plt.cm.Set2(np.linspace(0, 1, len(model_scores)))

    for (model_name, csv_path), color in zip(model_scores.items(), colors):
        if not Path(csv_path).exists():
            continue

        scores, stats = load_geobleu_scores(csv_path)
        scores_sorted = np.sort(scores)

        # CDF: fraction of users
        cdf = np.arange(1, len(scores_sorted) + 1) / len(scores_sorted)

        # Plot CDF
        axes[0].plot(scores_sorted, cdf, label=model_name, color=color, linewidth=2.5)

        # Plot distribution histogram
        axes[1].hist(scores, bins=50, alpha=0.5, label=model_name, color=color, edgecolor='black')

    axes[0].set_xlabel("GEO-BLEU Score", fontsize=11, fontweight='bold')
    axes[0].set_ylabel("Cumulative Fraction of Users", fontsize=11, fontweight='bold')
    axes[0].set_title(f"CDF of GEO-BLEU Scores - City {city}", fontsize=12, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel("GEO-BLEU Score", fontsize=11, fontweight='bold')
    axes[1].set_ylabel("Number of Users", fontsize=11, fontweight='bold')
    axes[1].set_title(f"Distribution of GEO-BLEU Scores - City {city}", fontsize=12, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(f"GEO-BLEU Score Analysis - City {city}", fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / f"geobleu_distribution_city{city}.png", dpi=300, facecolor='white', bbox_inches='tight')
    print(f"✓ Saved: {output_dir / f'geobleu_distribution_city{city}.png'}")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Visualize GEO-BLEU scores and model comparisons"
    )
    parser.add_argument("--lpbert_dir", default="predictions",
                        help="Directory containing LP-BERT predictions")
    parser.add_argument("--geoformer_dir", default="predictions",
                        help="Directory containing GeoFormer predictions")
    parser.add_argument("--cities", nargs="+", default=["A", "B", "C", "D"],
                        help="Cities to visualize")
    parser.add_argument("--output_dir", default="results_analysis",
                        help="Output directory for plots")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[visualize_results] Generating GEO-BLEU comparison plots\n")

    # Example: Configure model paths for each city
    model_configs = {
        "lpbert": {
            "A": f"{args.lpbert_dir}/geobleu_scores_cityA.csv",
            "B": f"{args.lpbert_dir}/geobleu_scores_cityB.csv",
            "C": f"{args.lpbert_dir}/geobleu_scores_cityC.csv",
            "D": f"{args.lpbert_dir}/geobleu_scores_cityD.csv",
        },
        "geoformer": {
            "A": f"{args.geoformer_dir}/geobleu_scores_cityA.csv",
            "B": f"{args.geoformer_dir}/geobleu_scores_cityB.csv",
            "C": f"{args.geoformer_dir}/geobleu_scores_cityC.csv",
            "D": f"{args.geoformer_dir}/geobleu_scores_cityD.csv",
        }
    }

    # Generate per-city comparison plots
    for city in args.cities:
        model_scores = {}
        for model_name, city_paths in model_configs.items():
            csv_path = city_paths.get(city)
            if csv_path:
                model_scores[model_name] = csv_path

        if model_scores:
            print(f"[visualize_results] Comparing models for City {city}")
            compare_models_per_city(model_scores, city, output_dir)
            plot_model_performance_curves(model_scores, city, output_dir)
        else:
            print(f"[visualize_results] No data found for City {city}")

    print(f"\n[visualize_results] ✓ All comparison plots saved to {output_dir}\n")


if __name__ == "__main__":
    main()
