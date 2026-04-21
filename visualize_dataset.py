#!/usr/bin/env python3
"""
visualize_dataset.py
====================
Dataset visualization and analysis for HuMob challenge.

Generates comprehensive plots showing:
(a) Data completeness rate per city
(b) Seasonality of daily movement count (with emergency period markers)
(c) Mobility reduction during emergency period
(d) User distribution per city
(e) Average records per user
(f) Temporal coverage distribution
"""

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from tqdm import tqdm

from geoformer.data import load_data, build_user_trajectories, TRAIN_DAYS, TEST_DAYS


# ─────────────────────────────────────────────────────────────────────────────
# Data loading and preprocessing
# ─────────────────────────────────────────────────────────────────────────────
def compute_dataset_stats(city: str, data_dir: Path = Path("Data")) -> Dict:
    """
    Compute comprehensive statistics for a city's dataset.

    Returns:
        dict with keys:
        - n_users: number of unique users
        - n_records: total records
        - n_records_train: records in training period (days 1-60)
        - n_records_test: records in test period (days 61-75)
        - daily_movement_count: [75] array of movement counts per day
        - user_record_counts: [n_users] array of records per user
        - data_completeness_per_day: [75] fraction of users with data that day
        - avg_records_per_user: mean records per user
        - users_per_city: total unique users
    """
    data_path = Path("Data") / f"city_{city.upper()}_alldata.parquet"

    if not data_path.exists():
        raise FileNotFoundError(f"Data not found: {data_path}")

    print(f"Loading {city}...")
    df = pd.read_csv(data_path) if str(data_path).endswith('.csv') else pd.read_parquet(data_path)

    # Basic info
    n_users = df["uid"].nunique()
    n_records = len(df)

    # Train/test split
    n_train = len(df[df["d"].isin(range(TRAIN_DAYS[0], TRAIN_DAYS[1] + 1))])
    n_test = len(df[df["d"].isin(range(TEST_DAYS[0], TEST_DAYS[1] + 1))])

    # Daily movement count
    daily_movement = df.groupby("d").size().values
    daily_movement_padded = np.zeros(75)
    daily_movement_padded[:len(daily_movement)] = daily_movement

    # User statistics
    user_counts = df.groupby("uid").size().values

    # Data completeness (fraction of users with data each day)
    data_completeness = []
    for day in range(1, 76):
        users_on_day = df[df["d"] == day]["uid"].nunique()
        completeness = users_on_day / n_users if n_users > 0 else 0
        data_completeness.append(completeness)

    return {
        "n_users": n_users,
        "n_records": n_records,
        "n_records_train": n_train,
        "n_records_test": n_test,
        "daily_movement_count": daily_movement_padded,
        "user_record_counts": user_counts,
        "data_completeness_per_day": np.array(data_completeness),
        "avg_records_per_user": np.mean(user_counts),
        "median_records_per_user": np.median(user_counts),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Individual plots
# ─────────────────────────────────────────────────────────────────────────────
def plot_data_completeness(stats_dict: Dict[str, Dict], output_dir: Path = Path(".")):
    """
    Plot (a): Data completeness rate per city (bar chart).
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    cities = sorted(stats_dict.keys())
    completeness_per_city = [
        np.mean(stats_dict[c]["data_completeness_per_day"]) * 100
        for c in cities
    ]

    bars = ax.bar(cities, completeness_per_city, color="#3498db", edgecolor="black", linewidth=1.5)

    # Add value labels on bars
    for bar, val in zip(bars, completeness_per_city):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax.set_xlabel("City", fontsize=12, fontweight='bold')
    ax.set_ylabel("Data Completeness (%)", fontsize=12, fontweight='bold')
    ax.set_title("Data Completeness Rate per City", fontsize=14, fontweight='bold')
    ax.set_ylim(0, 105)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    plt.tight_layout()
    plt.savefig(output_dir / "dataset_completeness.png", dpi=300, facecolor='white')
    print(f"✓ Saved: {output_dir / 'dataset_completeness.png'}")
    plt.close()


def plot_seasonality_with_emergency(stats_dict: Dict[str, Dict], output_dir: Path = Path(".")):
    """
    Plot (b) & (c): Seasonality of daily movement count with emergency markers.
    Shows all cities overlaid with training/test period markers.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    cities = sorted(stats_dict.keys())
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']

    for idx, (city, color) in enumerate(zip(cities, colors)):
        ax = axes[idx]
        stats = stats_dict[city]

        days = np.arange(1, 76)
        movement = stats["daily_movement_count"]

        ax.plot(days, movement, color=color, linewidth=2.5, marker='o', markersize=3, label=city)

        # Mark training/test boundary
        ax.axvline(x=60.5, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Train/Test')

        # Shade training and test regions
        ax.axvspan(1, 60, alpha=0.1, color='green', label='Training (Days 1-60)')
        ax.axvspan(61, 75, alpha=0.1, color='orange', label='Test (Days 61-75)')

        ax.set_xlabel("Day", fontsize=10, fontweight='bold')
        ax.set_ylabel("Daily Movement Count", fontsize=10, fontweight='bold')
        ax.set_title(f"Seasonality - City {city}", fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='upper right', fontsize=9)

    fig.suptitle("Seasonality of Daily Movement Count with Emergency Period Markers",
                 fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig(output_dir / "dataset_seasonality.png", dpi=300, facecolor='white', bbox_inches='tight')
    print(f"✓ Saved: {output_dir / 'dataset_seasonality.png'}")
    plt.close()


def plot_mobility_comparison(stats_dict: Dict[str, Dict], output_dir: Path = Path(".")):
    """
    Plot (c): Mobility reduction during emergency period.
    Compare training period (pre-emergency) vs test period (emergency).
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    cities = sorted(stats_dict.keys())

    train_avg = []
    test_avg = []

    for city in cities:
        stats = stats_dict[city]
        movement = stats["daily_movement_count"]

        # Days 1-60: training period (before emergency)
        # Days 61-75: test period (during emergency, reduced mobility expected)
        train_avg.append(np.mean(movement[0:60]))
        test_avg.append(np.mean(movement[60:75]))

    x = np.arange(len(cities))
    width = 0.35

    bars1 = ax.bar(x - width/2, train_avg, width, label='Training (Days 1-60)',
                   color='#2ecc71', edgecolor='black', linewidth=1.5)
    bars2 = ax.bar(x + width/2, test_avg, width, label='Test (Days 61-75)',
                   color='#e74c3c', edgecolor='black', linewidth=1.5)

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.0f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_xlabel("City", fontsize=12, fontweight='bold')
    ax.set_ylabel("Average Daily Movement Count", fontsize=12, fontweight='bold')
    ax.set_title("Mobility Reduction During Emergency Period", fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(cities)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    plt.tight_layout()
    plt.savefig(output_dir / "dataset_mobility_reduction.png", dpi=300, facecolor='white')
    print(f"✓ Saved: {output_dir / 'dataset_mobility_reduction.png'}")
    plt.close()


def plot_user_distribution(stats_dict: Dict[str, Dict], output_dir: Path = Path(".")):
    """
    Plot (d): User distribution per city.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    cities = sorted(stats_dict.keys())
    n_users = [stats_dict[c]["n_users"] for c in cities]

    bars = ax.bar(cities, n_users, color=['#e74c3c', '#3498db', '#2ecc71', '#f39c12'],
                  edgecolor='black', linewidth=1.5)

    # Add value labels
    for bar, val in zip(bars, n_users):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(val):,}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax.set_xlabel("City", fontsize=12, fontweight='bold')
    ax.set_ylabel("Number of Users", fontsize=12, fontweight='bold')
    ax.set_title("User Distribution per City", fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    plt.tight_layout()
    plt.savefig(output_dir / "dataset_user_distribution.png", dpi=300, facecolor='white')
    print(f"✓ Saved: {output_dir / 'dataset_user_distribution.png'}")
    plt.close()


def plot_records_per_user(stats_dict: Dict[str, Dict], output_dir: Path = Path(".")):
    """
    Plot (e): Average records per user per city (violin plot).
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    cities = sorted(stats_dict.keys())
    data_to_plot = [stats_dict[c]["user_record_counts"] for c in cities]

    parts = ax.violinplot(data_to_plot, positions=range(len(cities)), showmeans=True, showmedians=True)

    # Customize colors
    for pc in parts['bodies']:
        pc.set_facecolor('#3498db')
        pc.set_alpha(0.7)

    ax.set_xlabel("City", fontsize=12, fontweight='bold')
    ax.set_ylabel("Records per User", fontsize=12, fontweight='bold')
    ax.set_title("Distribution of Records per User", fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(cities)))
    ax.set_xticklabels(cities)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    plt.tight_layout()
    plt.savefig(output_dir / "dataset_records_per_user.png", dpi=300, facecolor='white')
    print(f"✓ Saved: {output_dir / 'dataset_records_per_user.png'}")
    plt.close()


def plot_temporal_coverage(stats_dict: Dict[str, Dict], output_dir: Path = Path(".")):
    """
    Plot (f): Temporal coverage - data completeness over days.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    cities = sorted(stats_dict.keys())
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']

    for idx, (city, color) in enumerate(zip(cities, colors)):
        ax = axes[idx]
        stats = stats_dict[city]

        days = np.arange(1, 76)
        completeness = stats["data_completeness_per_day"] * 100

        ax.fill_between(days, completeness, alpha=0.3, color=color)
        ax.plot(days, completeness, color=color, linewidth=2.5, marker='o', markersize=3)

        # Mark boundary
        ax.axvline(x=60.5, color='red', linestyle='--', linewidth=2, alpha=0.7)
        ax.axvspan(1, 60, alpha=0.05, color='green')
        ax.axvspan(61, 75, alpha=0.05, color='orange')

        ax.set_xlabel("Day", fontsize=10, fontweight='bold')
        ax.set_ylabel("Data Completeness (%)", fontsize=10, fontweight='bold')
        ax.set_title(f"Temporal Coverage - City {city}", fontsize=12, fontweight='bold')
        ax.set_ylim(0, 105)
        ax.grid(True, alpha=0.3, linestyle='--')

    fig.suptitle("Temporal Data Completeness Over 75 Days",
                 fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig(output_dir / "dataset_temporal_coverage.png", dpi=300, facecolor='white', bbox_inches='tight')
    print(f"✓ Saved: {output_dir / 'dataset_temporal_coverage.png'}")
    plt.close()


def plot_combined_overview(stats_dict: Dict[str, Dict], output_dir: Path = Path(".")):
    """
    Plot: Combined overview dashboard (2x3 grid).
    """
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    cities = sorted(stats_dict.keys())

    # (a) Data completeness
    ax_a = fig.add_subplot(gs[0, :2])
    completeness_per_city = [
        np.mean(stats_dict[c]["data_completeness_per_day"]) * 100 for c in cities
    ]
    bars = ax_a.bar(cities, completeness_per_city, color=['#e74c3c', '#3498db', '#2ecc71', '#f39c12'],
                    edgecolor='black', linewidth=1.5)
    for bar, val in zip(bars, completeness_per_city):
        height = bar.get_height()
        ax_a.text(bar.get_x() + bar.get_width()/2., height,
                  f'{val:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax_a.set_ylabel("Completeness (%)", fontsize=11, fontweight='bold')
    ax_a.set_title("(a) Data Completeness Rate per City", fontsize=12, fontweight='bold')
    ax_a.grid(axis='y', alpha=0.3)

    # (d) User distribution
    ax_d = fig.add_subplot(gs[0, 2])
    n_users = [stats_dict[c]["n_users"] for c in cities]
    bars = ax_d.bar(cities, n_users, color=['#e74c3c', '#3498db', '#2ecc71', '#f39c12'],
                    edgecolor='black', linewidth=1.5)
    ax_d.set_ylabel("# Users", fontsize=11, fontweight='bold')
    ax_d.set_title("(d) User Count", fontsize=12, fontweight='bold')
    ax_d.grid(axis='y', alpha=0.3)

    # (c) Mobility comparison
    ax_c = fig.add_subplot(gs[1, :2])
    train_avg = []
    test_avg = []
    for city in cities:
        stats = stats_dict[city]
        movement = stats["daily_movement_count"]
        train_avg.append(np.mean(movement[0:60]))
        test_avg.append(np.mean(movement[60:75]))

    x = np.arange(len(cities))
    width = 0.35
    ax_c.bar(x - width/2, train_avg, width, label='Training', color='#2ecc71', edgecolor='black', linewidth=1.5)
    ax_c.bar(x + width/2, test_avg, width, label='Test', color='#e74c3c', edgecolor='black', linewidth=1.5)
    ax_c.set_xticks(x)
    ax_c.set_xticklabels(cities)
    ax_c.set_ylabel("Avg Daily Movement", fontsize=11, fontweight='bold')
    ax_c.set_title("(c) Mobility: Training vs Test Period", fontsize=12, fontweight='bold')
    ax_c.legend(fontsize=10)
    ax_c.grid(axis='y', alpha=0.3)

    # (e) Records per user
    ax_e = fig.add_subplot(gs[1, 2])
    avg_records = [stats_dict[c]["avg_records_per_user"] for c in cities]
    bars = ax_e.bar(cities, avg_records, color=['#e74c3c', '#3498db', '#2ecc71', '#f39c12'],
                    edgecolor='black', linewidth=1.5)
    ax_e.set_ylabel("Avg Records", fontsize=11, fontweight='bold')
    ax_e.set_title("(e) Avg Records/User", fontsize=12, fontweight='bold')
    ax_e.grid(axis='y', alpha=0.3)

    # (b) & (f) Seasonality for all cities
    for subplot_idx, city in enumerate(cities):
        ax = fig.add_subplot(gs[2, subplot_idx])
        stats = stats_dict[city]
        days = np.arange(1, 76)
        movement = stats["daily_movement_count"]

        ax.plot(days, movement, color=['#e74c3c', '#3498db', '#2ecc71', '#f39c12'][subplot_idx],
                linewidth=2, marker='o', markersize=2)
        ax.axvline(x=60.5, color='black', linestyle='--', linewidth=1.5, alpha=0.5)
        ax.axvspan(1, 60, alpha=0.05, color='green')
        ax.axvspan(61, 75, alpha=0.05, color='orange')
        ax.set_xlabel("Day", fontsize=10, fontweight='bold')
        ax.set_ylabel("Movement Count", fontsize=10, fontweight='bold')
        ax.set_title(f"(b) Seasonality - City {city}", fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)

    fig.suptitle("HuMob Dataset Overview - All Cities", fontsize=16, fontweight='bold', y=0.995)
    plt.savefig(output_dir / "dataset_overview_dashboard.png", dpi=300, facecolor='white', bbox_inches='tight')
    print(f"✓ Saved: {output_dir / 'dataset_overview_dashboard.png'}")
    plt.close()


def plot_summary_statistics(stats_dict: Dict[str, Dict], output_dir: Path = Path(".")):
    """
    Print and save summary statistics table.
    """
    cities = sorted(stats_dict.keys())

    summary_data = []
    for city in cities:
        stats = stats_dict[city]
        summary_data.append({
            'City': city,
            'Users': f"{stats['n_users']:,}",
            'Total Records': f"{stats['n_records']:,}",
            'Train Records': f"{stats['n_records_train']:,}",
            'Test Records': f"{stats['n_records_test']:,}",
            'Avg Records/User': f"{stats['avg_records_per_user']:.1f}",
            'Data Completeness': f"{np.mean(stats['data_completeness_per_day'])*100:.1f}%",
        })

    summary_df = pd.DataFrame(summary_data)

    print("\n" + "="*100)
    print("DATASET SUMMARY STATISTICS")
    print("="*100)
    print(summary_df.to_string(index=False))
    print("="*100 + "\n")

    summary_df.to_csv(output_dir / "dataset_summary.csv", index=False)
    print(f"✓ Saved: {output_dir / 'dataset_summary.csv'}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Visualize HuMob dataset: seasonality, completeness, mobility patterns"
    )
    parser.add_argument("--cities", nargs="+", default=["A", "B", "C", "D"],
                        help="Cities to visualize (default: A B C D)")
    parser.add_argument("--data_dir", default="Data", help="Data directory")
    parser.add_argument("--output_dir", default="dataset_analysis", help="Output directory for plots")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[visualize] Computing dataset statistics for cities: {', '.join(args.cities)}\n")

    # Load stats for all cities
    stats_dict = {}
    for city in args.cities:
        try:
            stats_dict[city] = compute_dataset_stats(city, Path(args.data_dir))
        except Exception as e:
            print(f"  ✗ Failed to load City {city}: {e}")
            continue

    if not stats_dict:
        print("✗ No cities loaded successfully")
        return

    print("\n[visualize] Generating plots...\n")

    # Generate plots
    plot_data_completeness(stats_dict, output_dir)
    plot_seasonality_with_emergency(stats_dict, output_dir)
    plot_mobility_comparison(stats_dict, output_dir)
    plot_user_distribution(stats_dict, output_dir)
    plot_records_per_user(stats_dict, output_dir)
    plot_temporal_coverage(stats_dict, output_dir)
    plot_combined_overview(stats_dict, output_dir)
    plot_summary_statistics(stats_dict, output_dir)

    print(f"\n[visualize] ✓ All plots saved to {output_dir}\n")


if __name__ == "__main__":
    main()
