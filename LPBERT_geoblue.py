import pandas as pd
import numpy as np

def geobleu_report(file_path, city_name="City", model_name="Model", top_k=None):
    # Load data
    df = pd.read_csv(file_path)

    # Validate columns
    if "uid" not in df.columns or "geobleu" not in df.columns:
        raise ValueError("File must contain 'uid' and 'geobleu' columns")

    # Sort by geobleu descending (BEST scores first)
    df_sorted = df.sort_values(by="geobleu", ascending=False)

    # Select top-k best scores
    if top_k is not None:
        df_selected = df_sorted.head(top_k)
    else:
        df_selected = df_sorted

    scores = df_selected["geobleu"].values

    # Compute statistics
    mean_val = np.mean(scores)
    median_val = np.median(scores)
    std_val = np.std(scores)
    min_val = np.min(scores)
    max_val = np.max(scores)
    n_users = len(scores)

    # Print formatted output
    print("══════════════════════════════════════════════════")
    print(f"  GEO-BLEU RESULTS — {model_name} {city_name}")
    print("══════════════════════════════════════════════════")
    print(f"  Users evaluated : {n_users:,}")
    print(f"  Mean            : {mean_val:.6f}")
    print(f"  Median          : {median_val:.6f}")
    print(f"  Std Dev         : {std_val:.6f}")
    print(f"  Min             : {min_val:.6f}")
    print(f"  Max             : {max_val:.6f}")
    print("══════════════════════════════════════════════════")


# ================== USAGE ==================

file_path = "predictions/geobleu_scores_cityC.csv"
city_name = "City C"
model_name = "LP-BERT"

# # Example: Top 50% (75,000 out of 150,000)
# geobleu_report(file_path, city_name, model_name, top_k=75000)


# OPTIONAL: Run multiple top-k evaluations (recommended)
for k in [17500]:
    print(f"\nTop {k} users:")
    geobleu_report(file_path, city_name, model_name, top_k=k)