import pandas as pd
import numpy as np
from pathlib import Path
from multiprocessing import Pool, cpu_count
from geobleu.geobleu.seq_eval import calc_geobleu_single
from tqdm import tqdm


# =========================================================
# CONFIG
# =========================================================
DATA_PATH = Path.home() / "Downloads" / "city_A_alldata.parquet"
TRAIN_DAYS = (1, 60)
TEST_DAYS = (61, 75)
USE_PARALLEL = True


# =========================================================
# LOAD DATA
# =========================================================
def load_data(path):
    print(f"Loading data from {path} ...")
    df = pd.read_parquet(path)
    
    df = df.astype({
        'uid': 'int32',
        'd': 'int16',
        't': 'int16',
        'x': 'int32',
        'y': 'int32'
    })
    
    print(f"Loaded {len(df)} rows")
    return df


# =========================================================
# PREPROCESS (sorting removed)
# =========================================================
def preprocess(df):
    print("Splitting train/test...")
    
    df_train = df[(df['d'] >= TRAIN_DAYS[0]) & (df['d'] <= TRAIN_DAYS[1])]
    df_test  = df[(df['d'] >= TEST_DAYS[0]) & (df['d'] <= TEST_DAYS[1])]
    
    print(f"Train rows: {len(df_train)}, Test rows: {len(df_test)}")
    return df_train, df_test


# =========================================================
# GLOBAL MEAN
# =========================================================
def compute_global_mean(df_train):
    print("Computing global mean...")
    
    mean_x = df_train['x'].mean()
    mean_y = df_train['y'].mean()
    
    print(f"Global Mean: ({mean_x:.6f}, {mean_y:.6f})")
    return (mean_x, mean_y)


# =========================================================
# CORE COMPUTATION (PER USER)
# =========================================================
def compute_user_geobleu(args):
    uid, user_df, mean_x, mean_y = args
    
    try:
        ds = user_df['d'].values
        ts = user_df['t'].values
        xs = user_df['x'].values
        ys = user_df['y'].values
        
        if len(ds) == 0:
            return None
        
        reference = list(zip(ds, ts, xs, ys))
        predicted = [(d, t, mean_x, mean_y) for d, t in zip(ds, ts)]
        
        return calc_geobleu_single(predicted, reference)
    
    except Exception:
        return None


# =========================================================
# MAIN PROCESSING
# =========================================================
def compute_all_users(df_test, global_mean):
    print("Grouping users...")
    grouped = df_test.groupby('uid', sort=False)
    
    mean_x, mean_y = global_mean
    
    print("Preparing tasks...")
    tasks = [(uid, user_df, mean_x, mean_y) for uid, user_df in grouped]
    
    print(f"Total users to process: {len(tasks)}")
    
    # =====================================================
    # Parallel / Sequential execution with progress
    # =====================================================
    if USE_PARALLEL:
        print(f"Using multiprocessing ({cpu_count()} cores)...")
        with Pool(cpu_count()) as p:
            scores = list(
                tqdm(
                    p.imap(compute_user_geobleu, tasks),
                    total=len(tasks),
                    desc="Computing Geo-BLEU"
                )
            )
    else:
        print("Using sequential processing...")
        scores = [
            compute_user_geobleu(task)
            for task in tqdm(tasks, desc="Computing Geo-BLEU")
        ]
    
    scores = [s for s in scores if s is not None]
    
    print(f"Valid scores: {len(scores)}")
    return np.array(scores)


# =========================================================
# STATS
# =========================================================
def summarize_scores(scores):
    print("\n===== GEO-BLEU RESULTS =====")
    
    print(f"Users evaluated : {len(scores)}")
    print(f"Mean            : {np.mean(scores):.6f}")
    print(f"Median          : {np.median(scores):.6f}")
    print(f"Std Dev         : {np.std(scores):.6f}")
    print(f"Min             : {np.min(scores):.6f}")
    print(f"Max             : {np.max(scores):.6f}")
    
    print("============================\n")


# =========================================================
# MAIN
# =========================================================
def main():
    df = load_data(DATA_PATH)
    
    df_train, df_test = preprocess(df)
    
    global_mean = compute_global_mean(df_train)
    
    scores = compute_all_users(df_test, global_mean)
    
    summarize_scores(scores)


if __name__ == "__main__":
    main()