#!/usr/bin/env python3
"""
CSV.gz → Parquet converter with progress bars.

Usage:
    python convert_to_parquet.py --input file.csv.gz
"""

import pandas as pd
import gzip
import io
import argparse
from pathlib import Path
from tqdm import tqdm
import time


# =========================================================
# LOAD WITH PROGRESS (CHUNKED)
# =========================================================
def load_csv_with_progress(file_obj, chunksize=1_000_000):
    chunks = []
    total_rows = 0

    print("\n📥 Reading CSV in chunks...")

    reader = pd.read_csv(file_obj, chunksize=chunksize)

    for chunk in tqdm(reader, desc="Reading chunks"):
        chunks.append(chunk)
        total_rows += len(chunk)

    print(f"📊 Total rows loaded: {total_rows}")
    return pd.concat(chunks, ignore_index=True)


# =========================================================
# CORE FUNCTION
# =========================================================
def convert_double_gzip_to_parquet(input_path, output_path=None):
    input_path = Path(input_path).expanduser()

    if not input_path.exists():
        raise FileNotFoundError(f"File not found: {input_path}")

    if output_path is None:
        output_path = input_path.with_suffix('').with_suffix('.parquet')
    else:
        output_path = Path(output_path).expanduser()

    print(f"\n📂 Input file : {input_path}")
    print(f"📦 Output file: {output_path}")

    # =====================================================
    # STEP 1: Load data with progress
    # =====================================================
    print("\n🔄 Loading data...")

    try:
        with gzip.open(input_path, 'rb') as f1:
            with gzip.open(io.BytesIO(f1.read()), 'rt', encoding='utf-8') as f2:
                df = load_csv_with_progress(f2)
        print("✅ Loaded using double gzip decoding")

    except Exception as e:
        print(f"⚠️ Double gzip failed: {e}")
        print("🔁 Trying single gzip...")

        with gzip.open(input_path, 'rt', encoding='utf-8') as f:
            df = load_csv_with_progress(f)

        print("✅ Loaded using single gzip decoding")

    # =====================================================
    # STEP 2: Optimize dtypes
    # =====================================================
    print("\n⚙️ Optimizing data types...")

    dtype_map = {
        'uid': 'int32',
        'd': 'int16',
        't': 'int16',
        'x': 'int32',
        'y': 'int32'
    }

    for col, dtype in tqdm(dtype_map.items(), desc="Casting columns"):
        if col in df.columns:
            df[col] = df[col].astype(dtype)

    # =====================================================
    # STEP 3: Sort
    # =====================================================
    print("\n🔀 Sorting data (this may take time)...")
    df = df.sort_values(['uid', 'd', 't'])

    # =====================================================
    # STEP 4: Save parquet with progress
    # =====================================================
    print("\n💾 Writing Parquet file...")

    start = time.time()

    # fake progress bar (since parquet write is atomic)
    with tqdm(total=1, desc="Writing parquet") as pbar:
        df.to_parquet(
            output_path,
            engine='pyarrow',
            compression='snappy',
            index=False
        )
        pbar.update(1)

    end = time.time()
    print(f"⏱️ Write completed in {end - start:.2f} seconds")
    return output_path.resolve()


# =========================================================
# CLI ENTRY
# =========================================================
def main():
    parser = argparse.ArgumentParser(description="Convert CSV.gz → Parquet with progress bars")

    parser.add_argument("--input", required=True, help="Path to input CSV.gz file")
    parser.add_argument("--output", default=None, help="Optional output parquet path")

    args = parser.parse_args()

    output_path = convert_double_gzip_to_parquet(args.input, args.output)

    print(f"\nOUTPUT_PATH={output_path}")


if __name__ == "__main__":
    main()