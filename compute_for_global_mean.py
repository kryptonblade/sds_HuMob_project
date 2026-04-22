import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from pathlib import Path
from tqdm import tqdm
import gc
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp

# =========================================================
# DATASET (FAST + MEMORY SAFE)
# =========================================================

class MobTimeSeriesDataset(Dataset):
    def __init__(self, dataset,
                 input_seq_length,
                 predict_seq_length,
                 subsample=False,
                 subsample_number=100,
                 look_back_len=24,
                 multiple=2):

        self.input_seq_length = input_seq_length
        self.predict_seq_length = predict_seq_length
        self.look_back_len = look_back_len
        self.multiple = multiple

        dataset = pd.DataFrame(dataset)

        # vectorized label
        dataset['label'] = 200 * (dataset['x'].values - 1) + (dataset['y'].values - 1)

        # memory-safe storage
        self.user_data = []
        self.index_map = []

        # 🔥 efficient grouping
        grouped = dataset.groupby('uid', sort=False)
        uid_to_indices = grouped.indices

        uids = list(uid_to_indices.keys())
        total_users = len(uids)

        print(f"Processing {total_users} users...")

        # Process users in parallel batches for better performance
        batch_size = min(1000, total_users // mp.cpu_count() + 1)
        
        for batch_start in tqdm(range(0, total_users, batch_size), desc="User batches", unit="batch"):
            batch_end = min(batch_start + batch_size, total_users)
            batch_uids = uids[batch_start:batch_end]
            
            # Process batch in parallel
            with ThreadPoolExecutor(max_workers=min(8, mp.cpu_count())) as executor:
                batch_results = list(executor.map(self._process_user, 
                                                 [(uid, uid_to_indices[uid], dataset, input_seq_length, predict_seq_length, look_back_len, subsample, subsample_number, batch_start + i, total_users, multiple) 
                                                  for i, uid in enumerate(batch_uids)]))
            
            # Collect results
            for result in batch_results:
                if result is not None:
                    user_data, index_maps = result
                    user_idx = len(self.user_data)
                    self.user_data.append(user_data)
                    # Adjust user index in index_maps
                    adjusted_maps = [(user_idx, start) for _, start in index_maps]
                    self.index_map.extend(adjusted_maps)
            
            # Clear memory
            gc.collect()
        
        print(f"Total sequences: {len(self.index_map)}")

    def _process_user(self, args):
        uid, indices, dataset, input_seq_length, predict_seq_length, look_back_len, subsample, subsample_number, idx, total_users, multiple = args
        
        if subsample and idx >= subsample_number:
            return None
            
        # zero-copy slice
        uid_df = dataset.iloc[indices]
        seq_x, seq_y = self.generate_sequence(uid_df)
        
        total_len = len(seq_x)
        num_seq = (total_len - input_seq_length - predict_seq_length + 1) // look_back_len
        
        if num_seq <= 0:
            return None
            
        predict_user = idx >= (total_users - 3000)
        index_maps = []
        
        for i in range(num_seq):
            start = i * look_back_len
            index_maps.append((0, start))  # Will be adjusted later
            
            if predict_user:
                for _ in range(1, multiple):
                    index_maps.append((0, start))  # Will be adjusted later
                    
        return (seq_x, seq_y), index_maps

    # -----------------------------------------------------
    # FAST SEQUENCE GENERATION (NO ITERROWS)
    # -----------------------------------------------------
    def generate_sequence(self, data):
        uid = data['uid'].values[0]

        d = data['d'].values
        t = data['t'].values
        label = data['label'].values

        time_index = t + 48 * d
        delta_t = np.diff(time_index, prepend=time_index[0])

        seq_x = np.stack([
            d,
            t,
            np.full_like(d, uid),
            d % 7,
            t % 24,
            delta_t
        ], axis=1)

        return seq_x.astype(np.int64), label.astype(np.int64)

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        user_idx, start = self.index_map[idx]
        seq_x, seq_y = self.user_data[user_idx]

        x1 = seq_x[start : start + self.input_seq_length]
        y1 = seq_y[start : start + self.input_seq_length]

        x2 = seq_x[start + self.input_seq_length :
                   start + self.input_seq_length + self.predict_seq_length]

        y2 = seq_y[start + self.input_seq_length :
                   start + self.input_seq_length + self.predict_seq_length]

        return (
            torch.from_numpy(x1),
            torch.from_numpy(y1),
            torch.from_numpy(x2),
            torch.from_numpy(y2)
        )

# =========================================================
# TRAIN TEST LOADER
# =========================================================

def train_test_mob_time_series_dataloader(
    rank,
    world_size,
    city,
    input_seq_length,
    predict_seq_length,
    subsample=False,
    subsample_number=100,
    test_size=0.1,
    batch_size=64,
    random_seed=42,
    look_back_len=24):

    if city == 'A':
        data_path = Path.home() / "Downloads" / "city_A_alldata.parquet"
    else:
        data_path = Path.home() / "Downloads" / f"city_{city}_alldata.parquet"

    print("Loading parquet...")
    # For large datasets, use pyarrow engine for better performance
    try:
        dataset = pd.read_parquet(data_path, engine='pyarrow')
    except:
        # Fallback to default engine
        dataset = pd.read_parquet(data_path)
    print(f"Loaded {len(dataset)} rows")

    print("Building dataset...")
    dataset = MobTimeSeriesDataset(
        dataset,
        input_seq_length,
        predict_seq_length,
        subsample=subsample,
        subsample_number=subsample_number,
        look_back_len=look_back_len
    )

    dataset_size = len(dataset)
    train_size = int(dataset_size * (1 - test_size))
    test_size = dataset_size - train_size

    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset,
        [train_size, test_size],
        generator=torch.Generator().manual_seed(random_seed)
    )

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank, shuffle=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        drop_last=True,
        num_workers=min(8, mp.cpu_count()),
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        sampler=test_sampler,
        drop_last=False,
        num_workers=min(8, mp.cpu_count()),
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True
    )

    return train_loader, test_loader

# =========================================================
# UID SPLIT (UNCHANGED)
# =========================================================

def split_df_by_uid(city, test_size=0.1, random_seed=None,
                   subsample=False, subsample_number=100):

    if city == 'A':
        data_path = Path.home() / "Downloads" / "city_A_alldata.parquet"
    else:
        data_path = Path.home() / "Downloads" / f"city_{city}_alldata.parquet"

    df = pd.read_parquet(data_path)

    if subsample:
        uids = df['uid'].unique()[:subsample_number]
        df = df[df['uid'].isin(uids)]

    uids = df['uid'].unique()

    generate_uid_list = uids[-3000:]
    generate_df = df[df['uid'].isin(generate_uid_list)]

    remain_df = df[df['x'] != 999]

    selected_uids = uids[:-3000]

    _, test_uids = train_test_split(
        selected_uids,
        test_size=test_size,
        random_state=random_seed
    )

    test_df = remain_df[remain_df['uid'].isin(test_uids)]
    train_df = remain_df[~remain_df['uid'].isin(test_uids)]

    partial_generate_df = generate_df[generate_df['d'] < 60]
    partial_test_df = remain_df[remain_df['d'] < 60]

    train_df = pd.concat([train_df, partial_test_df, partial_generate_df])

    return train_df, test_df, generate_df

# =========================================================
# GENERATE MODE LOADER (RESTORED)
# =========================================================

def train_test_generate_mob_time_series_dataloader(
    city,
    input_seq_length,
    predict_seq_length,
    subsample=False,
    subsample_number=100,
    test_size=0.1,
    batch_size=64,
    random_seed=42,
    look_back_len=24,
    world_size=None,
    rank=None,
    multiple=2):

    train_df, test_df, generate_df = split_df_by_uid(
        city,
        test_size=test_size,
        random_seed=random_seed,
        subsample=subsample,
        subsample_number=subsample_number
    )

    dataset = MobTimeSeriesDataset(
        train_df,
        input_seq_length,
        predict_seq_length,
        look_back_len=look_back_len,
        multiple=multiple
    )

    if world_size is not None and rank is not None:
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)

        train_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            drop_last=True,
            num_workers=4,
            pin_memory=True
        )
    else:
        train_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=4,
            pin_memory=True
        )

    return train_loader, test_df, generate_df