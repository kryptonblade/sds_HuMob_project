import pandas as pd
import numpy as np
import time
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from geobleu import geobleu
from pathlib import Path


pd.options.mode.chained_assignment = None


def morton_encode(x, y, bits=8):
    """Encode 0-indexed grid coordinates to Morton / Z-order."""
    x = np.asarray(x, dtype=np.int64)
    y = np.asarray(y, dtype=np.int64)
    z = np.zeros_like(x, dtype=np.int64)
    for i in range(bits):
        z |= ((x >> i) & 1) << (2 * i)
        z |= ((y >> i) & 1) << (2 * i + 1)
    return z


def morton_decode(z, bits=8):
    """Decode Morton / Z-order values back to 0-indexed (x, y)."""
    z = np.asarray(z, dtype=np.int64)
    x = np.zeros_like(z, dtype=np.int64)
    y = np.zeros_like(z, dtype=np.int64)
    for i in range(bits):
        x |= ((z >> (2 * i)) & 1) << i
        y |= ((z >> (2 * i + 1)) & 1) << i
    return x, y


def convert_label_back(df):
    '''
    Convert Morton label back to x, y
    '''
    x, y = morton_decode(df['label'].to_numpy(dtype=np.int64))
    df['predict_x'] = x + 1
    df['predict_y'] = y + 1
    return df

def get_time_str():
    return time.strftime('%Y-%m-%d_%H:%M:%S', time.localtime(time.time()))

def load_data(city='A'):
    '''
    Load data from parquet file
    '''    
    if city == 'A':
        data_path = Path.home() / "Downloads" / "city_A_alldata.parquet"
    else:
        data_path = Path.home() / "Downloads" / f"city_{city}_alldata.parquet"
    
    df = pd.read_parquet(data_path)
    
    users = sorted(list(df['uid'].unique()))
    predict_users = users[-3000:]
    train_df = df[~df['uid'].isin(predict_users)]
    predict_df = df[df['uid'].isin(predict_users)]
    return train_df, predict_df

def calc_bleu_dtw_loss(generated, target):
    '''
    Calculate BLEU and DTW loss
    tuple format: (uid, d, t, x, y) or (d, t, x, y)
    '''
    assert len(generated) == len(target)
    geo_bleu = geobleu.calc_geobleu(generated, target, processes=3)
    dtw = geobleu.calc_dtw(generated, target, processes=3)
    accuracy = sum([1 for i in range(len(generated)) if generated[i] == target[i]]) / len(generated)
    return geo_bleu, dtw, accuracy
