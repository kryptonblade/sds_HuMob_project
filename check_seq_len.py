import torch
import sys
from geoformer.model import GeoFormerConfig
torch.serialization.add_safe_globals([GeoFormerConfig])
try:
    ckpt = torch.load('checkpoints/geoformer_cityB_small_best.pt', map_location='cpu')
    print("max_seq_len:", ckpt.get('cfg', GeoFormerConfig.small()).max_seq_len)
except Exception as e:
    ckpt = torch.load('checkpoints/geoformer_cityB_small_best.pt', map_location='cpu', weights_only=False)
    print("max_seq_len:", ckpt.get('cfg', GeoFormerConfig.small()).max_seq_len)
