import time
import torch
from geoformer.model import build_model

device = torch.device('mps')
model = build_model('small').to(device)
model.half() # convert to float16
model.eval()

B = 4000
T = 48
tok_t = torch.randint(0, 40000, (B, T), device=device)
tod_t = torch.randint(0, 48, (B, T), device=device)
dow_t = torch.randint(0, 7, (B, T), device=device)
key_pad = torch.zeros((B, T), dtype=torch.bool, device=device)

try:
    with torch.inference_mode():
        for _ in range(3):
            _ = model(tok_t, tod_t, dow_t, key_padding_mask=key_pad, return_last_logit_only=True)
    torch.mps.synchronize()
    start = time.time()
    with torch.inference_mode():
        for _ in range(50):
            _ = model(tok_t, tod_t, dow_t, key_padding_mask=key_pad, return_last_logit_only=True)
    torch.mps.synchronize()
    print(f"FP16 Batch size {B} - Time for 50 steps: {time.time() - start:.2f}s")
except Exception as e:
    print("OOM or error:", e)
