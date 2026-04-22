import torch
import time
from geoformer.model import build_model
device = torch.device('mps')
model = build_model('small').to(device)

def test_batch(B):
    T = 192
    tok_t = torch.randint(0, 40000, (B, T), device=device)
    tod_t = torch.randint(0, 48, (B, T), device=device)
    dow_t = torch.randint(0, 7, (B, T), device=device)
    labels = torch.randint(0, 40000, (B, T), device=device)

    try:
        with torch.autocast('mps', dtype=torch.float16):
            start = time.time()
            for _ in range(10):
                _, loss, _ = model(tok_t, tod_t, dow_t, labels=labels)
                loss.backward()
            torch.mps.synchronize()
            print(f"Batch {B}: {(time.time() - start)/10:.3f}s per step, {B/((time.time() - start)/10):.1f} samples/sec")
    except Exception as e:
        print(f"Batch {B} failed: {e}")

for b in [8, 16, 32, 64, 128]:
    test_batch(b)
