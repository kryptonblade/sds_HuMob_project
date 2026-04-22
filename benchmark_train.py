import time
import torch
import torch.nn as nn
from geoformer.model import build_model
from torch.optim import AdamW

device = torch.device('mps')
model = build_model('small').to(device)
model.train()
optimizer = AdamW(model.parameters(), lr=1e-3)

B = 8
T = 192
tok_t = torch.randint(0, 40000, (B, T), device=device)
tod_t = torch.randint(0, 48, (B, T), device=device)
dow_t = torch.randint(0, 7, (B, T), device=device)
labels = torch.randint(0, 40000, (B, T), device=device)

start = time.time()
for step in range(50):
    optimizer.zero_grad()
    _, loss, _ = model(tok_t, tod_t, dow_t, labels=labels)
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

torch.mps.synchronize()
print(f"B={B} Time for 50 steps: {time.time() - start:.2f}s ({((time.time() - start) / 50):.3f}s/it)")


# Benchmark FP16 Autocast and larger batch sizes
B = 64
tok_t = torch.randint(0, 40000, (B, T), device=device)
tod_t = torch.randint(0, 48, (B, T), device=device)
dow_t = torch.randint(0, 7, (B, T), device=device)
labels = torch.randint(0, 40000, (B, T), device=device)

start = time.time()
with torch.autocast('mps', dtype=torch.float16):
    for step in range(10):
        optimizer.zero_grad()
        _, loss, _ = model(tok_t, tod_t, dow_t, labels=labels)
        # In mixed precision, normally we need a grad scaler, but we can just test raw backward throughput
        loss.backward()
        # skip clip_grad_norm_ to see if it's the bottleneck
        optimizer.step()
torch.mps.synchronize()
print(f"B={B} (FP16 no-clip) Time for 10 steps: {time.time() - start:.2f}s ({((time.time() - start) / 10):.3f}s/it)")

