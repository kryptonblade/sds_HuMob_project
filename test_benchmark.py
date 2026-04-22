import time
import torch
import torch.nn as nn
from geoformer.model import build_model
from torch.optim import AdamW

device = torch.device('mps')
model = build_model('medium').to(device)
model.train()
optimizer = AdamW(model.parameters(), lr=1e-3)

B = 16
T = 256
tok_t = torch.randint(0, 40000, (B, T), device=device)
tod_t = torch.randint(0, 48, (B, T), device=device)
dow_t = torch.randint(0, 7, (B, T), device=device)
labels = torch.randint(0, 40000, (B, T), device=device)

def run_bench(name, iters=10, compiled=False, autocast=False):
    m = torch.compile(model) if compiled else model
    if compiled:
        # warmup
        _, loss, _ = m(tok_t, tod_t, dow_t, labels=labels)
        loss.backward()
        optimizer.zero_grad()
    
    torch.mps.synchronize()
    start = time.time()
    for _ in range(iters):
        optimizer.zero_grad()
        if autocast:
            with torch.autocast('mps', dtype=torch.bfloat16):
                _, loss, _ = m(tok_t, tod_t, dow_t, labels=labels)
        else:
            _, loss, _ = m(tok_t, tod_t, dow_t, labels=labels)
        loss.backward()
        optimizer.step()
    torch.mps.synchronize()
    print(f"[{name}] {iters} steps: {time.time() - start:.2f}s ({((time.time() - start)/iters):.3f}s/it)")

run_bench("Base Eager", 10, False, False)
run_bench("Base Autocast(bf16)", 10, False, True)
run_bench("Compiled", 10, True, False)
run_bench("Compiled + Autocast", 10, True, True)

