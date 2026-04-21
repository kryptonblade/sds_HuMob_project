import time
import torch
import torch.nn.functional as F
from geoformer.model import build_model
from geoformer.data import TOTAL_VOCAB, VOCAB_LOC_SIZE

device = torch.device('mps')
model = build_model('small').to(device)
model.eval()

B = 1000
T = 48
tok_t = torch.randint(0, 40000, (B, T), device=device)
tod_t = torch.randint(0, 48, (B, T), device=device)
dow_t = torch.randint(0, 7, (B, T), device=device)
key_pad = torch.zeros((B, T), dtype=torch.bool, device=device)

# warmup
with torch.inference_mode():
    for _ in range(10):
        logits, _, _ = model(tok_t, tod_t, dow_t, key_padding_mask=key_pad)
torch.mps.synchronize()

start = time.time()
with torch.inference_mode():
    for _ in range(720):
        logits, _, _ = model(tok_t, tod_t, dow_t, key_padding_mask=key_pad)
torch.mps.synchronize()
print("Model full forward:", time.time() - start)

next_logits = torch.randn(B, TOTAL_VOCAB, device=device)
start = time.time()
for _ in range(720):
    k_vals, _ = torch.topk(next_logits, 5, dim=-1)
torch.mps.synchronize()
print("TopK:", time.time() - start)

probs = F.softmax(next_logits, dim=-1)
start = time.time()
for _ in range(720):
    next_toks = torch.multinomial(probs, num_samples=1)
torch.mps.synchronize()
print("Multinomial:", time.time() - start)

probs = F.softmax(next_logits, dim=-1)
inf_mask = torch.zeros((B, VOCAB_LOC_SIZE), dtype=torch.bool, device=device)
start = time.time()
for _ in range(720):
    next_logits[:, :VOCAB_LOC_SIZE].masked_fill_(inf_mask, float("-inf"))
    probs.sum(dim=-1, keepdim=True)
    tf = torch.zeros_like(probs)
torch.mps.synchronize()
print("Misc operations:", time.time() - start)

