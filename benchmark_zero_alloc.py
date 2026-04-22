import time
import torch
import torch.nn.functional as F
from geoformer.model import build_model
from geoformer.data import TOTAL_VOCAB, VOCAB_LOC_SIZE

device = torch.device('mps')
model = build_model('small').to(device)
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
            logits, _, _ = model(tok_t, tod_t, dow_t, key_padding_mask=key_pad, return_last_logit_only=True)
            k_vals, _ = torch.topk(logits[:, -1, :], 5, dim=-1)
    torch.mps.synchronize()
    start = time.time()
    
    # PRE-ALLOCATION
    next_logits = torch.zeros((B, TOTAL_VOCAB), dtype=torch.float32, device=device)
    fallback_probs = torch.zeros((B, TOTAL_VOCAB), dtype=torch.float32, device=device)
    fallback_probs[:, :VOCAB_LOC_SIZE] = 1.0 / VOCAB_LOC_SIZE
    inf_mask = torch.zeros((B, VOCAB_LOC_SIZE), dtype=torch.bool, device=device)
    
    with torch.inference_mode(), torch.autocast('mps', dtype=torch.float16):
        for _ in range(50):
            logits, _, _ = model(tok_t, tod_t, dow_t, key_padding_mask=key_pad, return_last_logit_only=True)
            # ZERO ALLOCATION inner operations
            next_logits.copy_(logits[:, -1, :].float())
            next_logits[:, :VOCAB_LOC_SIZE].masked_fill_(inf_mask, float("-inf"))
            k_vals, _ = torch.topk(next_logits, 5, dim=-1)
            kth_val = k_vals[:, -1].unsqueeze(1)
            next_logits.masked_fill_(next_logits < kth_val, float("-inf"))
            
            probs = F.softmax(next_logits, dim=-1)
            probs = torch.nan_to_num_(probs, nan=0.0)
            
            empty_mask = (probs.sum(dim=-1, keepdim=True) == 0)
            probs = torch.where(empty_mask, fallback_probs, probs)
            next_toks = torch.multinomial(probs, num_samples=1).squeeze(-1)
            
    torch.mps.synchronize()
    print(f"Zero-Allocation FP16 Batch {B} - Time for 50 steps: {time.time() - start:.2f}s")
except Exception as e:
    print("OOM or error:", e)
