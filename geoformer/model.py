"""
geoformer/model.py
==================
GeoFormer: decoder-only GPT-style transformer for human mobility prediction.

Architecture (following the paper, with configurable scale):
  - Embedding layer: location token + time-of-day + day-of-week + positional
  - N × causal self-attention blocks (TransformerEncoderLayer with causal mask)
  - LM head: linear projection → vocabulary logits

Token vocabulary:
  0..39999   = location tokens (x*200 + y)
  40000      = PAD
  40001      = BOS
  40002      = EOS
  Total      = 40003
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from geoformer.data import (
    TOTAL_VOCAB, PAD_TOKEN, SLOTS_PER_DAY, DOW_COUNT, TOD_COUNT
)


# ─────────────────────────────────────────────────────────────────────────────
# Config dataclass
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class GeoFormerConfig:
    # Vocabulary
    vocab_size:   int = TOTAL_VOCAB       # 40003
    pad_token_id: int = PAD_TOKEN         # 40000

    # Model dimensions
    d_model:      int = 256               # embedding dimension
    n_heads:      int = 8                 # attention heads
    n_layers:     int = 4                 # transformer blocks
    d_ff:         int = 1024              # feedforward hidden dim
    dropout:      float = 0.1

    # Sequence length
    max_seq_len:  int = 384               # CONTEXT_DAYS * SLOTS_PER_DAY

    # Context feature sizes
    tod_vocab:    int = TOD_COUNT         # 48
    dow_vocab:    int = DOW_COUNT         # 7
    city_vocab:   int = 4                 # 4 cities

    # City embedding (optional, included in input sum)
    use_city_embed: bool = True

    @classmethod
    def tiny(cls):
        """Micro config for extreme fast smoke testing."""
        return cls(d_model=64, n_heads=2, n_layers=1, d_ff=256, max_seq_len=192)

    @classmethod
    def small(cls):
        """CPU/MPS-friendly config for development / smoke tests."""
        return cls(d_model=128, n_heads=4, n_layers=2, d_ff=512, max_seq_len=192)

    @classmethod
    def medium(cls):
        """Balanced config: good quality, trainable on MPS/GPU."""
        return cls(d_model=256, n_heads=8, n_layers=4, d_ff=1024, max_seq_len=256)

    @classmethod
    def full(cls):
        """Paper config: 12-layer GPT, needs A100-class GPU for city A."""
        return cls(d_model=768, n_heads=12, n_layers=12, d_ff=3072)


# ─────────────────────────────────────────────────────────────────────────────
# Causal mask utility
# ─────────────────────────────────────────────────────────────────────────────
def make_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """
    Upper-triangular mask of shape [seq_len, seq_len].
    True positions are MASKED (ignored) in nn.MultiheadAttention.
    """
    return torch.triu(
        torch.ones(seq_len, seq_len, dtype=torch.bool, device=device), diagonal=1
    )


# ─────────────────────────────────────────────────────────────────────────────
# Single Transformer block
# ─────────────────────────────────────────────────────────────────────────────
class TransformerBlock(nn.Module):
    """
    Pre-LayerNorm GPT block:
      x = x + Attn(LayerNorm(x))
      x = x + FFN(LayerNorm(x))
    """

    def __init__(self, cfg: GeoFormerConfig):
        super().__init__()
        self.cfg = cfg
        self.ln1  = nn.LayerNorm(cfg.d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=cfg.d_model,
            num_heads=cfg.n_heads,
            dropout=cfg.dropout,
            batch_first=True,
        )
        self.ln2  = nn.LayerNorm(cfg.d_model)
        self.ff   = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_ff),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_ff, cfg.d_model),
            nn.Dropout(cfg.dropout),
        )

    def forward(self,
                x: torch.Tensor,
                causal_mask: Optional[torch.Tensor] = None,
                key_padding_mask: Optional[torch.Tensor] = None,
                past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        
        normed = self.ln1(x)
        B, T, C = normed.shape
        
        # ── Fast KV-Caching Override ──
        # Instead of calling self.attn(...), manually project Q, K, V
        # using the exact identical checkpoint parameter weights natively.
        qkv = F.linear(normed, self.attn.in_proj_weight, self.attn.in_proj_bias)
        q, k, v = qkv.chunk(3, dim=-1)
        
        n_heads = self.cfg.n_heads
        d_head  = C // n_heads
        
        # [B, n_heads, T, d_head]
        q = q.view(B, T, n_heads, d_head).transpose(1, 2)
        k = k.view(B, T, n_heads, d_head).transpose(1, 2)
        v = v.view(B, T, n_heads, d_head).transpose(1, 2)
        
        if past_kv is not None:
            pk, pv = past_kv
            k = torch.cat([pk, k], dim=2)
            v = torch.cat([pv, v], dim=2)
            
        present_kv = (k, v)
        
        # Self attention computation smoothly using F.scaled_dot_product_attention
        y = F.scaled_dot_product_attention(
            q, k, v, 
            is_causal=(past_kv is None and T > 1)
        )
        
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        attn_out = F.linear(y, self.attn.out_proj.weight, self.attn.out_proj.bias)

        x = x + attn_out
        x = x + self.ff(self.ln2(x))
        return x, present_kv


# ─────────────────────────────────────────────────────────────────────────────
# GeoFormer model
# ─────────────────────────────────────────────────────────────────────────────
class GeoFormer(nn.Module):
    """
    Decoder-only transformer for human mobility sequence modeling.

    Input per time step: (location_token, time_of_day, day_of_week, [city_id])
    Each is embedded independently and summed → d_model vector.
    A learned positional embedding is also summed in.
    The result passes through N causal transformer blocks.
    A linear head maps to vocabulary logits.
    """

    def __init__(self, cfg: GeoFormerConfig):
        super().__init__()
        self.cfg = cfg

        # ── Embeddings ──
        self.loc_embed  = nn.Embedding(cfg.vocab_size, cfg.d_model,
                                       padding_idx=cfg.pad_token_id)
        self.tod_embed  = nn.Embedding(cfg.tod_vocab,  cfg.d_model)
        self.dow_embed  = nn.Embedding(cfg.dow_vocab,  cfg.d_model)
        self.pos_embed  = nn.Embedding(cfg.max_seq_len, cfg.d_model)

        if cfg.use_city_embed:
            self.city_embed = nn.Embedding(cfg.city_vocab, cfg.d_model)
        else:
            self.city_embed = None

        self.emb_drop = nn.Dropout(cfg.dropout)

        # ── Transformer blocks ──
        self.blocks = nn.ModuleList([
            TransformerBlock(cfg) for _ in range(cfg.n_layers)
        ])

        # ── Output head ──
        self.ln_f  = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        # Weight tying: share location embedding & lm_head weights
        # (only for the location portion of vocab for efficiency)
        self.lm_head.weight = self.loc_embed.weight

        self._init_weights()

    def _init_weights(self):
        """Initialize weights following GPT-2 convention."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()

    def forward(
        self,
        tokens:          torch.Tensor,          # [B, T]
        tod:             torch.Tensor,          # [B, T]
        dow:             torch.Tensor,          # [B, T]
        city_id:         Optional[torch.Tensor] = None,  # [B] or scalar
        key_padding_mask: Optional[torch.Tensor] = None,  # [B, T] True=pad
        labels:          Optional[torch.Tensor] = None,  # [B, T] -100=ignore
        past_kvs:        Optional[list] = None,          # list of tuples
        start_pos:       int = 0,                        # sequence positional offset
        return_last_logit_only: bool = False
    ):
        B, T = tokens.shape
        device = tokens.device
        
        # ── Positional indices (offset dynamically for KV caches) ──
        positions = torch.arange(start_pos, start_pos + T, device=device).unsqueeze(0).expand(B, T)

        # ── Sum embeddings ──
        x  = self.loc_embed(tokens)
        x += self.tod_embed(tod)
        x += self.dow_embed(dow)
        x += self.pos_embed(positions)

        if self.city_embed is not None and city_id is not None:
            if city_id.dim() == 1:
                city_id = city_id.unsqueeze(1).expand(B, T)
            x += self.city_embed(city_id)

        x = self.emb_drop(x)

        # ── Causal mask ──
        causal_mask = None
        if past_kvs is None and T > 1:
            causal_mask = make_causal_mask(T, device)

        # ── Transformer blocks ──
        present_kvs = []
        for i, block in enumerate(self.blocks):
            past_kv = past_kvs[i] if past_kvs is not None else None
            x, p_kv = block(x, causal_mask=causal_mask, key_padding_mask=key_padding_mask, past_kv=past_kv)
            present_kvs.append(p_kv)

        # ── Head ──
        x = self.ln_f(x)
        if return_last_logit_only:
            logits = self.lm_head(x[:, -1:, :])     # [B, 1, vocab_size]
        else:
            logits = self.lm_head(x)                # [B, T, vocab_size]


        # ── Loss ──
        loss = None
        if labels is not None:
            # Shift: predict token[i+1] from position[i]
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.cfg.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        return logits, loss, present_kvs

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─────────────────────────────────────────────────────────────────────────────
# Convenience factory
# ─────────────────────────────────────────────────────────────────────────────
def build_model(size: str = "medium") -> GeoFormer:
    """
    Build a GeoFormer model from a named size preset.

    Args:
        size: one of "small", "medium", "full"
    """
    cfg_map = {
        "tiny":   GeoFormerConfig.tiny(),
        "small":  GeoFormerConfig.small(),
        "medium": GeoFormerConfig.medium(),
        "full":   GeoFormerConfig.full(),
    }
    if size not in cfg_map:
        raise ValueError(f"Unknown size '{size}'. Choose from: {list(cfg_map)}")
    cfg = cfg_map[size]
    model = GeoFormer(cfg)
    print(f"[model] GeoFormer ({size}): {model.num_parameters():,} parameters")
    print(f"        d_model={cfg.d_model}, n_heads={cfg.n_heads}, "
          f"n_layers={cfg.n_layers}, d_ff={cfg.d_ff}")
    return model
