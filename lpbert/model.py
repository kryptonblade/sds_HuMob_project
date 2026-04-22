"""
lpbert/model.py
===============
LP-BERT: BERT-style encoder for human mobility next-location prediction.

Architecture (Terashima et al., 2023):
  Input per record: (location_id, date, time, timedelta) → each embedded to
  d_model dimensions and SUMMED (Figure 3 in paper).

  Encoder: N bidirectional TransformerEncoder layers (no causal mask).
  Head: Linear(d_model → loc_vocab_size) predicting masked location tokens.

Paper defaults: d_model=128, n_layers=4, n_heads=8, batch_size=16, epochs=200.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from lpbert.data import (
    TOTAL_LOC_VOCAB, MASK_TOKEN,
    MAX_DATE, SLOTS_PER_DAY, TIMEDELTA_BUCKETS,
)


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class LPBertConfig:
    # Model size (paper default: 128-dim, 4-layer, 8-head)
    d_model:   int   = 128
    n_layers:  int   = 4
    n_heads:   int   = 8
    d_ff:      int   = 512
    dropout:   float = 0.1

    # Vocab sizes for each input feature
    loc_vocab_size:    int = TOTAL_LOC_VOCAB         # 40001  (0..39999 locs + MASK)
    date_vocab_size:   int = MAX_DATE                # 76     (days 0..75)
    time_vocab_size:   int = SLOTS_PER_DAY + 1       # 49     (slots 0..48)
    timedelta_buckets: int = TIMEDELTA_BUCKETS        # 721    (0..720)

    # Sequence cap
    max_seq_len: int = 2048

    @classmethod
    def paper(cls):
        """Exact paper configuration: 128-dim, 4-layer, 8-head."""
        return cls(d_model=128, n_layers=4, n_heads=8, d_ff=512)

    @classmethod
    def small(cls):
        """Smaller/faster variant for development."""
        return cls(d_model=64, n_layers=2, n_heads=4, d_ff=256)

    @classmethod
    def medium(cls):
        """Larger variant for higher accuracy."""
        return cls(d_model=256, n_layers=6, n_heads=8, d_ff=1024)


# ─────────────────────────────────────────────────────────────────────────────
# LP-BERT model
# ─────────────────────────────────────────────────────────────────────────────
class LPBert(nn.Module):
    """
    Encoder-only BERT for masked location prediction.

    Forward inputs (all [B, T]):
      locs       — location token IDs (masked positions = MASK_TOKEN)
      days       — 1-indexed day numbers
      times      — 1-indexed time slots (1..48)
      timedeltas — time gap in 30-min slots since previous record

    Returns:
      logits [B, T, loc_vocab_size]  — raw scores for each location
      loss   scalar (only when labels provided, -100 positions ignored)
    """

    def __init__(self, cfg: LPBertConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model

        # ── Four input embeddings (Figure 3 in paper) ──────────────────────
        self.loc_embed      = nn.Embedding(cfg.loc_vocab_size,    d)
        self.date_embed     = nn.Embedding(cfg.date_vocab_size,   d)
        self.time_embed     = nn.Embedding(cfg.time_vocab_size,   d)
        self.timedelta_embed = nn.Embedding(cfg.timedelta_buckets, d)

        self.embed_ln   = nn.LayerNorm(d)
        self.embed_drop = nn.Dropout(cfg.dropout)

        # ── Bidirectional Transformer encoder (no causal mask) ─────────────
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.d_ff,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,    # Pre-LayerNorm (modern BERT variant)
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=cfg.n_layers, enable_nested_tensor=False
        )

        # ── MLM prediction head ────────────────────────────────────────────
        self.pred_head = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, cfg.loc_vocab_size),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        locs:             torch.Tensor,                    # [B, T]
        days:             torch.Tensor,                    # [B, T]
        times:            torch.Tensor,                    # [B, T]
        timedeltas:       torch.Tensor,                    # [B, T]
        key_padding_mask: Optional[torch.Tensor] = None,   # [B, T] True=padding
        labels:           Optional[torch.Tensor] = None,   # [B, T] -100=ignore
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:

        cfg = self.cfg

        # Clamp indices into valid embedding ranges
        safe_days   = days.clamp(0, cfg.date_vocab_size - 1)
        safe_times  = times.clamp(0, cfg.time_vocab_size - 1)
        safe_deltas = timedeltas.clamp(0, cfg.timedelta_buckets - 1)

        # Sum four embeddings (LP-BERT input design, paper §4)
        x = (
            self.loc_embed(locs)
            + self.date_embed(safe_days)
            + self.time_embed(safe_times)
            + self.timedelta_embed(safe_deltas)
        )
        x = self.embed_drop(self.embed_ln(x))

        # Bidirectional encoder — no causal mask
        x = self.encoder(x, src_key_padding_mask=key_padding_mask)

        # Location logits for every position
        logits = self.pred_head(x)   # [B, T, loc_vocab_size]

        loss = None
        if labels is not None:
            mask = labels != -100          # [B, T] — only masked positions
            if mask.any():
                loss = F.cross_entropy(
                    logits[mask],          # [n_masked, vocab_size]
                    labels[mask],          # [n_masked]
                )

        return logits, loss

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────
def build_lpbert(size: str = "paper") -> LPBert:
    cfg_map = {
        "small":  LPBertConfig.small(),
        "paper":  LPBertConfig.paper(),
        "medium": LPBertConfig.medium(),
    }
    if size not in cfg_map:
        raise ValueError(f"Unknown size '{size}'. Choose from: {list(cfg_map)}")
    cfg   = cfg_map[size]
    model = LPBert(cfg)
    print(f"[model] LP-BERT ({size}): {model.num_parameters():,} parameters  "
          f"d={cfg.d_model}, layers={cfg.n_layers}, heads={cfg.n_heads}")
    return model
