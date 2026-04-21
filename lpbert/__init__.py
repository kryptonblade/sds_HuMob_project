"""
LP-BERT: Location Prediction BERT for Human Mobility.

BERT-style encoder with continuous masking for parallel next-location prediction.
Reference: Terashima et al. (2023), HuMob-Challenge '23, Hamburg, Germany.

Architecture:
  - Input: (location_id, date, time, timedelta) per observed record, all summed
  - Encoder: bidirectional TransformerEncoder (no causal mask)
  - Training: randomly mask α=15 consecutive days of location IDs (MLM)
  - Inference: predict all masked positions in parallel (not autoregressive)
"""
