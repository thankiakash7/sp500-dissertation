"""
Deep learning model architectures: LSTM/GRU sequence regressor and a
lightweight Temporal Fusion Transformer.

Previously these were defined three times (step2_deep.py, step3_tft.py, and
src/evaluation/walk_forward_dl.py had its own copy of the LSTM/GRU class) —
three copies that had to be kept in sync by hand. Factored out here so
step2, step3, and the walk-forward engine all share one definition, and a
future architecture change only has to happen once.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class SequenceRegressor(nn.Module):
    """Single-feature LSTM/GRU regressor: recurrent layer -> dropout -> linear."""

    def __init__(self, cell_type: str, hidden_size: int = 64, num_layers: int = 1, dropout: float = 0.2):
        super().__init__()
        cell_type = cell_type.upper()
        recurrent_class = {"LSTM": nn.LSTM, "GRU": nn.GRU}.get(cell_type)
        if recurrent_class is None:
            raise ValueError("cell_type must be 'LSTM' or 'GRU'.")
        recurrent_dropout = dropout if num_layers > 1 else 0.0
        self.recurrent = recurrent_class(
            input_size=1, hidden_size=hidden_size, num_layers=num_layers,
            batch_first=True, dropout=recurrent_dropout,
        )
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(hidden_size, 1)

    def forward(self, x):
        sequence_output, _ = self.recurrent(x)
        last_output = sequence_output[:, -1, :]
        return self.output(self.dropout(last_output)).squeeze(-1)


class TFTLite(nn.Module):
    """
    Simplified TFT: positional encoding + gated residual network + multi-head
    self-attention over the encoder, then a GRN decoder on the last token ->
    point forecast. Matches the dissertation spec without the full
    quantile/covariate machinery, which adds no value for a single-feature
    daily-return point forecast and is far too slow on CPU.

    seq_len must be fixed at construction time (needed for the positional
    encoding buffer), so a new instance is required per LOOKBACK value.
    """

    def __init__(self, seq_len: int, d_model: int = 32, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Linear(1, d_model)

        pe = torch.zeros(seq_len, d_model)
        pos = torch.arange(seq_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, seq, d)

        self.grn_enc = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ELU(), nn.Dropout(dropout),
            nn.Linear(d_model, d_model), nn.Sigmoid(),
        )
        self.ln_enc = nn.LayerNorm(d_model)

        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ln_attn = nn.LayerNorm(d_model)

        self.grn_dec = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ELU(), nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )
        self.ln_dec = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x):                    # x: (B, seq, 1)
        h = self.input_proj(x) + self.pe      # (B, seq, d)
        g = self.grn_enc(h)
        h = self.ln_enc(h * g + h)
        a, _ = self.attn(h, h, h)
        h = self.ln_attn(h + a)
        last = h[:, -1, :]                    # (B, d)
        out = self.grn_dec(last)
        out = self.ln_dec(out + last)
        return self.head(out).squeeze(-1)     # (B,)


def build_model(cell_type: str, seq_len: int, hidden_size: int = 64,
                 tft_d_model: int = 32, tft_n_heads: int = 4, tft_dropout: float = 0.1) -> nn.Module:
    """Factory used by the walk-forward engine so it can stay model-agnostic."""
    cell_type = cell_type.upper()
    if cell_type in ("LSTM", "GRU"):
        return SequenceRegressor(cell_type, hidden_size=hidden_size)
    if cell_type == "TFT":
        return TFTLite(seq_len=seq_len, d_model=tft_d_model, n_heads=tft_n_heads, dropout=tft_dropout)
    raise ValueError(f"cell_type must be 'LSTM', 'GRU', or 'TFT', got {cell_type!r}")
