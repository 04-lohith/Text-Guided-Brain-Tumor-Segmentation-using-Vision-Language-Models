"""
models/text_encoder.py
─────────────────────────────────────────────────────────────────────────────
Text encoding utilities.

Two pathways are supported:
 1. Precomputed embeddings  → identity pass-through (already (768,))
 2. Raw text               → BioClinicalBERT → mean-pool → (768,)

The TextProjectionHead further projects (768,) → (fusion_dim,) with
LayerNorm + GELU, used by Model 2 and Model 3.
─────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations
import torch
import torch.nn as nn


class TextProjectionHead(nn.Module):
    """
    Project a pooled BERT embedding down to fusion_dim.

        (B, text_dim) → Linear → LN → GELU → Linear → LN  → (B, out_dim)

    Parameters
    ----------
    text_dim : int   Input dimension (768 for BioClinicalBERT).
    out_dim  : int   Output dimension (fusion_dim from config).
    dropout  : float Dropout probability.
    """

    def __init__(self, text_dim: int = 768, out_dim: int = 256,
                 dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(text_dim, out_dim * 2),
            nn.LayerNorm(out_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim * 2, out_dim),
            nn.LayerNorm(out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : (B, text_dim) → (B, out_dim)."""
        return self.net(x)
