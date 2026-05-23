"""
models/fusion.py
─────────────────────────────────────────────────────────────────────────────
Fusion mechanisms for combining image and text features.

  ConcatFusion    — Model 2: simple concatenation → linear projection
  CrossAttnFusion — Model 3: image patches as queries, text as key/value

Both output a tensor of shape (B, fusion_dim) that is then injected into
the U-Net decoder via FiLM conditioning.
─────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ─────────────────────────────────────────────────────────────────────────────
# Model 2 — Concatenation Fusion
# ─────────────────────────────────────────────────────────────────────────────

class ConcatFusion(nn.Module):
    """
    Concatenate global image embedding + text embedding and project to
    fusion_dim.

        [img_feat | txt_feat] ∈ ℝ^(img_dim + txt_dim)
        → Linear → LN → GELU → Linear → LN → ℝ^(fusion_dim)

    Parameters
    ----------
    img_dim    : int  Dimension of projected CLIP global feature.
    txt_dim    : int  Dimension of projected text feature.
    fusion_dim : int  Output dimension.
    dropout    : float
    """

    def __init__(
        self,
        img_dim: int = 256,
        txt_dim: int = 256,
        fusion_dim: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        in_dim = img_dim + txt_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, fusion_dim * 2),
            nn.LayerNorm(fusion_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.LayerNorm(fusion_dim),
        )

    def forward(
        self, img_feat: torch.Tensor, txt_feat: torch.Tensor
    ) -> torch.Tensor:
        """
        img_feat : (B, img_dim)
        txt_feat : (B, txt_dim)
        → (B, fusion_dim)
        """
        combined = torch.cat([img_feat, txt_feat], dim=-1)
        return self.net(combined)


# ─────────────────────────────────────────────────────────────────────────────
# Model 3 — Cross-Attention Fusion
# ─────────────────────────────────────────────────────────────────────────────

class CrossAttentionBlock(nn.Module):
    """
    Single cross-attention block.

    Query  (Q): image patch tokens  (B, N_patches, d_model)
    Key    (K): BERT token sequence (B, seq_len,   d_model)
    Value  (V): BERT token sequence (B, seq_len,   d_model)

    Each image patch independently attends over ALL text tokens, allowing
    different patches to pick up on different clinical keywords
    (e.g., "edema", "enhancement", "frontal").

    Attention weights are stored in self.last_attn_weights:
      shape (B, N_heads, N_patches, seq_len)  — per-head, not averaged.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.n_heads = n_heads
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads,
            dropout=dropout, batch_first=True,
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff    = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
        )
        # Populated during every forward pass for visualization
        self.last_attn_weights: Optional[torch.Tensor] = None

    def forward(
        self,
        q: torch.Tensor,   # (B, N_q, d_model)  — image patches
        kv: torch.Tensor,  # (B, N_kv, d_model) — text tokens
    ) -> torch.Tensor:
        # Cross-attention — store per-head weights for explainability
        attn_out, attn_weights = self.attn(
            query=q, key=kv, value=kv,
            need_weights=True,
            average_attn_weights=False,   # preserve (B, n_heads, N_q, N_kv)
        )
        # Store detached weights so they survive no_grad() contexts
        self.last_attn_weights = attn_weights.detach()  # (B, H, N_q, N_kv)
        q = self.norm1(q + attn_out)
        # Feed-forward
        q = self.norm2(q + self.ff(q))
        return q


class CrossAttnFusion(nn.Module):
    """
    Multi-layer cross-attention between image patches and BERT token sequence.

    TRUE token-wise cross-attention design:
      Q = image patch tokens  (B, N_patches, fusion_dim)
      K = BERT token sequence (B, seq_len,   fusion_dim)    ← FIXED
      V = BERT token sequence (B, seq_len,   fusion_dim)    ← FIXED

    Each image patch can now attend to DIFFERENT words in the radiology
    report, producing genuine text-guided spatial localization.

    Attention weights from the LAST block are accessible via
    self.blocks[-1].last_attn_weights  →  (B, n_heads, N_patches, seq_len)

    Parameters
    ----------
    img_patch_dim : int   CLIP ViT-B/32 patch token dim (768).
    txt_token_dim : int   BERT last_hidden_state dim (768).
    fusion_dim    : int   Internal attention dim (256).
    n_heads       : int   Attention heads (8).
    n_layers      : int   Cross-attention blocks (2).
    dropout       : float
    fusion_mode   : str   'global' | 'spatial'
    """

    def __init__(
        self,
        img_patch_dim: int = 768,
        txt_token_dim: int = 768,     # BERT last_hidden_state dim (NOT pooled)
        fusion_dim: int = 256,
        n_heads: int = 8,
        n_layers: int = 2,
        dropout: float = 0.1,
        fusion_mode: str = "global",
    ) -> None:
        super().__init__()
        self.fusion_mode = fusion_mode
        d = fusion_dim

        # Project image patches to fusion_dim
        self.img_proj = nn.Linear(img_patch_dim, d)
        # Project FULL text token sequence to fusion_dim (replaces single-vector unsqueeze)
        self.txt_proj = nn.Linear(txt_token_dim, d)

        self.blocks = nn.ModuleList([
            CrossAttentionBlock(d, n_heads, dropout)
            for _ in range(n_layers)
        ])

        self.out_norm = nn.LayerNorm(d)

    def get_last_attn_weights(self) -> Optional[torch.Tensor]:
        """Return per-head attention weights from the last cross-attn block.

        Returns
        -------
        attn_weights : (B, n_heads, N_patches, seq_len)
        """
        return self.blocks[-1].last_attn_weights if self.blocks else None

    def compute_attn_entropy_loss(self, target_entropy: float = 1.5) -> torch.Tensor:
        """
        Attention entropy regularization loss.

        Penalizes attention distributions that are too uniform (high entropy),
        encouraging the model to focus on specific text tokens per patch.

        target_entropy : nats of entropy to target (lower = more focused).
        """
        attn = self.get_last_attn_weights()   # (B, H, N_patches, seq_len)
        if attn is None:
            return torch.tensor(0.0)          # device-agnostic: no loss to add
        # Normalize across seq_len dimension
        p = attn / (attn.sum(dim=-1, keepdim=True) + 1e-10)
        entropy = -(p * torch.log(p + 1e-10)).sum(dim=-1)  # (B, H, N_patches)
        # Penalize entropy above target
        excess = torch.clamp(entropy - target_entropy, min=0.0)
        return excess.mean()                  # result is on same device as attn

    def forward(
        self,
        img_patches: torch.Tensor,   # (B, N, img_patch_dim)
        txt_tokens: torch.Tensor,    # (B, seq_len, txt_token_dim)  ← FULL sequence
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Parameters
        ----------
        img_patches : (B, N_patches, img_patch_dim)  — CLIP image patches
        txt_tokens  : (B, seq_len,   txt_token_dim)  — BERT last_hidden_state

        Returns
        -------
        fused         : (B, fusion_dim)   in global mode
        spatial_tokens: (B, N, fusion_dim)  always returned for viz
        """
        # Project both to shared fusion_dim
        q  = self.img_proj(img_patches)   # (B, N, d)       Q: image patches
        kv = self.txt_proj(txt_tokens)    # (B, seq_len, d) K/V: text tokens

        # Stack cross-attention blocks
        for block in self.blocks:
            q = block(q, kv)              # (B, N, d)

        spatial_tokens = self.out_norm(q) # (B, N, d)

        if self.fusion_mode == "spatial":
            return spatial_tokens, spatial_tokens
        else:
            fused = spatial_tokens.mean(dim=1)  # (B, d)  global avg-pool
            return fused, spatial_tokens

