"""
models/model3_cross_attention.py
─────────────────────────────────────────────────────────────────────────────
Model 3 — True Token-Wise Cross-Attention Multimodal Fusion (improved).

Key improvements over v1:
  1. TRUE token-wise cross-attention:
       Q = CLIP image patches
       K = BERT last_hidden_state (full token sequence, NOT pooled)
       V = BERT last_hidden_state
     Each image patch attends to DIFFERENT words → genuine spatial grounding.

  2. Partial CLIP unfreeze:
       The last `unfreeze_clip_blocks` transformer blocks are trainable,
       enabling medical-domain adaptation of CLIP features.

  3. Attention entropy regularization via CrossAttnFusion.compute_attn_entropy_loss()
     added to training loss (controlled by loss.attn_entropy_weight in config).

Pipeline:
    FLAIR slice → CLIP image encoder → patch_tokens (B, 49, 768)
    text_tokens (B, 128, 768) → CrossAttnFusion (Q=patches, K/V=tokens)
    fused_feat → 256-d global vector → FiLM conditioning in U-Net decoder
─────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations
from typing import Dict, Optional
import torch
import torch.nn as nn
from .unet import UNet
from .image_encoder import CLIPImageEncoder
from .fusion import CrossAttnFusion


class CrossAttentionFusionModel(nn.Module):
    """
    Model 3 — True token-wise cross-attention multimodal fusion.

    Parameters
    ----------
    cfg : DotDict  Project configuration.
    """

    MODEL_NAME = "model3_cross_attention"

    def __init__(self, cfg=None) -> None:
        super().__init__()
        fusion_dim         = int(getattr(cfg.model, "fusion_dim",           256)) if cfg else 256
        txt_dim            = int(getattr(cfg.model, "text_embed_dim",       768)) if cfg else 768
        base_ch            = int(getattr(cfg.model, "unet_base_channels",    32)) if cfg else 32
        depth              = int(getattr(cfg.model, "unet_depth",             4)) if cfg else 4
        dropout            = float(getattr(cfg.model, "dropout",            0.1)) if cfg else 0.1
        n_heads            = int(getattr(cfg.model, "cross_attn_heads",       8)) if cfg else 8
        n_layers           = int(getattr(cfg.model, "cross_attn_layers",      2)) if cfg else 2
        freeze_clip        = bool(getattr(cfg.model, "freeze_clip",         True)) if cfg else True
        unfreeze_blocks    = int(getattr(cfg.model, "unfreeze_clip_blocks",    2)) if cfg else 2

        # ── CLIP image encoder ────────────────────────────────────────────
        self.image_encoder = CLIPImageEncoder(
            out_dim=fusion_dim,
            freeze=freeze_clip,
            clip_model="ViT-B/32",
            unfreeze_last_n_blocks=unfreeze_blocks,
        )

        # ── Cross-attention fusion (TRUE token-level K/V) ─────────────────
        # img_patch_dim = 768  (ViT-B/32 internal hidden dim)
        # txt_token_dim = 768  (BERT last_hidden_state, NOT pooled)
        self.cross_fusion = CrossAttnFusion(
            img_patch_dim=768,
            txt_token_dim=txt_dim,      # 768 raw BERT tokens
            fusion_dim=fusion_dim,
            n_heads=n_heads,
            n_layers=n_layers,
            dropout=dropout,
        )

        # ── U-Net with FiLM decoder conditioning ─────────────────────────
        self.unet = UNet(
            in_channels=1,
            out_channels=1,
            base_ch=base_ch,
            depth=depth,
            dropout=dropout,
            text_dim=fusion_dim,
        )

    def forward(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        batch : dict containing:
            'image'       : (B, 1, H, W)
            'text_emb'    : (B, 768)          — pooled, for FiLM
            'text_tokens' : (B, seq_len, 768) — full sequence, for cross-attn K/V

        Returns
        -------
        logits : (B, 1, H, W)
        """
        img         = batch["image"]          # (B, 1, H, W)
        txt_tokens  = batch["text_tokens"]    # (B, seq_len, 768) full sequence

        # 1. Encode image: get CLIP patch tokens (B, 49, 768)
        _, patch_tokens = self.image_encoder(img)   # (B, N, 768)

        # 2. TRUE token-wise cross-attention
        #    Q = image patches (B, N, 768)
        #    K = V = BERT last_hidden_state token sequence (B, seq_len, 768)
        fused, spatial_tokens = self.cross_fusion(patch_tokens, txt_tokens)
        # fused         : (B, 256) — global vector carrying image-grounded text context
        # spatial_tokens: (B, N, 256) — per-patch fused features for visualization

        # Cache for post-inference visualization access
        self._last_spatial_tokens = spatial_tokens

        # 3. Inject fused multimodal vector into U-Net decoder via FiLM
        logits = self.unet(img, text=fused)
        return logits

    # ── Visualization helpers ─────────────────────────────────────────────

    def get_attn_weights(self) -> Optional[torch.Tensor]:
        """Return per-head cross-attention weights from the last forward pass.

        Returns
        -------
        attn_weights : (B, n_heads, N_patches, seq_len)  — tokens not collapsed!
        """
        return self.cross_fusion.get_last_attn_weights()

    def get_spatial_tokens(self) -> Optional[torch.Tensor]:
        """Return the cross-attention output patch tokens from the last forward pass.

        Returns
        -------
        spatial_tokens : (B, N_patches, fusion_dim) or None
        """
        return getattr(self, "_last_spatial_tokens", None)

    def get_attn_entropy_loss(self, target_entropy: float = 1.5) -> torch.Tensor:
        """Compute attention entropy regularization loss.
        Call this AFTER forward() during training to add to total loss.
        """
        return self.cross_fusion.compute_attn_entropy_loss(target_entropy)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
