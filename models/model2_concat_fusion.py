"""
models/model2_concat_fusion.py
─────────────────────────────────────────────────────────────────────────────
Model 2 — Concatenation-based multimodal fusion.

Pipeline:
    FLAIR slice → CLIP image encoder → global_feat  (B, img_dim)
    text_emb              → TextProjectionHead      (B, txt_dim)
    ConcatFusion([img, txt])                        (B, fusion_dim)
    fusion_feat → FiLM conditioning in U-Net decoder
─────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations
from typing import Dict
import torch
import torch.nn as nn
from .unet import UNet
from .image_encoder import CLIPImageEncoder
from .text_encoder import TextProjectionHead
from .fusion import ConcatFusion


class ConcatFusionModel(nn.Module):
    """
    Model 2 — Simple concatenation multimodal baseline.

    Parameters
    ----------
    cfg : DotDict   Project configuration.
    """

    MODEL_NAME = "model2_concat_fusion"

    def __init__(self, cfg=None) -> None:
        super().__init__()
        fusion_dim = int(getattr(cfg.model, "fusion_dim",        256)) if cfg else 256
        img_dim    = int(getattr(cfg.model, "image_embed_dim",   512)) if cfg else 512
        txt_dim    = int(getattr(cfg.model, "text_embed_dim",    768)) if cfg else 768
        base_ch    = int(getattr(cfg.model, "unet_base_channels", 32)) if cfg else 32
        depth      = int(getattr(cfg.model, "unet_depth",          4)) if cfg else 4
        dropout    = float(getattr(cfg.model, "dropout",         0.1)) if cfg else 0.1
        freeze     = bool(getattr(cfg.model, "freeze_clip",      True)) if cfg else True
        clip_name  = getattr(cfg.model, "clip_model", "ViT-B/32") if cfg else "ViT-B/32"
        # Normalise clip model name for OpenAI CLIP API
        clip_name  = clip_name.replace("openai/clip-", "").upper().replace("PATCH", "/")
        # e.g. "vit-base-patch32" → "ViT-B/32" is the right format for openai CLIP

        # ── Sub-modules ───────────────────────────────────────────────────
        self.image_encoder = CLIPImageEncoder(
            out_dim=fusion_dim,
            freeze=freeze,
            clip_model="ViT-B/32",
        )

        self.text_proj = TextProjectionHead(
            text_dim=txt_dim,
            out_dim=fusion_dim,
            dropout=dropout,
        )

        self.fusion = ConcatFusion(
            img_dim=fusion_dim,
            txt_dim=fusion_dim,
            fusion_dim=fusion_dim,
            dropout=dropout,
        )

        self.unet = UNet(
            in_channels=1,
            out_channels=1,
            base_ch=base_ch,
            depth=depth,
            dropout=dropout,
            text_dim=fusion_dim,    # FiLM receives fusion_dim-dim vector
        )

    def forward(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        batch : dict with 'image' (B,1,H,W) and 'text_emb' (B,768)

        Returns
        -------
        logits : (B, 1, H, W)
        """
        img      = batch["image"]       # (B, 1, H, W)
        txt_raw  = batch["text_emb"]    # (B, 768)

        # Encode image (global feature only)
        img_feat, _ = self.image_encoder(img)    # (B, fusion_dim)

        # Project text
        txt_feat = self.text_proj(txt_raw)        # (B, fusion_dim)

        # Concatenation fusion
        fused = self.fusion(img_feat, txt_feat)   # (B, fusion_dim)

        # U-Net with FiLM conditioning
        logits = self.unet(img, text=fused)
        return logits

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
