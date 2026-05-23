"""
models/model1_image_only.py
─────────────────────────────────────────────────────────────────────────────
Model 1 — Pure image U-Net baseline (no text conditioning).

Pipeline:
    FLAIR slice (B,1,H,W)  →  U-Net  →  logits (B,1,H,W)
─────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations
from typing import Dict
import torch
import torch.nn as nn
from .unet import UNet


class ImageOnlyUNet(nn.Module):
    """
    Model 1 — Baseline U-Net without any text information.

    Parameters
    ----------
    cfg : DotDict   Project configuration.
    """

    MODEL_NAME = "model1_image_only"

    def __init__(self, cfg=None) -> None:
        super().__init__()
        base_ch = int(getattr(cfg.model, "unet_base_channels", 32)) if cfg else 32
        depth   = int(getattr(cfg.model, "unet_depth",         4))   if cfg else 4
        dropout = float(getattr(cfg.model, "dropout",          0.1)) if cfg else 0.1

        self.unet = UNet(
            in_channels=1,
            out_channels=1,
            base_ch=base_ch,
            depth=depth,
            dropout=dropout,
            text_dim=0,         # no text conditioning
        )

    def forward(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        batch : dict with keys 'image' (B,1,H,W), (others are ignored)

        Returns
        -------
        logits : (B, 1, H, W)
        """
        return self.unet(batch["image"])

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
