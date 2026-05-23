"""
models/unet.py
─────────────────────────────────────────────────────────────────────────────
Standard 2-D U-Net segmentation backbone.

Architecture
─────────────────────────────────────────────────────────────────────────────
Encoder (4 down-stages) → Bottleneck → Decoder (4 up-stages) → Output head.

Each encoder block:   ConvBnReLU × 2  → MaxPool
Bottleneck:           ConvBnReLU × 2
Each decoder block:   UpConv → concat skip → ConvBnReLU × 2
Output head:          Conv 1×1 → (sigmoid for binary)

The decoder can optionally receive a text conditioning tensor that is
added (FiLM-style) at each decoder stage. When text_dim=0 the model
behaves as a pure image-only U-Net (Model 1).
─────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations
from typing import List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Building blocks
# ─────────────────────────────────────────────────────────────────────────────

class ConvBnReLU(nn.Module):
    """3×3 Conv → BN → ReLU (inplace)."""

    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0) -> None:
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DoubleConv(nn.Module):
    """Two consecutive ConvBnReLU blocks."""

    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            ConvBnReLU(in_ch, out_ch, dropout),
            ConvBnReLU(out_ch, out_ch, dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Down(nn.Module):
    """MaxPool → DoubleConv (encoder stage)."""

    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_ch, out_ch, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


class Up(nn.Module):
    """Bilinear upsample → cat(skip) → DoubleConv (decoder stage)."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int,
                 dropout: float = 0.0) -> None:
        super().__init__()
        self.up   = nn.Upsample(scale_factor=2, mode="bilinear",
                                align_corners=True)
        self.conv = DoubleConv(in_ch + skip_ch, out_ch, dropout)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # Pad if spatial dims differ
        dh = skip.shape[-2] - x.shape[-2]
        dw = skip.shape[-1] - x.shape[-1]
        x  = F.pad(x, [dw // 2, dw - dw // 2, dh // 2, dh - dh // 2])
        return self.conv(torch.cat([skip, x], dim=1))


# ─────────────────────────────────────────────────────────────────────────────
# FiLM conditioning module  (feature-wise linear modulation)
# ─────────────────────────────────────────────────────────────────────────────

class FiLM(nn.Module):
    """
    Applies scale (γ) and shift (β) from a text vector to a feature map.

        y = γ(text) * x + β(text)

    text_dim → linear projection → (2 * channels,) → split into γ, β
    Both γ and β have shape (B, C, 1, 1) so they broadcast over H, W.
    """

    def __init__(self, text_dim: int, channels: int) -> None:
        super().__init__()
        self.proj = nn.Linear(text_dim, 2 * channels)

    def forward(self, x: torch.Tensor, text: torch.Tensor) -> torch.Tensor:
        """
        x    : (B, C, H, W)
        text : (B, text_dim)
        """
        params = self.proj(text)          # (B, 2*C)
        gamma, beta = params.chunk(2, dim=-1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)   # (B, C, 1, 1)
        beta  = beta.unsqueeze(-1).unsqueeze(-1)
        return gamma * x + beta


# ─────────────────────────────────────────────────────────────────────────────
# U-Net
# ─────────────────────────────────────────────────────────────────────────────

class UNet(nn.Module):
    """
    2-D U-Net with optional FiLM text conditioning.

    Parameters
    ----------
    in_channels : int
        Input image channels (1 for grayscale FLAIR slices).
    out_channels : int
        Number of output classes (1 for binary segmentation).
    base_ch : int
        Channel width at the first encoder level. Doubles each level.
    depth : int
        Number of encoder/decoder stages (4 by default → 16× downsampling).
    dropout : float
        Spatial dropout applied inside DoubleConv blocks.
    text_dim : int
        Dimension of incoming text embedding. 0 → no conditioning (Model 1).
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_ch: int = 32,
        depth: int = 4,
        dropout: float = 0.1,
        text_dim: int = 0,          # 0 = image-only mode
    ) -> None:
        super().__init__()
        self.depth    = depth
        self.text_dim = text_dim
        # ch = [base_ch, base_ch*2, ..., base_ch*2^depth]
        # e.g. depth=4, base_ch=32 → [32, 64, 128, 256, 512]
        ch = [base_ch * (2 ** i) for i in range(depth + 1)]
        self.ch = ch  # store for reference

        # ── Encoder ───────────────────────────────────────────────────────
        # enc_in: in → ch[0]
        # downs[k]: ch[k] → ch[k+1]  (k = 0..depth-1)
        self.enc_in = DoubleConv(in_channels, ch[0], dropout)
        self.downs  = nn.ModuleList([
            Down(ch[i], ch[i + 1], dropout) for i in range(depth)
        ])

        # ── Bottleneck ────────────────────────────────────────────────────
        # Input: ch[depth], Output: ch[depth]
        self.bottleneck = DoubleConv(ch[depth], ch[depth], dropout)

        # ── Decoder ───────────────────────────────────────────────────────
        # Decoder stage k (k=0..depth-1):
        #   x coming in  : ch[depth]   if k==0, else ch[depth-k+1-1] = ch[depth-k]
        #   skip          : ch[depth-k-1]   (from encoder)
        #   concat input  : ch[depth-k] + ch[depth-k-1]   → wait, this is wrong for k=0
        #
        # Correct trace:
        #   k=0: x=ch[depth], skip=ch[depth-1], Up(ch[depth], ch[depth-1], ch[depth-1])
        #   k=1: x=ch[depth-1], skip=ch[depth-2], Up(ch[depth-1], ch[depth-2], ch[depth-2])
        #   ...
        # General: Up(ch[depth-k], ch[depth-k-1], ch[depth-k-1])
        self.ups = nn.ModuleList([
            Up(ch[depth - k], ch[depth - k - 1], ch[depth - k - 1], dropout)
            for k in range(depth)
        ])

        # ── FiLM layers (applied AFTER each up block output, if text_dim > 0) ─
        # films[0] → after bottleneck     → ch[depth] channels
        # films[k+1] → after ups[k]       → ch[depth-k-1] channels
        if text_dim > 0:
            film_chs = [ch[depth]] + [ch[depth - k - 1] for k in range(depth)]
            self.films = nn.ModuleList([
                FiLM(text_dim, c) for c in film_chs
            ])

        # ── Output head ───────────────────────────────────────────────────
        self.out_conv = nn.Conv2d(ch[0], out_channels, kernel_size=1)

        self._init_weights()

    # ─── Weight init ──────────────────────────────────────────────────────

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    # ─── Forward ──────────────────────────────────────────────────────────

    def forward(
        self,
        x: torch.Tensor,
        text: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x    : (B, 1, H, W)   — FLAIR slice
        text : (B, text_dim)  — pooled text embedding (None for image-only)

        Returns
        -------
        logits : (B, out_channels, H, W)
        """
        # ── Encoder ──────────────────────────────────────────────────────
        # skips[0] = enc_in output  (ch[0] channels, full H×W)
        # skips[k] = downs[k-1] output  (ch[k] channels, H/2^k × W/2^k)  k=1..depth
        skips: List[torch.Tensor] = []
        x = self.enc_in(x)
        skips.append(x)                   # skips[0]  → ch[0]
        for down in self.downs:
            x = down(x)
            skips.append(x)               # skips[1..depth] → ch[1..depth]

        # At this point x == skips[-1] == skips[depth] (ch[depth] channels)
        # Feed to bottleneck (same channel count in/out)
        x = self.bottleneck(x)

        # FiLM at bottleneck (ch[depth] channels)
        if self.text_dim > 0 and text is not None:
            x = self.films[0](x, text)

        # ── Decoder ───────────────────────────────────────────────────────
        # k=0: skip = skips[depth-1]  (ch[depth-1]),  x has ch[depth]
        # k=1: skip = skips[depth-2]  (ch[depth-2]),  x has ch[depth-1]
        # ...
        # k=depth-1: skip = skips[0]  (ch[0]),        x has ch[1]
        for k, up in enumerate(self.ups):
            skip = skips[self.depth - 1 - k]   # ch[depth-k-1]
            x = up(x, skip)
            if self.text_dim > 0 and text is not None:
                x = self.films[k + 1](x, text)

        # ── Output ────────────────────────────────────────────────────────
        return self.out_conv(x)   # logits (B, out_ch, H, W)

    def get_encoder_features(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Return all encoder feature maps (for Grad-CAM)."""
        feats: List[torch.Tensor] = []
        x = self.enc_in(x)
        feats.append(x)
        for down in self.downs:
            x = down(x)
            feats.append(x)
        return feats
