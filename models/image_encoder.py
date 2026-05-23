"""
models/image_encoder.py
─────────────────────────────────────────────────────────────────────────────
CLIP ViT-B/32 image encoder for Models 2 and 3.

Since CLIP was trained on RGB images but our input is single-channel FLAIR,
we replicate the grayscale slice to 3 channels (1→3) before passing to CLIP.

The encoder is frozen by default (freeze_clip=True in config) and only the
projection head is fine-tuned. This dramatically reduces memory overhead.

Output: a (B, image_embed_dim) global feature vector per slice.
        Additionally, patch-level tokens (B, N_patches, 512) are exposed
        for cross-attention fusion in Model 3.
─────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class CLIPImageEncoder(nn.Module):
    """
    Wraps a CLIP ViT-B/32 model and exposes:
      - global_embed : (B, 512)         — CLS token projected by CLIP
      - patch_tokens : (B, N, 512)      — patch-level tokens for cross-attn

    A lightweight projection head maps (512,) → (out_dim,).
    Single-channel images are expanded to 3 channels internally.
    CLIP preprocessing (resize + normalise) is done inside forward().

    Parameters
    ----------
    out_dim    : int   Projection output dimension (fusion_dim from config).
    freeze     : bool  Freeze CLIP weights (recommended for MacBook).
    clip_model : str   CLIP model name ('ViT-B/32' etc.)
    """

    CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
    CLIP_STD  = (0.26862954, 0.26130258, 0.27577711)
    CLIP_SIZE = 224   # CLIP ViT-B/32 input resolution

    def __init__(
        self,
        out_dim: int = 256,
        freeze: bool = True,
        clip_model: str = "ViT-B/32",
        unfreeze_last_n_blocks: int = 0,   # 0 = fully frozen; 2 = unfreeze last 2 blocks
    ) -> None:
        super().__init__()
        try:
            import clip
            self.clip, _ = clip.load(clip_model, device="cpu", jit=False)
        except ImportError:
            raise ImportError(
                "OpenAI CLIP not installed. "
                "Run: pip install git+https://github.com/openai/CLIP.git"
            )

        self._embed_dim = 512   # ViT-B/32 output dim

        # Step 1: freeze everything
        if freeze:
            for p in self.clip.parameters():
                p.requires_grad_(False)

        # Step 2: selectively unfreeze the last N transformer blocks
        # ViT-B/32 has 12 transformer residual blocks (visual.transformer.resblocks)
        if unfreeze_last_n_blocks > 0 and freeze:
            resblocks = self.clip.visual.transformer.resblocks
            n_total   = len(resblocks)   # 12 for ViT-B/32
            start_idx = max(0, n_total - unfreeze_last_n_blocks)
            for block in resblocks[start_idx:]:
                for p in block.parameters():
                    p.requires_grad_(True)
            # Also unfreeze ln_post (post-transformer layer norm)
            for p in self.clip.visual.ln_post.parameters():
                p.requires_grad_(True)

        # Lightweight projection head
        self.proj = nn.Sequential(
            nn.Linear(self._embed_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
        )

        # Register CLIP normalisation constants as buffers
        mean = torch.tensor(self.CLIP_MEAN).view(1, 3, 1, 1)
        std  = torch.tensor(self.CLIP_STD).view(1, 3, 1, 1)
        self.register_buffer("clip_mean", mean)
        self.register_buffer("clip_std",  std)

    def _preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, 1, H, W) — z-scored grayscale slice
        Returns (B, 3, 224, 224) normalised for CLIP.
        """
        # 1) Rescale to [0, 1] per-sample (approximate, since already z-scored)
        xmin = x.flatten(1).min(1)[0].view(-1, 1, 1, 1)
        xmax = x.flatten(1).max(1)[0].view(-1, 1, 1, 1)
        x = (x - xmin) / (xmax - xmin + 1e-8)

        # 2) Repeat to 3 channels
        x = x.expand(-1, 3, -1, -1)

        # 3) Resize to 224×224
        x = F.interpolate(x, size=self.CLIP_SIZE, mode="bilinear",
                          align_corners=False)

        # 4) CLIP normalisation
        x = (x - self.clip_mean) / self.clip_std
        return x

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x : (B, 1, H, W)

        Returns
        -------
        global_feat  : (B, out_dim)     projected CLS embedding
        patch_tokens : (B, N, 512)      raw patch tokens (for cross-attn)
        """
        x_clip = self._preprocess(x)

        # Forward through CLIP ViT visual encoder
        # We need patch-level tokens, so we hook into the transformer
        visual = self.clip.visual

        # Extract patch tokens via a forward hook approach
        patch_tokens, cls_feat = self._extract_tokens(visual, x_clip)

        global_feat = self.proj(cls_feat.float())   # (B, out_dim)
        return global_feat, patch_tokens.float()    # (B, N, 512)

    @staticmethod
    def _extract_tokens(
        visual_encoder, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Run the CLIP ViT visual encoder and return
        (patch_tokens, cls_token) without modifying the CLIP module.

        Gradients are allowed to flow so that partially unfrozen transformer
        blocks can be fine-tuned. Fully frozen layers produce zero gradients
        naturally — no explicit no_grad() needed.
        """
        # Determine the expected dtype from CLIP's own weights
        clip_dtype = next(visual_encoder.parameters()).dtype
        x_fp = x.to(dtype=clip_dtype)

        # Patch embedding
        x_p = visual_encoder.conv1(x_fp)
        x_p = x_p.reshape(x_p.shape[0], x_p.shape[1], -1)
        x_p = x_p.permute(0, 2, 1)

        # Class token
        cls = visual_encoder.class_embedding.unsqueeze(0).expand(
            x_p.shape[0], -1, -1
        )
        x_p = torch.cat([cls, x_p], dim=1)
        x_p = x_p + visual_encoder.positional_embedding

        x_p = visual_encoder.ln_pre(x_p)
        x_p = x_p.permute(1, 0, 2)   # NLD → LND
        x_p = visual_encoder.transformer(x_p)
        x_p = x_p.permute(1, 0, 2)   # LND → NLD

        # CLS token at position 0
        cls_out = visual_encoder.ln_post(x_p[:, 0, :])
        if visual_encoder.proj is not None:
            cls_out = cls_out @ visual_encoder.proj

        patch_out = x_p[:, 1:, :]   # (B, N, D) raw patch tokens

        return patch_out.float(), cls_out.float()
