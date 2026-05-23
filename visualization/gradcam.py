"""
visualization/gradcam.py
─────────────────────────────────────────────────────────────────────────────
Decoder-Grad-CAM for FiLM-conditioned U-Net segmentation models.

Key corrections vs. previous version:
  1. TARGET LAYER  — now resolves to the LAST DECODER DOUBLECONV before
     the segmentation head (unet.ups[-1].conv), where FiLM conditioning
     has already been applied.  Encoder/bottleneck layers are NOT used.

  2. TARGET FUNCTION — three configurable modes (see GradCAMMode):
       'pred_tumor'  : gradient w.r.t. predicted tumor pixels only
       'gt_tumor'    : gradient w.r.t. ground-truth tumor region
       'top_k'       : gradient w.r.t. top-K logit activations

  3. QUALITY        — decoder features at 64×64 (one Up before final) give
     sharper, semantically richer maps than the 8×8 bottleneck.
─────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Grad-CAM target modes ──────────────────────────────────────────────────

class GradCAMMode:
    PRED_TUMOR = "pred_tumor"   # pixels where sigmoid(logit) > threshold
    GT_TUMOR   = "gt_tumor"    # pixels where GT mask == 1
    TOP_K      = "top_k"       # top-K highest-confidence activations


def _resolve_decoder_target(model: nn.Module) -> nn.Module:
    """
    Robustly resolve the LAST DECODER DoubleConv block.

    Priority order:
      1. model.unet.ups[-1].conv.conv  (final Up block DoubleConv)
      2. model.unet.ups[-1].conv       (DoubleConv directly)
      3. model.unet.ups[-1]            (entire Up block)
      4. model.unet.bottleneck         (last resort)

    This ensures the CAM captures FiLM-conditioned multimodal features.
    """
    unet = getattr(model, "unet", model)
    if hasattr(unet, "ups") and len(unet.ups) > 0:
        last_up = unet.ups[-1]
        if hasattr(last_up, "conv"):
            dconv = last_up.conv
            if hasattr(dconv, "conv"):          # DoubleConv.conv = Sequential
                return dconv.conv               # last ConvBnReLU sequence
            return dconv
        return last_up
    if hasattr(unet, "bottleneck"):
        return unet.bottleneck
    raise AttributeError("Cannot resolve decoder target layer in model.")


# ── Core Grad-CAM class ────────────────────────────────────────────────────

class GradCAM:
    """
    Decoder-targeted Grad-CAM for FiLM-conditioned U-Net models.

    Parameters
    ----------
    model        : nn.Module  Full segmentation model.
    target_layer : nn.Module  Layer to capture (use _resolve_decoder_target).
    device       : str
    mode         : str        GradCAMMode constant. Default: 'pred_tumor'.
    threshold    : float      Sigmoid threshold for 'pred_tumor' mode.
    top_k        : int        Number of pixels for 'top_k' mode.
    """

    def __init__(
        self,
        model: nn.Module,
        target_layer: Optional[nn.Module] = None,
        device: str = "cpu",
        mode: str = GradCAMMode.PRED_TUMOR,
        threshold: float = 0.5,
        top_k: int = 50,
    ) -> None:
        self.model        = model
        self.device       = device
        self.mode         = mode
        self.threshold    = threshold
        self.top_k        = top_k

        # Auto-resolve if not provided
        self.target_layer = target_layer or _resolve_decoder_target(model)

        self._activations: Optional[torch.Tensor] = None
        self._gradients:   Optional[torch.Tensor] = None
        self._hooks: List = []
        self._register_hooks()

    def _register_hooks(self) -> None:
        def fwd_hook(module, inp, out):
            # For Sequential, grab the last tensor output
            if isinstance(out, (tuple, list)):
                self._activations = out[0].detach()
            else:
                self._activations = out.detach()

        def bwd_hook(module, grad_in, grad_out):
            if grad_out[0] is not None:
                self._gradients = grad_out[0].detach()

        self._hooks.append(self.target_layer.register_forward_hook(fwd_hook))
        self._hooks.append(self.target_layer.register_full_backward_hook(bwd_hook))

    def remove_hooks(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def _build_target(
        self,
        logits: torch.Tensor,
        gt_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute scalar target from logits according to self.mode."""
        if self.mode == GradCAMMode.GT_TUMOR and gt_mask is not None:
            # Gradient w.r.t. GT tumor region
            gt = gt_mask.to(logits.device).float()
            if gt.dim() == 3:
                gt = gt.unsqueeze(1)     # (B,1,H,W)
            return (logits * gt).sum()

        elif self.mode == GradCAMMode.TOP_K:
            # Gradient w.r.t. highest-confidence pixels
            flat = logits.view(-1)
            k    = min(self.top_k, flat.numel())
            vals, _ = flat.topk(k)
            return vals.mean()

        else:
            # Default: pred_tumor — only pixels predicted as tumor
            probs = torch.sigmoid(logits)
            tumor_mask = (probs > self.threshold).float()
            if tumor_mask.sum() == 0:
                # Fallback to global mean if no tumor predicted
                return logits.mean()
            return (logits * tumor_mask).sum() / (tumor_mask.sum() + 1e-8)

    @torch.enable_grad()
    def compute(
        self,
        batch: Dict[str, torch.Tensor],
        gt_mask: Optional[torch.Tensor] = None,
    ) -> np.ndarray:
        """
        Compute Grad-CAM saliency map.

        Parameters
        ----------
        batch   : dict with 'image' and optional 'text_emb'.
        gt_mask : (B, H, W) ground-truth binary mask (needed for gt_tumor mode).

        Returns
        -------
        cam : numpy array (B, H, W)  normalised heatmap in [0, 1].
        """
        self.model.eval()
        batch_dev = {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

        logits = self.model(batch_dev)           # (B, 1, H, W)
        target = self._build_target(logits, gt_mask)

        self.model.zero_grad()
        target.backward()

        grads = self._gradients          # (B, C, h, w)  or None
        acts  = self._activations        # (B, C, h, w)

        if grads is None or acts is None:
            raise RuntimeError(
                "Grad-CAM hooks returned None. The target layer may have no "
                "learnable parameters or is fully detached from the graph."
            )

        # Grad-CAM: channel weights = spatial avg of gradients
        weights = grads.mean(dim=(-2, -1), keepdim=True)   # (B, C, 1, 1)
        cam     = (weights * acts).sum(dim=1, keepdim=True) # (B, 1, h, w)
        cam     = F.relu(cam)

        # Resize to input resolution
        H, W = batch_dev["image"].shape[-2:]
        cam   = F.interpolate(cam, size=(H, W), mode="bilinear", align_corners=False)
        cam   = cam.squeeze(1).cpu().numpy()   # (B, H, W)

        # Normalise per sample
        for i in range(cam.shape[0]):
            mn, mx = cam[i].min(), cam[i].max()
            cam[i] = (cam[i] - mn) / (mx - mn + 1e-8)

        return cam


# ── Publication-quality figure ─────────────────────────────────────────────

def save_gradcam_figures(
    images:      torch.Tensor,     # (B, 1, H, W)
    cam_maps:    np.ndarray,       # (B, H, W)
    masks_gt:    torch.Tensor,     # (B, H, W)
    save_path:   Path,
    patient_ids: Optional[List[str]] = None,
    max_samples: int = 6,
    model_name:  str = "Model",
    cam_mode:    str = GradCAMMode.PRED_TUMOR,
) -> None:
    """
    Save publication-quality Grad-CAM overlay figure.

    Panels per row:
      FLAIR MRI | Grad-CAM heatmap | CAM + MRI overlay | GT mask overlay
    """
    import matplotlib.gridspec as gridspec
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    B     = min(images.shape[0], max_samples)
    fig   = plt.figure(figsize=(18, 4.2 * B), facecolor="#0d0d0d")
    gs    = gridspec.GridSpec(B, 4, figure=fig, hspace=0.05, wspace=0.04)

    col_titles = ["FLAIR MRI", "Grad-CAM Heat", "CAM Overlay", "Ground Truth"]
    cmap_hot   = plt.cm.inferno

    for col, title in enumerate(col_titles):
        ax_title = fig.add_subplot(gs[0, col])
        ax_title.set_title(title, fontsize=12, fontweight="bold",
                           color="white", pad=8)
        ax_title.axis("off")

    for i in range(B):
        img_np = images[i, 0].cpu().numpy()
        img_n  = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)
        cam    = cam_maps[i]
        gt_np  = masks_gt[i].cpu().numpy() if isinstance(masks_gt, torch.Tensor) \
                 else masks_gt[i]

        pid = patient_ids[i] if patient_ids else f"Sample {i}"

        # Panel 0: MRI
        ax = fig.add_subplot(gs[i, 0])
        ax.imshow(img_n, cmap="gray")
        ax.set_ylabel(pid, fontsize=7, color="white", rotation=0,
                      labelpad=65, va="center")
        ax.axis("off")

        # Panel 1: Raw Grad-CAM
        ax = fig.add_subplot(gs[i, 1])
        im = ax.imshow(cam, cmap="inferno", vmin=0, vmax=1)
        ax.axis("off")
        # Colorbar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="4%", pad=0.04)
        cb  = plt.colorbar(im, cax=cax)
        cb.ax.tick_params(labelsize=6, colors="white")
        cb.outline.set_edgecolor("white")

        # Panel 2: Overlay — MRI + CAM
        ax = fig.add_subplot(gs[i, 2])
        ax.imshow(img_n, cmap="gray")
        ax.imshow(cam, cmap="inferno", alpha=0.55, vmin=0, vmax=1)
        ax.axis("off")
        # Annotate Dice if possible
        tumor_px = int((gt_np > 0.5).sum())
        ax.set_xlabel(f"GT tumor px: {tumor_px}", fontsize=7, color="#aaaaaa")

        # Panel 3: GT mask overlay
        ax = fig.add_subplot(gs[i, 3])
        ax.imshow(img_n, cmap="gray")
        ax.imshow(
            np.ma.masked_where(gt_np < 0.5, gt_np),
            cmap="Greens", alpha=0.65, vmin=0, vmax=1,
        )
        ax.axis("off")

    mode_label = cam_mode.replace("_", " ").title()
    fig.suptitle(
        f"{model_name} — Decoder Grad-CAM  [target: {mode_label}]",
        fontsize=13, color="white", y=1.01,
    )
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight", dpi=140,
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Grad-CAM figure saved → {save_path}")
