"""
visualization/plot_segmentation.py
─────────────────────────────────────────────────────────────────────────────
Visualise MRI slices with ground-truth and predicted segmentation masks.

Panels per sample:
  1. FLAIR MRI slice (grayscale)
  2. Ground-truth mask (red overlay)
  3. Predicted mask   (blue overlay)
  4. Overlay: MRI + GT (green) + Pred (red) for error analysis
─────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


def _to_numpy(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().numpy()


def overlay_mask(
    img: np.ndarray,     # (H, W)  grayscale, normalised
    mask: np.ndarray,    # (H, W)  binary float
    color: Tuple[float, float, float] = (1.0, 0.0, 0.0),
    alpha: float = 0.45,
) -> np.ndarray:
    """
    Return an RGB image (H, W, 3) with the mask region coloured.
    """
    # Normalise image to [0, 1]
    img_n = (img - img.min()) / (img.max() - img.min() + 1e-8)
    rgb   = np.stack([img_n, img_n, img_n], axis=-1)

    # Blend mask region
    for c, col in enumerate(color):
        rgb[:, :, c] = np.where(
            mask > 0.5,
            alpha * col + (1 - alpha) * img_n,
            rgb[:, :, c],
        )
    return np.clip(rgb, 0, 1)


def visualise_predictions(
    images:   torch.Tensor,    # (B, 1, H, W)
    masks_gt: torch.Tensor,    # (B, H, W)
    logits:   torch.Tensor,    # (B, 1, H, W)
    save_path: Path,
    patient_ids: Optional[List[str]] = None,
    threshold: float = 0.5,
    max_samples: int = 8,
) -> None:
    """
    Save a grid figure with (MRI | GT | Pred | Overlay) for up to max_samples.

    Parameters
    ----------
    images     : FLAIR slices from a batch
    masks_gt   : ground-truth binary masks
    logits     : model output logits
    save_path  : where to save the figure
    patient_ids: list of patient ID strings for titles
    threshold  : binarisation threshold
    max_samples: cap on number of samples shown
    """
    from evaluation.metrics import logits_to_binary

    preds = logits_to_binary(logits, threshold)          # (B, H, W) bool
    B     = min(images.shape[0], max_samples)

    fig, axes = plt.subplots(B, 4, figsize=(16, 4 * B))
    if B == 1:
        axes = axes[np.newaxis, :]

    col_titles = ["FLAIR MRI", "Ground Truth", "Prediction", "Overlay (GT/Pred)"]
    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=11, fontweight="bold")

    for i in range(B):
        img_np  = images[i, 0].cpu().numpy()             # (H, W)
        gt_np   = masks_gt[i].cpu().numpy().astype(float)
        pred_np = preds[i].astype(float)

        pid = patient_ids[i] if patient_ids else f"Sample {i}"

        # Panel 1: MRI
        axes[i, 0].imshow(img_np, cmap="gray")
        axes[i, 0].set_ylabel(pid, fontsize=8, rotation=0, labelpad=60, va="center")
        axes[i, 0].axis("off")

        # Panel 2: GT mask
        axes[i, 1].imshow(overlay_mask(img_np, gt_np, color=(0.1, 0.9, 0.1)))
        axes[i, 1].axis("off")

        # Panel 3: Predicted mask
        axes[i, 2].imshow(overlay_mask(img_np, pred_np, color=(0.9, 0.2, 0.2)))
        axes[i, 2].axis("off")

        # Panel 4: Combined overlay (GT=green, Pred=red)
        img_n  = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)
        rgb    = np.stack([img_n, img_n, img_n], axis=-1).copy()
        tp_mask = (gt_np > 0.5) & (pred_np > 0.5)
        fp_mask = (gt_np < 0.5) & (pred_np > 0.5)
        fn_mask = (gt_np > 0.5) & (pred_np < 0.5)
        for r, g, b, m in [
            (0.1, 0.9, 0.1, tp_mask),   # TP green
            (0.9, 0.2, 0.2, fp_mask),   # FP red
            (0.9, 0.9, 0.1, fn_mask),   # FN yellow
        ]:
            for c, col in enumerate([r, g, b]):
                rgb[:, :, c] = np.where(m, 0.5 * col + 0.5 * rgb[:, :, c], rgb[:, :, c])
        axes[i, 3].imshow(np.clip(rgb, 0, 1))
        axes[i, 3].axis("off")

    fig.suptitle("Segmentation Results  (Green=TP  Red=FP  Yellow=FN)", fontsize=12, y=1.01)
    fig.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight", dpi=120)
    plt.close(fig)
    print(f"Segmentation visualisation saved → {save_path}")
