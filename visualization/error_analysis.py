"""
visualization/error_analysis.py
─────────────────────────────────────────────────────────────────────────────
Error analysis utilities:

  1. Dice score distribution histogram
  2. Worst predictions visualisation
  3. Failure case statistics
─────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from evaluation.metrics import MetricAccumulator, logits_to_binary


def collect_per_slice_metrics(
    model,
    loader: DataLoader,
    device: str = "cpu",
    max_batches: Optional[int] = None,
) -> List[Dict]:
    """
    Run model on val set and collect per-slice records.

    Returns
    -------
    records : list of dicts with keys:
        'dice', 'iou', 'patient_id', 'slice_idx',
        'image', 'mask', 'pred'
    """
    from evaluation.metrics import dice_coeff, iou_score

    model.eval()
    records = []

    with torch.no_grad():
        for i, batch in enumerate(loader):
            if max_batches and i >= max_batches:
                break

            batch_dev = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            logits  = model(batch_dev)
            preds   = logits_to_binary(logits)   # (B, H, W) bool
            gt      = batch_dev["mask"].cpu().numpy().astype(bool)
            imgs    = batch["image"]
            pids    = batch["patient_id"]
            sidxs   = batch["slice_idx"].tolist()

            for b in range(logits.shape[0]):
                d  = dice_coeff(preds[b], gt[b])
                io = iou_score(preds[b], gt[b])
                records.append({
                    "dice":       d,
                    "iou":        io,
                    "patient_id": pids[b],
                    "slice_idx":  sidxs[b],
                    "image":      imgs[b].squeeze(0).numpy(),   # (H, W)
                    "mask":       gt[b],                         # (H, W) bool
                    "pred":       preds[b],                      # (H, W) bool
                })

    return records


def plot_dice_distribution(
    records:    List[Dict],
    save_path:  Path,
    model_name: str = "model",
) -> None:
    """
    Histogram of per-slice Dice scores with statistics overlay.
    """
    dices = [r["dice"] for r in records]
    mean_d = np.mean(dices)
    std_d  = np.std(dices)
    med_d  = np.median(dices)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(dices, bins=40, color="#5470c6", edgecolor="white",
            alpha=0.85, density=True)
    ax.axvline(mean_d, color="#e07b54", lw=2, label=f"Mean={mean_d:.3f}")
    ax.axvline(med_d,  color="#91cc75", lw=2, ls="--", label=f"Median={med_d:.3f}")
    ax.fill_betweenx(
        [0, ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 1],
        mean_d - std_d, mean_d + std_d,
        alpha=0.15, color="#e07b54", label=f"±1σ={std_d:.3f}",
    )
    ax.set_xlabel("Dice Score per Slice")
    ax.set_ylabel("Density")
    ax.set_xlim(0, 1)
    ax.set_title(f"{model_name} — Dice Score Distribution")
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=120)
    plt.close(fig)
    print(f"Dice distribution saved → {save_path}")


def plot_worst_predictions(
    records:    List[Dict],
    save_path:  Path,
    n_worst:    int = 6,
    model_name: str = "model",
) -> None:
    """
    Show the N slices with the lowest Dice score (worst predictions).
    """
    # Filter out slices where GT is all-background (Dice≈1 trivially)
    non_trivial = [r for r in records if r["mask"].sum() > 10]
    sorted_recs = sorted(non_trivial, key=lambda x: x["dice"])[:n_worst]

    if not sorted_recs:
        print("No non-trivial slices found for worst prediction analysis.")
        return

    fig, axes = plt.subplots(n_worst, 3, figsize=(12, 4 * n_worst))
    if n_worst == 1:
        axes = axes[np.newaxis, :]

    for col, title in enumerate(["FLAIR MRI", "GT Mask", "Prediction"]):
        axes[0, col].set_title(title, fontsize=11, fontweight="bold")

    for i, rec in enumerate(sorted_recs):
        img = rec["image"]
        gt  = rec["mask"].astype(float)
        pr  = rec["pred"].astype(float)
        img_n = (img - img.min()) / (img.max() - img.min() + 1e-8)

        pid = f"{rec['patient_id']} s={rec['slice_idx']}  Dice={rec['dice']:.3f}"
        axes[i, 0].set_ylabel(pid, fontsize=7, rotation=0, labelpad=75, va="center")

        axes[i, 0].imshow(img_n, cmap="gray")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(img_n, cmap="gray")
        axes[i, 1].imshow(np.ma.masked_where(gt < 0.5, gt),
                          cmap="Greens", alpha=0.6, vmin=0, vmax=1)
        axes[i, 1].axis("off")

        axes[i, 2].imshow(img_n, cmap="gray")
        axes[i, 2].imshow(np.ma.masked_where(pr < 0.5, pr),
                          cmap="Reds", alpha=0.6, vmin=0, vmax=1)
        axes[i, 2].axis("off")

    fig.suptitle(f"{model_name} — Worst {n_worst} Predictions (by Dice)", fontsize=13)
    fig.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight", dpi=120)
    plt.close(fig)
    print(f"Worst predictions saved → {save_path}")
