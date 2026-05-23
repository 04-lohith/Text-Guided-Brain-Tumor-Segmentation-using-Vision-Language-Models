"""
evaluation/metrics.py
─────────────────────────────────────────────────────────────────────────────
Segmentation metrics for binary brain tumour masks.

Implemented:
  - Dice Coefficient
  - Intersection over Union (IoU / Jaccard)
  - Precision
  - Recall (Sensitivity)
  - Hausdorff Distance (95th percentile)

Note on ROUGE / BLEU:
  These metrics measure token-level overlap between generated text and
  reference text. In this project we do NOT generate any text; we use text
  as a *conditioning signal* for image segmentation. Therefore ROUGE/BLEU
  are NOT applicable to our evaluation protocol. Segmentation quality is
  fully captured by Dice, IoU, Precision, Recall, and Hausdorff Distance.
─────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations
import warnings
from typing import Dict, List, Optional
import numpy as np
import torch


# ─────────────────────────────────────────────────────────────────────────────
# Helper: convert logits → binary mask
# ─────────────────────────────────────────────────────────────────────────────

def logits_to_binary(logits: torch.Tensor, threshold: float = 0.5) -> np.ndarray:
    """
    logits : (B, 1, H, W) or (B, H, W)
    Returns numpy bool array (B, H, W).
    """
    probs = torch.sigmoid(logits)
    if probs.ndim == 4:
        probs = probs.squeeze(1)
    return (probs.detach().cpu().numpy() >= threshold).astype(bool)


def to_numpy_bool(mask: torch.Tensor) -> np.ndarray:
    """Convert mask tensor to numpy bool array."""
    m = mask.detach().cpu().numpy()
    return m.astype(bool)


# ─────────────────────────────────────────────────────────────────────────────
# Per-sample metric functions
# ─────────────────────────────────────────────────────────────────────────────

def dice_coeff(pred: np.ndarray, gt: np.ndarray, smooth: float = 1e-6) -> float:
    """Dice coefficient for a single binary pair. Returns float in [0, 1]."""
    inter = (pred & gt).sum()
    union = pred.sum() + gt.sum()
    return float((2.0 * inter + smooth) / (union + smooth))


def iou_score(pred: np.ndarray, gt: np.ndarray, smooth: float = 1e-6) -> float:
    """Intersection over Union (Jaccard index)."""
    inter = (pred & gt).sum()
    union = (pred | gt).sum()
    return float((inter + smooth) / (union + smooth))


def precision_score(pred: np.ndarray, gt: np.ndarray, smooth: float = 1e-6) -> float:
    """Precision = TP / (TP + FP)."""
    tp = (pred & gt).sum()
    fp = (pred & ~gt).sum()
    return float((tp + smooth) / (tp + fp + smooth))


def recall_score(pred: np.ndarray, gt: np.ndarray, smooth: float = 1e-6) -> float:
    """Recall (Sensitivity) = TP / (TP + FN)."""
    tp = (pred & gt).sum()
    fn = (~pred & gt).sum()
    return float((tp + smooth) / (tp + fn + smooth))


def hausdorff_distance_95(
    pred: np.ndarray, gt: np.ndarray, percentile: int = 95
) -> float:
    """
    Compute the 95th-percentile Hausdorff Distance using scipy.

    Returns np.inf when either mask is empty (graceful handling for
    all-background slices).
    """
    try:
        from scipy.ndimage import distance_transform_edt
    except ImportError:
        warnings.warn("scipy not installed; Hausdorff Distance unavailable.")
        return float("nan")

    if pred.sum() == 0 or gt.sum() == 0:
        return float("inf")

    dt_pred = distance_transform_edt(~pred)
    dt_gt   = distance_transform_edt(~gt)

    surface_pred = dt_gt[pred]
    surface_gt   = dt_pred[gt]

    hd95 = max(
        np.percentile(surface_pred, percentile),
        np.percentile(surface_gt,   percentile),
    )
    return float(hd95)


# ─────────────────────────────────────────────────────────────────────────────
# Batch metric accumulator
# ─────────────────────────────────────────────────────────────────────────────

class MetricAccumulator:
    """
    Accumulate per-sample metrics over an entire epoch and compute means.

    Usage
    -----
    acc = MetricAccumulator()
    for batch in loader:
        logits, targets = model(batch), batch['mask']
        acc.update(logits, targets)
    results = acc.compute()
    """

    def __init__(
        self,
        threshold: float = 0.5,
        hausdorff_percentile: int = 95,
        compute_hausdorff: bool = True,
    ) -> None:
        self.threshold           = threshold
        self.hausdorff_pct       = hausdorff_percentile
        self.compute_hausdorff   = compute_hausdorff
        self._reset()

    def _reset(self) -> None:
        self._dice:       List[float] = []
        self._iou:        List[float] = []
        self._precision:  List[float] = []
        self._recall:     List[float] = []
        self._hausdorff:  List[float] = []

    def update(
        self,
        logits:  torch.Tensor,   # (B, 1, H, W)
        targets: torch.Tensor,   # (B, H, W)
    ) -> Dict[str, float]:
        """
        Compute metrics for the current batch and accumulate.
        Returns per-batch mean metrics dict.
        """
        preds_np = logits_to_binary(logits, self.threshold)
        gts_np   = to_numpy_bool(targets)

        batch_dice, batch_iou, batch_prec, batch_rec, batch_hd = [], [], [], [], []

        for pred, gt in zip(preds_np, gts_np):
            d  = dice_coeff(pred, gt)
            io = iou_score(pred, gt)
            pr = precision_score(pred, gt)
            re = recall_score(pred, gt)

            self._dice.append(d)
            self._iou.append(io)
            self._precision.append(pr)
            self._recall.append(re)

            batch_dice.append(d)
            batch_iou.append(io)
            batch_prec.append(pr)
            batch_rec.append(re)

            if self.compute_hausdorff:
                hd = hausdorff_distance_95(pred, gt, self.hausdorff_pct)
                # Exclude inf values for running mean (they'll appear in final report)
                if not np.isinf(hd):
                    self._hausdorff.append(hd)
                batch_hd.append(hd if not np.isinf(hd) else float("nan"))

        result = {
            "dice":      float(np.mean(batch_dice)),
            "iou":       float(np.mean(batch_iou)),
            "precision": float(np.mean(batch_prec)),
            "recall":    float(np.mean(batch_rec)),
        }
        if self.compute_hausdorff:
            finite = [v for v in batch_hd if not np.isnan(v)]
            result["hausdorff"] = float(np.mean(finite)) if finite else float("nan")
        return result

    def compute(self) -> Dict[str, float]:
        """Return epoch-level mean of all accumulated metrics."""
        result = {
            "dice":      float(np.mean(self._dice))      if self._dice      else 0.0,
            "iou":       float(np.mean(self._iou))       if self._iou       else 0.0,
            "precision": float(np.mean(self._precision)) if self._precision else 0.0,
            "recall":    float(np.mean(self._recall))    if self._recall    else 0.0,
        }
        if self._hausdorff:
            result["hausdorff"] = float(np.mean(self._hausdorff))
        else:
            result["hausdorff"] = float("nan")
        return result

    def reset(self) -> None:
        self._reset()

    def get_per_sample_dice(self) -> List[float]:
        return list(self._dice)
