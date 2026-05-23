"""
training/losses.py
─────────────────────────────────────────────────────────────────────────────
Loss functions for binary brain tumour segmentation.

Available losses:
  - DiceLoss         (soft Dice)
  - BCEDiceLoss      (weighted BCE + Dice)   ← default
  - FocalDiceLoss    (Focal + Dice)
  - FocalLoss
─────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """
    Soft Dice Loss for binary segmentation.

        Dice = 1 - (2 * |X ∩ Y| + ε) / (|X| + |Y| + ε)

    Parameters
    ----------
    smooth : float   Laplace smoothing constant to avoid division by zero.
    """

    def __init__(self, smooth: float = 1e-6) -> None:
        super().__init__()
        self.smooth = float(smooth)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits  : (B, 1, H, W)  raw logits (before sigmoid)
        targets : (B, H, W)     binary float mask
        """
        probs = torch.sigmoid(logits).squeeze(1)    # (B, H, W)
        flat_p = probs.reshape(probs.shape[0], -1)
        flat_t = targets.reshape(targets.shape[0], -1)
        inter  = (flat_p * flat_t).sum(dim=1)
        union  = flat_p.sum(dim=1) + flat_t.sum(dim=1)
        dice   = (2.0 * inter + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()


class FocalLoss(nn.Module):
    """
    Binary Focal Loss.

        FL(p) = -α * (1 - p)^γ * log(p)

    Parameters
    ----------
    gamma : float  Focusing parameter (default 2.0).
    alpha : float  Class balance weight for positive class (default 0.25).
    """

    def __init__(self, gamma: float = 2.0, alpha: float = 0.25) -> None:
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits  : (B, 1, H, W)
        targets : (B, H, W)
        """
        targets_exp = targets.unsqueeze(1).float()    # (B, 1, H, W)
        bce  = F.binary_cross_entropy_with_logits(
            logits, targets_exp, reduction="none"
        )
        p    = torch.exp(-bce)
        focal = self.alpha * ((1 - p) ** self.gamma) * bce
        return focal.mean()


class BCEDiceLoss(nn.Module):
    """
    Weighted sum of Binary Cross-Entropy and Soft Dice Loss.

        L = w_bce * BCE + w_dice * Dice

    Parameters
    ----------
    bce_weight  : float  Weight for BCE component.
    dice_weight : float  Weight for Dice component.
    smooth      : float  Dice smoothing.
    pos_weight  : float  BCEWithLogitsLoss positive class weight (handles imbalance).
    """

    def __init__(
        self,
        bce_weight:  float = 0.3,
        dice_weight: float = 0.7,
        smooth:      float = 1e-6,
        pos_weight:  float = 5.0,
    ) -> None:
        super().__init__()
        self.bce_weight  = float(bce_weight)
        self.dice_weight = float(dice_weight)
        self.dice_loss   = DiceLoss(smooth=float(smooth))
        pw = torch.tensor([float(pos_weight)])
        self.register_buffer("pos_weight", pw)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets_exp = targets.unsqueeze(1).float()
        bce  = F.binary_cross_entropy_with_logits(
            logits, targets_exp, pos_weight=self.pos_weight
        )
        dice = self.dice_loss(logits, targets)
        return self.bce_weight * bce + self.dice_weight * dice


class FocalDiceLoss(nn.Module):
    """
    Weighted sum of Focal Loss and Soft Dice Loss.

        L = w_focal * Focal + w_dice * Dice
    """

    def __init__(
        self,
        focal_weight: float = 0.3,
        dice_weight:  float = 0.7,
        gamma:        float = 2.0,
        alpha:        float = 0.25,
        smooth:       float = 1e-6,
    ) -> None:
        super().__init__()
        self.focal_weight = focal_weight
        self.dice_weight  = dice_weight
        self.focal = FocalLoss(gamma=gamma, alpha=alpha)
        self.dice  = DiceLoss(smooth=smooth)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return (self.focal_weight * self.focal(logits, targets)
                + self.dice_weight  * self.dice(logits,  targets))


class TverskyLoss(nn.Module):
    """
    Tversky Loss — better for small tumor segmentation.

        Tversky = 1 - (TP + ε) / (TP + α*FP + β*FN + ε)

    α < β penalizes false negatives more than false positives,
    which helps recall on small tumors.

    Recommended: alpha=0.3, beta=0.7 for medical segmentation.
    """

    def __init__(self, alpha: float = 0.3, beta: float = 0.7,
                 smooth: float = 1e-6) -> None:
        super().__init__()
        self.alpha  = alpha
        self.beta   = beta
        self.smooth = float(smooth)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs  = torch.sigmoid(logits).squeeze(1)     # (B, H, W)
        flat_p = probs.reshape(probs.shape[0], -1)
        flat_t = targets.reshape(targets.shape[0], -1).float()
        tp     = (flat_p * flat_t).sum(dim=1)
        fp     = (flat_p * (1 - flat_t)).sum(dim=1)
        fn     = ((1 - flat_p) * flat_t).sum(dim=1)
        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        return 1.0 - tversky.mean()


class TverskyFocalLoss(nn.Module):
    """
    Focal Tversky Loss — Tversky raised to the power of (1/gamma)
    to focus on hard examples.

        FTL = (1 - Tversky)^(1/gamma)
    """

    def __init__(self, alpha: float = 0.3, beta: float = 0.7,
                 gamma: float = 0.75, smooth: float = 1e-6) -> None:
        super().__init__()
        self.tversky = TverskyLoss(alpha=alpha, beta=beta, smooth=smooth)
        self.gamma   = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        t_loss = self.tversky(logits, targets)
        return t_loss ** (1.0 / self.gamma)


def compute_attention_entropy_loss(
    model: nn.Module,
    weight: float = 0.01,
    target_entropy: float = 1.5,
) -> torch.Tensor:
    """
    Compute attention entropy regularization for Model 3.
    Safe to call on any model — returns 0.0 for Models 1 and 2.
    """
    if hasattr(model, "get_attn_entropy_loss"):
        return weight * model.get_attn_entropy_loss(target_entropy)
    return torch.tensor(0.0)


def build_loss(cfg) -> nn.Module:
    """
    Instantiate the correct loss function from config.

    config.loss.type ∈ {'dice', 'bce', 'dice_bce', 'focal_dice',
                         'tversky', 'tversky_focal'}
    """
    loss_cfg  = cfg.loss
    loss_type = str(loss_cfg.type).lower()

    if loss_type == "dice":
        return DiceLoss(smooth=float(loss_cfg.smooth))

    elif loss_type in ("bce", "bce_dice", "dice_bce"):
        return BCEDiceLoss(
            bce_weight=float(loss_cfg.bce_weight),
            dice_weight=float(loss_cfg.dice_weight),
            smooth=float(loss_cfg.smooth),
        )

    elif loss_type in ("focal_dice", "focal"):
        return FocalDiceLoss(
            gamma=float(loss_cfg.focal_gamma),
            alpha=float(loss_cfg.focal_alpha),
            smooth=float(loss_cfg.smooth),
        )

    elif loss_type == "tversky":
        return TverskyLoss(
            alpha=float(getattr(loss_cfg, "tversky_alpha", 0.3)),
            beta=float(getattr(loss_cfg, "tversky_beta",  0.7)),
            smooth=float(loss_cfg.smooth),
        )

    elif loss_type == "tversky_focal":
        return TverskyFocalLoss(
            alpha=float(getattr(loss_cfg, "tversky_alpha", 0.3)),
            beta=float(getattr(loss_cfg, "tversky_beta",  0.7)),
            gamma=float(getattr(loss_cfg, "tversky_gamma", 0.75)),
            smooth=float(loss_cfg.smooth),
        )

    else:
        raise ValueError(
            f"Unknown loss type '{loss_type}'. "
            "Choose from: 'dice', 'dice_bce', 'focal_dice', 'tversky', 'tversky_focal'."
        )
