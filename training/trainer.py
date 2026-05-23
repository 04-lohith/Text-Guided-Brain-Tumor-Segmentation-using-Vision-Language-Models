"""
training/trainer.py
─────────────────────────────────────────────────────────────────────────────
Complete training engine for all three model variants.

Features
────────
  • AdamW + cosine / step / plateau LR scheduling
  • Gradient accumulation (memory efficient for MacBook)
  • Gradient clipping
  • Early stopping with configurable patience
  • Checkpoint saving (best model + periodic)
  • TensorBoard logging
  • Full metric tracking (Dice, IoU, Precision, Recall, Hausdorff)
  • Validation loop
─────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import logging
import math
import os
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    StepLR,
    ReduceLROnPlateau,
)
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from evaluation.metrics import MetricAccumulator
from training.losses import build_loss, compute_attention_entropy_loss

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint utilities
# ─────────────────────────────────────────────────────────────────────────────

def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    metrics: Dict[str, float],
    path: Path,
    is_best: bool = False,
) -> None:
    state = {
        "epoch":     epoch,
        "model":     model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if hasattr(scheduler, "state_dict") else None,
        "metrics":   metrics,
    }
    torch.save(state, path)
    if is_best:
        best_path = path.parent / f"{path.stem.split('_ep')[0]}_best.pth"
        torch.save(state, best_path)
        logger.info(f"  ✔ Best model saved → {best_path}")


def load_checkpoint(
    model: nn.Module,
    path: Path,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler=None,
    device: str = "cpu",
) -> Dict:
    state = torch.load(path, map_location=device)
    model.load_state_dict(state["model"])
    if optimizer and "optimizer" in state:
        optimizer.load_state_dict(state["optimizer"])
    if scheduler and state.get("scheduler"):
        scheduler.load_state_dict(state["scheduler"])
    logger.info(f"Checkpoint loaded from {path}  (epoch {state['epoch']})")
    return state


# ─────────────────────────────────────────────────────────────────────────────
# Trainer
# ─────────────────────────────────────────────────────────────────────────────

class Trainer:
    """
    Unified training loop for Models 1, 2, and 3.

    Parameters
    ----------
    model       : nn.Module         Any of the three model variants.
    train_loader: DataLoader        Training data loader.
    val_loader  : DataLoader        Validation data loader.
    cfg         : DotDict           Project config.
    model_name  : str               Used for checkpoint naming.
    resume_from : Path | None       Resume training from a checkpoint.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        cfg,
        model_name: str = "model",
        resume_from: Optional[Path] = None,
    ) -> None:
        self.model        = model
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.cfg          = cfg
        self.model_name   = model_name

        # Device
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        logger.info(f"Using device: {self.device}")
        self.model.to(self.device)

        # Loss
        self.criterion = build_loss(cfg).to(self.device)

        # Optimizer — 4-group differential learning rates
        # Transformer / pretrained layers need much lower LR than random-init layers.
        lr = float(cfg.training.lr)         # base: 3e-5
        self.base_lr = lr
        clip_scale       = float(getattr(cfg.training, "clip_lr_scale",       0.033))
        cross_attn_scale = float(getattr(cfg.training, "cross_attn_lr_scale", 0.167))
        film_scale       = float(getattr(cfg.training, "film_lr_scale",       0.333))

        clip_params, cross_attn_params, film_params, rest_params = [], [], [], []

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if "image_encoder.clip" in name:
                clip_params.append(param)          # unfrozen CLIP blocks → 1e-6
            elif "cross_fusion" in name:
                cross_attn_params.append(param)    # cross-attention layers → 5e-6
            elif "film" in name.lower() or "gamma" in name or "beta" in name:
                film_params.append(param)          # FiLM scale/shift → 1e-5
            else:
                rest_params.append(param)          # U-Net body → base LR

        param_groups = [{"params": rest_params, "lr": lr, "name": "unet"}]
        if cross_attn_params:
            param_groups.append({"params": cross_attn_params,
                                  "lr": lr * cross_attn_scale, "name": "cross_attn"})
        if film_params:
            param_groups.append({"params": film_params,
                                  "lr": lr * film_scale, "name": "film"})
        if clip_params:
            param_groups.append({"params": clip_params,
                                  "lr": lr * clip_scale, "name": "clip"})

        self.optimizer = AdamW(
            param_groups,
            weight_decay=float(cfg.training.weight_decay),
        )
        logger.info(
            f"Param groups: unet={len(rest_params)} │ "
            f"cross_attn={len(cross_attn_params)} │ "
            f"film={len(film_params)} │ "
            f"clip={len(clip_params)} "
            f"| LRs ≈ {lr:.0e} / {lr*cross_attn_scale:.0e} / "
            f"{lr*film_scale:.0e} / {lr*clip_scale:.0e}"
        )

        # Scheduler
        self.scheduler = self._build_scheduler()

        # Metrics
        self.train_acc = MetricAccumulator(
            threshold=cfg.evaluation.threshold,
            hausdorff_percentile=cfg.evaluation.hausdorff_percentile,
            compute_hausdorff=False,   # skip HD during training (slow)
        )
        self.val_acc = MetricAccumulator(
            threshold=cfg.evaluation.threshold,
            hausdorff_percentile=cfg.evaluation.hausdorff_percentile,
            compute_hausdorff=True,
        )

        # Logging
        log_dir = Path(cfg.paths.logs) / model_name
        self.writer = SummaryWriter(log_dir=str(log_dir))
        self.ckpt_dir = Path(cfg.paths.checkpoints)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        # Training state
        self.start_epoch   = 0
        self.best_dice     = 0.0
        self.patience_ctr  = 0
        self.history: Dict[str, List[float]] = {
            "train_loss": [], "val_loss": [],
            "train_dice": [], "val_dice": [],
            "train_iou":  [], "val_iou":  [],
            "val_precision": [], "val_recall": [],
            "val_hausdorff": [],
        }

        self.grad_accum = int(cfg.training.gradient_accumulation_steps)
        self.clip_norm  = float(cfg.training.clip_grad_norm)
        self.num_epochs = int(cfg.training.num_epochs)
        self.patience   = int(cfg.training.early_stopping_patience)
        self.ckpt_every = int(cfg.training.checkpoint_every)
        self.warmup_ep  = int(cfg.training.warmup_epochs)

        # Resume
        if resume_from and Path(resume_from).exists():
            state = load_checkpoint(
                self.model, Path(resume_from),
                self.optimizer, self.scheduler, str(self.device)
            )
            self.start_epoch = state["epoch"] + 1
            self.best_dice   = state["metrics"].get("val_dice", 0.0)
            logger.info(f"Resumed from epoch {self.start_epoch}")

    # ─── Scheduler ───────────────────────────────────────────────────────────

    def _build_scheduler(self):
        sched_type = str(self.cfg.training.lr_scheduler).lower()
        if sched_type == "cosine":
            return CosineAnnealingLR(
                self.optimizer,
                T_max=self.cfg.training.num_epochs,
                eta_min=1e-6,
            )
        elif sched_type == "step":
            return StepLR(self.optimizer, step_size=10, gamma=0.5)
        elif sched_type == "plateau":
            return ReduceLROnPlateau(
                self.optimizer, mode="max", patience=5, factor=0.5, verbose=True
            )
        else:
            raise ValueError(f"Unknown scheduler: {sched_type}")

    # ─── Warmup LR ───────────────────────────────────────────────────────────

    def _warmup_lr(self, epoch: int) -> None:
        """Linear warmup for the first warmup_epochs epochs."""
        if epoch < self.warmup_ep:
            scale = (epoch + 1) / max(self.warmup_ep, 1)
            for pg in self.optimizer.param_groups:
                pg["lr"] = self.base_lr * scale
        elif epoch == self.warmup_ep:
            # Restore base LR so the scheduler picks it up cleanly
            for pg in self.optimizer.param_groups:
                pg["lr"] = self.base_lr

    # ─── One training epoch ───────────────────────────────────────────────────

    def _train_epoch(self, epoch: int) -> float:
        self.model.train()
        self.train_acc.reset()
        total_loss = 0.0
        n_batches  = len(self.train_loader)

        pbar = tqdm(
            enumerate(self.train_loader),
            total=n_batches,
            desc=f"[Train] Epoch {epoch+1}/{self.num_epochs}",
            leave=False,
        )

        self.optimizer.zero_grad()

        for step, batch in pbar:
            batch = {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            targets = batch["mask"]

            # Forward
            logits = self.model(batch)
            seg_loss   = self.criterion(logits, targets)
            # Attention entropy regularization (Model 3 only; 0.0 no-op for Models 1 & 2)
            attn_weight = float(getattr(self.cfg.loss, "attn_entropy_weight", 0.001))
            attn_loss   = compute_attention_entropy_loss(
                self.model, weight=attn_weight,
            )
            if attn_loss.device.type == "cpu" and str(self.device) != "cpu":
                attn_loss = attn_loss.to(self.device)
            loss = (seg_loss + attn_loss) / self.grad_accum
            loss.backward()

            total_loss += loss.item() * self.grad_accum

            # Gradient accumulation step
            if (step + 1) % self.grad_accum == 0 or (step + 1) == n_batches:
                if self.clip_norm > 0:
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.clip_norm
                    )
                self.optimizer.step()
                self.optimizer.zero_grad()

            # Update metrics (no-grad)
            with torch.no_grad():
                self.train_acc.update(logits, targets)

            pbar.set_postfix(loss=f"{loss.item() * self.grad_accum:.4f}")

        return total_loss / n_batches

    # ─── One validation epoch ────────────────────────────────────────────────

    @torch.no_grad()
    def _val_epoch(self, epoch: int) -> float:
        self.model.eval()
        self.val_acc.reset()
        total_loss = 0.0
        n_batches  = len(self.val_loader)
        pred_pos_pixels = 0
        gt_pos_pixels   = 0
        total_pixels    = 0

        pbar = tqdm(
            self.val_loader,
            desc=f"[Val]   Epoch {epoch+1}/{self.num_epochs}",
            leave=False,
        )

        for batch in pbar:
            batch = {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            targets = batch["mask"]
            logits  = self.model(batch)
            loss    = self.criterion(logits, targets)
            total_loss += loss.item()
            self.val_acc.update(logits, targets)

            # Track positive pixel ratios
            preds_bin = (torch.sigmoid(logits).squeeze(1) >= 0.5)
            pred_pos_pixels += preds_bin.sum().item()
            gt_pos_pixels   += targets.bool().sum().item()
            total_pixels    += targets.numel()

        # Store for monitoring
        self._last_pred_pos_ratio = pred_pos_pixels / max(total_pixels, 1)
        self._last_gt_pos_ratio   = gt_pos_pixels   / max(total_pixels, 1)

        return total_loss / n_batches

    # ─── Main train loop ──────────────────────────────────────────────────────

    def train(self) -> Dict[str, List[float]]:
        """
        Run the full training loop.

        Returns
        -------
        history : dict of metric lists for plotting.
        """
        logger.info(
            f"\n{'='*60}\n"
            f"  Training {self.model_name}\n"
            f"  Epochs  : {self.num_epochs}\n"
            f"  Device  : {self.device}\n"
            f"{'='*60}"
        )

        for epoch in range(self.start_epoch, self.num_epochs):
            t0 = time.time()

            # Warmup
            self._warmup_lr(epoch)

            # Train
            train_loss = self._train_epoch(epoch)
            train_metrics = self.train_acc.compute()

            # Validate
            val_loss = self._val_epoch(epoch)
            val_metrics = self.val_acc.compute()

            # LR scheduler step
            sched_type = str(self.cfg.training.lr_scheduler).lower()
            if sched_type == "plateau":
                self.scheduler.step(val_metrics["dice"])
            elif epoch >= self.warmup_ep:
                self.scheduler.step()

            cur_lr = self.optimizer.param_groups[0]["lr"]
            elapsed = time.time() - t0

            # ── Positive pixel monitoring ─────────────────────────────────
            pred_pos = getattr(self, "_last_pred_pos_ratio", float("nan"))
            gt_pos   = getattr(self, "_last_gt_pos_ratio",   float("nan"))

            # ── Attention entropy (Model 3 only) ──────────────────────────
            attn_entropy_str = ""
            if hasattr(self.model, "get_attn_weights"):
                attn = self.model.get_attn_weights()
                if attn is not None:
                    p = attn / (attn.sum(dim=-1, keepdim=True) + 1e-10)
                    H = -(p * torch.log(p + 1e-10)).sum(dim=-1).mean().item()
                    attn_entropy_str = f"  AttnH {H:.2f}"
                    self.writer.add_scalar("Model3/attn_entropy", H, epoch)
                    self.writer.add_scalar("Model3/pred_pos_ratio", pred_pos, epoch)

            # ── Safe HD95 display (NaN = all-background predictions) ──────
            hd95_val = val_metrics.get("hausdorff", float("nan"))
            hd95_str = (f"{hd95_val:.2f}"
                        if not (hd95_val != hd95_val or hd95_val == float("inf"))
                        else "---")

            # ── Log to console ────────────────────────────────────────────
            print(
                f"Epoch [{epoch+1:03d}/{self.num_epochs}]  "
                f"Loss {train_loss:.4f}/{val_loss:.4f}  "
                f"Dice {train_metrics['dice']:.4f}/{val_metrics['dice']:.4f}  "
                f"IoU {val_metrics['iou']:.4f}  "
                f"HD95 {hd95_str}  "
                f"Pos {pred_pos:.3f}/{gt_pos:.3f}"
                f"{attn_entropy_str}  "
                f"LR {cur_lr:.2e}  [{elapsed:.1f}s]"
            )

            # ── Early collapse detection ──────────────────────────────────
            if epoch > 0 and len(self.history["val_dice"]) > 0:
                prev_dice = self.history["val_dice"][-1]
                if val_metrics["dice"] < prev_dice - 0.15:
                    logger.warning(
                        f"  ⚠ Dice dropped {prev_dice:.4f} → {val_metrics['dice']:.4f}. "
                        f"Halving all LRs."
                    )
                    for pg in self.optimizer.param_groups:
                        pg["lr"] *= 0.5

            # ── TensorBoard ───────────────────────────────────────────────
            self.writer.add_scalar("Loss/train", train_loss, epoch)
            self.writer.add_scalar("Loss/val",   val_loss,   epoch)
            for k, v in train_metrics.items():
                self.writer.add_scalar(f"Train/{k}", v, epoch)
            for k, v in val_metrics.items():
                self.writer.add_scalar(f"Val/{k}", v, epoch)
            self.writer.add_scalar("LR", cur_lr, epoch)

            # ── History ───────────────────────────────────────────────────
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["train_dice"].append(train_metrics["dice"])
            self.history["val_dice"].append(val_metrics["dice"])
            self.history["train_iou"].append(train_metrics["iou"])
            self.history["val_iou"].append(val_metrics["iou"])
            self.history["val_precision"].append(val_metrics["precision"])
            self.history["val_recall"].append(val_metrics["recall"])
            self.history["val_hausdorff"].append(
                val_metrics.get("hausdorff", float("nan"))
            )

            # ── Checkpoint ────────────────────────────────────────────────
            is_best = val_metrics["dice"] > self.best_dice
            if is_best:
                self.best_dice   = val_metrics["dice"]
                self.patience_ctr = 0
            else:
                self.patience_ctr += 1

            if is_best or (epoch + 1) % self.ckpt_every == 0:
                ckpt_path = (
                    self.ckpt_dir
                    / f"{self.model_name}_ep{epoch+1:03d}.pth"
                )
                save_checkpoint(
                    self.model, self.optimizer, self.scheduler,
                    epoch, val_metrics, ckpt_path, is_best=is_best
                )

            # ── Early stopping ────────────────────────────────────────────
            if self.patience_ctr >= self.patience:
                logger.info(
                    f"Early stopping triggered after "
                    f"{self.patience_ctr} epochs without improvement."
                )
                break

        self.writer.close()
        return self.history
