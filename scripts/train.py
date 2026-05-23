"""
scripts/train.py
─────────────────────────────────────────────────────────────────────────────
Main training entry-point for a single model.

Usage (inside brain_tumor_vlm conda env):
  python scripts/train.py --model model1_image_only
  python scripts/train.py --model model2_concat_fusion
  python scripts/train.py --model model3_cross_attention
  python scripts/train.py --model model1_image_only --resume checkpoints/model1_ep010.pth
─────────────────────────────────────────────────────────────────────────────
"""
import sys
import os
import argparse
import logging
import random

import numpy as np
import torch

# ── Ensure project root is on the path ───────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from configs.config_loader import load_config
from data.dataloader import build_dataloaders
from models import build_model
from training.trainer import Trainer
from visualization.plot_training import plot_training_curves, save_history


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a brain tumour segmentation model."
    )
    parser.add_argument(
        "--model", type=str, required=True,
        choices=["model1_image_only", "model2_concat_fusion", "model3_cross_attention"],
        help="Which model variant to train.",
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to config.yaml (default: configs/config.yaml).",
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Path to checkpoint to resume training from.",
    )
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="Override number of training epochs.",
    )
    parser.add_argument(
        "--lr", type=float, default=None,
        help="Override learning rate.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # ── Config ────────────────────────────────────────────────────────────
    cfg = load_config(args.config)
    if args.epochs:
        cfg.training.num_epochs = args.epochs
    if args.lr:
        cfg.training.lr = args.lr

    set_seed(cfg.training.seed)
    logger.info(f"Config loaded | Seed={cfg.training.seed}")

    # ── Data ──────────────────────────────────────────────────────────────
    train_loader, val_loader = build_dataloaders(cfg)

    # ── Model ─────────────────────────────────────────────────────────────
    model = build_model(args.model, cfg)

    # ── Trainer ───────────────────────────────────────────────────────────
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        cfg=cfg,
        model_name=args.model,
        resume_from=args.resume,
    )

    # ── Train ─────────────────────────────────────────────────────────────
    history = trainer.train()

    # ── Save history & plots ──────────────────────────────────────────────
    from pathlib import Path
    results_dir = Path(cfg.paths.results) / "plots" / args.model
    results_dir.mkdir(parents=True, exist_ok=True)

    save_history(history, results_dir / "history.json")
    plot_training_curves(history, results_dir, model_name=args.model)

    logger.info(
        f"\nTraining complete. Best Val Dice: {trainer.best_dice:.4f}\n"
        f"Plots saved to {results_dir}"
    )


if __name__ == "__main__":
    main()
