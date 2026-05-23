"""
visualization/plot_training.py
─────────────────────────────────────────────────────────────────────────────
Generate publication-quality training curve plots from history dicts.

Plots:
  1. Loss curves (train + val)
  2. Dice Score curves (train + val)
  3. IoU curves (train + val)
  4. Combined multi-panel figure
─────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


STYLE = {
    "figure.dpi":         150,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          True,
    "grid.alpha":         0.35,
    "font.family":        "DejaVu Sans",
    "axes.titlesize":     13,
    "axes.labelsize":     11,
    "legend.fontsize":    10,
    "lines.linewidth":    2.0,
}


def save_history(history: Dict, path: Path) -> None:
    """Serialise training history to JSON."""
    with open(path, "w") as f:
        json.dump(history, f, indent=2, default=lambda x: None)


def load_history(path: Path) -> Dict:
    with open(path) as f:
        return json.load(f)


def _smooth(values: List[float], window: int = 3) -> np.ndarray:
    """Simple moving average for smoother curves."""
    arr = np.array(values, dtype=float)
    if len(arr) < window:
        return arr
    kernel = np.ones(window) / window
    padded = np.pad(arr, (window // 2, window // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")[: len(arr)]


def plot_training_curves(
    history: Dict[str, List],
    save_dir: Path,
    model_name: str = "model",
    smooth: bool = True,
) -> None:
    """
    Render and save training curve figures.

    Parameters
    ----------
    history   : dict returned by Trainer.train()
    save_dir  : directory to save figures
    model_name: prefix for filenames
    smooth    : apply moving-average smoothing
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    with plt.rc_context(STYLE):
        epochs = range(1, len(history["train_loss"]) + 1)

        # ── 1. Loss ───────────────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(8, 4))
        tl = history["train_loss"]
        vl = history["val_loss"]
        ax.plot(epochs, _smooth(tl) if smooth else tl, label="Train Loss", color="#e07b54")
        ax.plot(epochs, _smooth(vl) if smooth else vl, label="Val Loss",   color="#5470c6")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title(f"{model_name} — Loss Curves")
        ax.legend()
        fig.tight_layout()
        fig.savefig(save_dir / f"{model_name}_loss.png")
        plt.close(fig)

        # ── 2. Dice ───────────────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(8, 4))
        td = history["train_dice"]
        vd = history["val_dice"]
        ax.plot(epochs, _smooth(td) if smooth else td, label="Train Dice", color="#e07b54")
        ax.plot(epochs, _smooth(vd) if smooth else vd, label="Val Dice",   color="#5470c6")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Dice Coefficient")
        ax.set_title(f"{model_name} — Dice Score")
        ax.set_ylim(0, 1)
        ax.legend()
        fig.tight_layout()
        fig.savefig(save_dir / f"{model_name}_dice.png")
        plt.close(fig)

        # ── 3. IoU ────────────────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(8, 4))
        ti = history["train_iou"]
        vi = history["val_iou"]
        ax.plot(epochs, _smooth(ti) if smooth else ti, label="Train IoU", color="#e07b54")
        ax.plot(epochs, _smooth(vi) if smooth else vi, label="Val IoU",   color="#5470c6")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("IoU")
        ax.set_title(f"{model_name} — IoU Score")
        ax.set_ylim(0, 1)
        ax.legend()
        fig.tight_layout()
        fig.savefig(save_dir / f"{model_name}_iou.png")
        plt.close(fig)

        # ── 4. Combined 3-panel ───────────────────────────────────────────
        fig, axes = plt.subplots(1, 3, figsize=(18, 4))
        titles  = ["Loss", "Dice Coefficient", "IoU"]
        y_train = [history["train_loss"], history["train_dice"], history["train_iou"]]
        y_val   = [history["val_loss"],   history["val_dice"],   history["val_iou"]]
        ylims   = [None, (0, 1), (0, 1)]

        for ax, title, yt, yv, ylim in zip(axes, titles, y_train, y_val, ylims):
            ax.plot(epochs, _smooth(yt) if smooth else yt, color="#e07b54", label="Train")
            ax.plot(epochs, _smooth(yv) if smooth else yv, color="#5470c6", label="Val")
            ax.set_title(title)
            ax.set_xlabel("Epoch")
            if ylim:
                ax.set_ylim(*ylim)
            ax.legend()

        fig.suptitle(f"{model_name} — Training Summary", fontsize=14, y=1.02)
        fig.tight_layout()
        fig.savefig(save_dir / f"{model_name}_training_summary.png", bbox_inches="tight")
        plt.close(fig)

    print(f"Training curves saved to {save_dir}")
