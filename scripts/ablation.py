"""
scripts/ablation.py
─────────────────────────────────────────────────────────────────────────────
Ablation study: compare all three model variants and produce:
  - Combined Dice/IoU comparison bar charts
  - Performance table (printed + saved as CSV)

Usage (all three checkpoints must exist):
  python scripts/ablation.py
  python scripts/ablation.py --config configs/config.yaml
─────────────────────────────────────────────────────────────────────────────
"""
import sys
import os
import argparse
import logging
from pathlib import Path

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from configs.config_loader import load_config
from data.dataloader import build_dataloaders
from models import build_model
from training.trainer import load_checkpoint
from evaluation.metrics import MetricAccumulator

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
logger = logging.getLogger(__name__)


MODEL_NAMES = [
    "model1_image_only",
    "model2_concat_fusion",
    "model3_cross_attention",
]

DISPLAY_NAMES = {
    "model1_image_only":      "Model 1\nImage-Only U-Net",
    "model2_concat_fusion":   "Model 2\nConcat Fusion",
    "model3_cross_attention": "Model 3\nCross-Attn Fusion",
}


def evaluate_model(model, loader, device, cfg) -> dict:
    acc = MetricAccumulator(
        threshold=cfg.evaluation.threshold,
        hausdorff_percentile=cfg.evaluation.hausdorff_percentile,
        compute_hausdorff=True,
    )
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch_dev = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            logits = model(batch_dev)
            acc.update(logits, batch_dev["mask"])
    return acc.compute()


def plot_ablation_bar(
    results:  dict,
    save_dir: Path,
    metric:   str = "dice",
    ylabel:   str = "Dice Coefficient",
) -> None:
    names   = [DISPLAY_NAMES[m] for m in MODEL_NAMES if m in results]
    values  = [results[m][metric] for m in MODEL_NAMES if m in results]
    all_colours = ["#91cc75", "#5470c6", "#ee6666"]
    colours = all_colours[: len(values)]

    if not values:
        print(f"No results for {metric}, skipping plot.")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(names, values, color=colours, edgecolor="white", width=0.5)

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{val:.4f}",
            ha="center", va="bottom", fontsize=11, fontweight="bold",
        )

    ax.set_ylabel(ylabel, fontsize=12)
    ymax = max(values) if max(values) > 0 else 1.0
    ax.set_ylim(0, min(1.05, ymax * 1.2))
    ax.set_title(f"Ablation Study — {ylabel}", fontsize=13)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_dir / f"ablation_{metric}.png", dpi=120)
    plt.close(fig)
    print(f"Ablation {metric} plot saved → {save_dir / f'ablation_{metric}.png'}")


def plot_ablation_radar(results: dict, save_dir: Path) -> None:
    """Spider / radar chart comparing all 5 metrics across models."""
    metrics = ["dice", "iou", "precision", "recall"]
    labels  = ["Dice", "IoU", "Precision", "Recall"]
    N       = len(metrics)

    angles  = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    colours = ["#91cc75", "#5470c6", "#ee6666"]
    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"polar": True})

    for (mname, colour) in zip(MODEL_NAMES, colours):
        if mname not in results:
            continue
        vals = [results[mname].get(m, 0) for m in metrics] + [results[mname].get(metrics[0], 0)]
        ax.plot(angles, vals, "o-", color=colour, lw=2, label=DISPLAY_NAMES[mname].replace("\n", " "))
        ax.fill(angles, vals, alpha=0.15, color=colour)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylim(0, 1)
    ax.set_title("Ablation Study — Radar Chart", y=1.08, fontsize=13)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15))
    fig.tight_layout()
    fig.savefig(save_dir / "ablation_radar.png", dpi=120)
    plt.close(fig)
    print(f"Radar chart saved → {save_dir / 'ablation_radar.png'}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=None)
    p.add_argument("--skip-missing", action="store_true",
                   help="Skip models whose checkpoint is missing.")
    return p.parse_args()


def main():
    args   = parse_args()
    cfg    = load_config(args.config)

    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    _, val_loader = build_dataloaders(cfg)

    ckpt_dir = Path(cfg.paths.checkpoints)
    save_dir = Path(cfg.paths.results) / "tables"
    save_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = Path(cfg.paths.results) / "plots" / "ablation"
    plot_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    for mname in MODEL_NAMES:
        ckpt_path = ckpt_dir / f"{mname}_best.pth"
        if not ckpt_path.exists():
            if args.skip_missing:
                logger.warning(f"Checkpoint not found: {ckpt_path} — skipping.")
                continue
            else:
                raise FileNotFoundError(
                    f"Checkpoint not found: {ckpt_path}. "
                    "Train this model first or use --skip-missing."
                )

        logger.info(f"Evaluating {mname} …")
        model = build_model(mname, cfg)
        load_checkpoint(model, ckpt_path, device=device)
        model.to(device)
        metrics = evaluate_model(model, val_loader, device, cfg)
        results[mname] = metrics
        logger.info(
            f"  {mname}: Dice={metrics['dice']:.4f}  IoU={metrics['iou']:.4f}  "
            f"HD95={metrics.get('hausdorff', float('nan')):.2f}"
        )

    if not results:
        logger.error("No models evaluated.")
        return

    # ── Performance Table ─────────────────────────────────────────────────
    rows = []
    for mname in MODEL_NAMES:
        if mname not in results:
            continue
        m = results[mname]
        rows.append({
            "Model":      DISPLAY_NAMES[mname].replace("\n", " "),
            "Dice":       f"{m['dice']:.4f}",
            "IoU":        f"{m['iou']:.4f}",
            "Precision":  f"{m['precision']:.4f}",
            "Recall":     f"{m['recall']:.4f}",
            "HD95 (mm)":  f"{m.get('hausdorff', float('nan')):.2f}",
        })
    df = pd.DataFrame(rows)
    print("\n" + "=" * 65)
    print(df.to_string(index=False))
    print("=" * 65 + "\n")
    table_path = save_dir / "ablation_results.csv"
    df.to_csv(table_path, index=False)
    print(f"Table saved → {table_path}")

    # ── Plots ────────────────────────────────────────────────────────────
    plot_ablation_bar(results, plot_dir, "dice",      "Dice Coefficient")
    plot_ablation_bar(results, plot_dir, "iou",       "IoU Score")
    plot_ablation_bar(results, plot_dir, "precision", "Precision")
    plot_ablation_bar(results, plot_dir, "recall",    "Recall")
    # HD95 — lower is better, so we remap to "hausdorff" key
    hd95_results = {k: {"hausdorff": v.get("hausdorff", float("nan"))} for k, v in results.items()}
    plot_ablation_bar(hd95_results, plot_dir, "hausdorff", "HD95 (mm)  ← lower is better")
    plot_ablation_radar(results, plot_dir)


if __name__ == "__main__":
    main()
