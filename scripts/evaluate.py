"""
scripts/evaluate.py
─────────────────────────────────────────────────────────────────────────────
Evaluate a trained model on the validation set and generate all visualisations.

Usage:
  # Basic metrics only
  python scripts/evaluate.py --model model1_image_only \\
      --checkpoint checkpoints/model1_image_only_best.pth

  # Full visualisation suite (recommended)
  python scripts/evaluate.py --model model3_cross_attention \\
      --checkpoint checkpoints/model3_cross_attention_best.pth --all-viz

  # Choose Grad-CAM target function
  python scripts/evaluate.py --model model3_cross_attention \\
      --checkpoint checkpoints/model3_cross_attention_best.pth \\
      --all-viz --cam-mode gt_tumor
─────────────────────────────────────────────────────────────────────────────
"""
import sys
import os
import argparse
import logging
from pathlib import Path

import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from configs.config_loader import load_config
from data.dataloader import build_dataloaders
from models import build_model
from training.trainer import load_checkpoint
from evaluation.metrics import MetricAccumulator
from visualization.plot_segmentation import visualise_predictions
from visualization.gradcam import (
    GradCAM, GradCAMMode, _resolve_decoder_target, save_gradcam_figures,
)
from visualization.attention_maps import (
    plot_attention_avg_heads, plot_attention_per_head, plot_attention_head_entropy,
)
from visualization.error_analysis import (
    collect_per_slice_metrics,
    plot_dice_distribution,
    plot_worst_predictions,
)
from visualization.tsne_viz import extract_embeddings, plot_tsne

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained segmentation model.")
    parser.add_argument("--model", required=True,
                        choices=["model1_image_only", "model2_concat_fusion",
                                 "model3_cross_attention"])
    parser.add_argument("--checkpoint", required=True, help="Path to .pth checkpoint.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--all-viz", action="store_true",
                        help="Run all visualisations (slower).")
    parser.add_argument("--n-samples", type=int, default=6,
                        help="Number of samples for segmentation visualisation.")
    parser.add_argument(
        "--cam-mode",
        default=GradCAMMode.PRED_TUMOR,
        choices=[GradCAMMode.PRED_TUMOR, GradCAMMode.GT_TUMOR, GradCAMMode.TOP_K],
        help=(
            "Grad-CAM target function:\n"
            "  pred_tumor  – gradient w.r.t. predicted tumor pixels (default)\n"
            "  gt_tumor    – gradient w.r.t. ground-truth tumor region\n"
            "  top_k       – gradient w.r.t. top-K activation pixels\n"
        ),
    )
    return parser.parse_args()


def _find_tumor_batch(loader, max_search: int = 40):
    """
    Iterate the loader until a batch with a non-empty GT mask is found.
    Returns the first such batch or falls back to the very first batch.
    """
    first_batch = None
    for i, batch in enumerate(loader):
        if first_batch is None:
            first_batch = batch
        if batch["mask"].sum() > 0:
            return batch
        if i >= max_search:
            break
    return first_batch


def main():
    args = parse_args()
    cfg  = load_config(args.config)

    # ── Device ────────────────────────────────────────────────────────────
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    logger.info(f"Device: {device}")

    # ── Data (val only) ───────────────────────────────────────────────────
    _, val_loader = build_dataloaders(cfg)

    # ── Model ─────────────────────────────────────────────────────────────
    model = build_model(args.model, cfg)
    load_checkpoint(model, Path(args.checkpoint), device=device)
    model.to(device)
    model.eval()

    # ── Compute metrics ───────────────────────────────────────────────────
    acc = MetricAccumulator(
        threshold=cfg.evaluation.threshold,
        hausdorff_percentile=cfg.evaluation.hausdorff_percentile,
        compute_hausdorff=True,
    )

    logger.info("Running evaluation …")
    with torch.no_grad():
        for batch in val_loader:
            batch_dev = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            logits = model(batch_dev)
            acc.update(logits, batch_dev["mask"])

    results = acc.compute()
    print("\n" + "=" * 55)
    print(f"  Model       : {args.model}")
    print(f"  Checkpoint  : {args.checkpoint}")
    print(f"  Dice        : {results['dice']:.4f}")
    print(f"  IoU         : {results['iou']:.4f}")
    print(f"  Precision   : {results['precision']:.4f}")
    print(f"  Recall      : {results['recall']:.4f}")
    print(f"  HD95        : {results.get('hausdorff', float('nan')):.2f} mm")
    print("=" * 55 + "\n")

    viz_dir = Path(cfg.paths.results) / "plots" / args.model
    viz_dir.mkdir(parents=True, exist_ok=True)

    # ── Segmentation visualisation ────────────────────────────────────────
    logger.info("Generating segmentation visualisation …")

    # Find a batch that actually contains tumor pixels (avoids blank plots)
    sample_batch = _find_tumor_batch(val_loader)

    sample_batch_dev = {
        k: v.to(device) if isinstance(v, torch.Tensor) else v
        for k, v in sample_batch.items()
    }
    with torch.no_grad():
        logits_viz = model(sample_batch_dev)

    visualise_predictions(
        images    = sample_batch["image"],
        masks_gt  = sample_batch["mask"],
        logits    = logits_viz.cpu(),
        save_path = viz_dir / "segmentation_samples.png",
        patient_ids = list(sample_batch["patient_id"]),
        max_samples = args.n_samples,
    )

    if args.all_viz:
        # ── Decoder Grad-CAM ──────────────────────────────────────────────
        logger.info(f"Computing Decoder Grad-CAM  [mode={args.cam_mode}] …")
        target_layer = _resolve_decoder_target(model)
        logger.info(f"  CAM target layer: {target_layer.__class__.__name__}")
        gradcam = GradCAM(
            model,
            target_layer=target_layer,
            device=device,
            mode=args.cam_mode,
        )
        cam_maps = gradcam.compute(
            sample_batch_dev,
            gt_mask=sample_batch["mask"].to(device),
        )
        save_gradcam_figures(
            images     = sample_batch["image"],
            cam_maps   = cam_maps,
            masks_gt   = sample_batch["mask"],
            save_path  = viz_dir / "gradcam.png",
            patient_ids= list(sample_batch["patient_id"]),
            max_samples= args.n_samples,
            model_name = args.model,
            cam_mode   = args.cam_mode,
        )
        gradcam.remove_hooks()

        # ── Cross-attention visualisation (Model 3 only) ──────────────────
        is_model3 = args.model == "model3_cross_attention"
        if is_model3:
            logger.info("Generating cross-attention visualisations …")

            # Run one forward pass so attention weights are populated
            with torch.no_grad():
                _ = model(sample_batch_dev)

            plot_attention_avg_heads(
                model      = model,
                images     = sample_batch["image"],
                masks_gt   = sample_batch["mask"],
                save_path  = viz_dir / "attention_avg_heads.png",
                patient_ids= list(sample_batch["patient_id"]),
                model_name = args.model,
                max_samples= args.n_samples,
            )

            plot_attention_per_head(
                model      = model,
                images     = sample_batch["image"],
                save_path  = viz_dir / "attention_per_head.png",
                patient_ids= list(sample_batch["patient_id"]),
                model_name = args.model,
                sample_idx = 0,
            )

            plot_attention_head_entropy(
                model      = model,
                save_path  = viz_dir / "attention_head_entropy.png",
                model_name = args.model,
            )

        # ── Error analysis ────────────────────────────────────────────────
        logger.info("Running error analysis …")
        records = collect_per_slice_metrics(
            model, val_loader, device=device, max_batches=50,
        )
        plot_dice_distribution(records, viz_dir / "dice_distribution.png", args.model)
        plot_worst_predictions(
            records, viz_dir / "worst_predictions.png", model_name=args.model,
        )

        # ── t-SNE ─────────────────────────────────────────────────────────
        logger.info("Computing t-SNE embeddings …")
        img_embs, txt_embs, pids = extract_embeddings(
            model, val_loader, device=device, max_batches=30,
        )
        plot_tsne(
            img_embs, txt_embs, pids,
            save_path  = viz_dir / "tsne.png",
            model_name = args.model,
        )

    logger.info(f"Evaluation complete. Results saved to {viz_dir}")


if __name__ == "__main__":
    main()
