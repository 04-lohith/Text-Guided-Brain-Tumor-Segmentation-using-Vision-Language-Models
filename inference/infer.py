"""
inference/infer.py
─────────────────────────────────────────────────────────────────────────────
Standalone inference pipeline for a single patient.

Given:
  - A patient .npy volume (128,128,128)
  - Optionally a text report path or pre-computed embedding .npy
  - A trained model checkpoint

Outputs:
  - Per-slice segmentation predictions (numpy array + visualisation)
  - Optional combined 3-D mask

Usage:
  python inference/infer.py \
    --model model3_cross_attention \
    --checkpoint checkpoints/model3_best.pth \
    --image FLAIR_BRATS2020/train/images/image_0.npy \
    --text TextBRats/TextBraTSData/BraTS20_Training_001/BraTS20_Training_001_flair_text.npy \
    --output results/inference/patient_001_pred.npy
─────────────────────────────────────────────────────────────────────────────
"""
import sys
import os
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from configs.config_loader import load_config
from models import build_model
from training.trainer import load_checkpoint


def load_text_embedding(text_path: str) -> np.ndarray:
    """Load .npy or .txt text embedding → (768,)."""
    p = Path(text_path)
    if p.suffix == ".npy":
        emb = np.load(str(p)).astype(np.float32)
        if emb.ndim == 3:
            emb = emb[0]
        return emb.mean(axis=0)   # (768,)
    elif p.suffix == ".txt":
        from data.dataset import BioClinicalBERTEncoder
        enc = BioClinicalBERTEncoder()
        with open(p, "r", encoding="utf-8") as f:
            text = f.read().strip()
        return enc.encode(text)
    else:
        raise ValueError(f"Unsupported text file format: {p.suffix}")


def run_inference(
    model_name:    str,
    checkpoint:    str,
    image_path:    str,
    text_path:     str | None = None,
    output_path:   str | None = None,
    config_path:   str | None = None,
    threshold:     float = 0.5,
    visualise:     bool  = True,
) -> np.ndarray:
    """
    Run full inference on one patient volume.

    Returns
    -------
    pred_vol : (128, 128, 128) binary numpy array
    """
    cfg = load_config(config_path)

    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    # Load model
    model = build_model(model_name, cfg)
    load_checkpoint(model, Path(checkpoint), device=device)
    model.to(device)
    model.eval()

    # Load volume
    vol = np.load(image_path).astype(np.float32)   # (H, W, D)
    H, W, D = vol.shape
    pred_vol = np.zeros((H, W, D), dtype=np.float32)

    # Load text embedding (one per patient)
    if text_path and model_name != "model1_image_only":
        text_emb = load_text_embedding(text_path)
        text_t   = torch.from_numpy(text_emb).float().unsqueeze(0).to(device)  # (1, 768)
    else:
        text_t = torch.zeros(1, 768, device=device)

    # Slice-wise inference
    with torch.no_grad():
        for s in range(D):
            img_s = vol[:, :, s]                                # (H, W)
            # Normalise
            p1, p99 = np.percentile(img_s, [1, 99])
            img_s   = np.clip(img_s, p1, p99)
            std     = img_s.std() + 1e-8
            img_s   = (img_s - img_s.mean()) / std

            img_t = torch.from_numpy(img_s[None, None]).float().to(device)  # (1,1,H,W)

            batch = {"image": img_t, "text_emb": text_t}
            logits = model(batch)                               # (1,1,H,W)
            pred   = torch.sigmoid(logits).squeeze().cpu().numpy()
            pred_vol[:, :, s] = pred

    # Threshold
    binary_vol = (pred_vol >= threshold).astype(np.float32)

    # Save
    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(out), binary_vol)
        print(f"Prediction saved → {out}")

    # Visualise mid-slices
    if visualise:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        mid = D // 2
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        for ax, (title, data) in zip(axes, [
            ("FLAIR (raw)",     vol[:, :, mid]),
            ("Pred (soft)",    pred_vol[:, :, mid]),
            ("Pred (binary)",  binary_vol[:, :, mid]),
        ]):
            ax.imshow(data, cmap="gray" if "FLAIR" in title else "hot")
            ax.set_title(title, fontsize=11)
            ax.axis("off")
        fig.suptitle(f"Inference — {model_name} | slice {mid}", fontsize=12)
        fig.tight_layout()
        viz_path = (Path(output_path).with_suffix(".png")
                    if output_path else Path("inference_preview.png"))
        fig.savefig(viz_path, dpi=100)
        plt.close(fig)
        print(f"Visualisation saved → {viz_path}")

    return binary_vol


def main():
    parser = argparse.ArgumentParser(description="Brain tumour segmentation inference.")
    parser.add_argument("--model",      required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image",      required=True)
    parser.add_argument("--text",       default=None)
    parser.add_argument("--output",     default=None)
    parser.add_argument("--config",     default=None)
    parser.add_argument("--threshold",  type=float, default=0.5)
    parser.add_argument("--no-viz",     action="store_true")
    args = parser.parse_args()

    run_inference(
        model_name  = args.model,
        checkpoint  = args.checkpoint,
        image_path  = args.image,
        text_path   = args.text,
        output_path = args.output,
        config_path = args.config,
        threshold   = args.threshold,
        visualise   = not args.no_viz,
    )


if __name__ == "__main__":
    main()
