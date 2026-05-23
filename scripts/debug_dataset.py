"""
scripts/debug_dataset.py
─────────────────────────────────────────────────────────────────────────────
Debugging utilities to verify correct dataset loading:
  - Print dataset statistics
  - Verify image–mask–text alignment
  - Visualise sample slices
  - Inspect text reports
  - Print tensor shapes

Usage:
  python scripts/debug_dataset.py
  python scripts/debug_dataset.py --split val --n-samples 4
─────────────────────────────────────────────────────────────────────────────
"""
import sys
import os
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from configs.config_loader import load_config
from data.dataset import BraTSMultimodalDataset


def parse_args():
    p = argparse.ArgumentParser(description="Debug BraTS dataset loading.")
    p.add_argument("--split", default="train", choices=["train", "val"])
    p.add_argument("--n-samples", type=int, default=4)
    p.add_argument("--config", default=None)
    p.add_argument("--save-dir", default=None,
                   help="Directory to save debug visualisations.")
    return p.parse_args()


def main():
    args = parse_args()
    cfg  = load_config(args.config)

    flair_root = Path(cfg.paths.flair_root)
    text_root  = Path(cfg.paths.text_root)

    print(f"\n{'='*60}")
    print(f"  Debugging {args.split.upper()} split")
    print(f"  FLAIR root: {flair_root}")
    print(f"  Text root : {text_root}")
    print(f"{'='*60}\n")

    ds = BraTSMultimodalDataset(
        flair_split_dir = flair_root / args.split,
        text_root       = text_root,
        split           = args.split,
        cfg             = cfg,
        use_precomputed_text = cfg.dataset.use_precomputed_text,
        augment         = False,
    )

    ds.print_stats()

    # ── Verify N random samples ───────────────────────────────────────────
    indices = np.random.choice(len(ds), min(args.n_samples, len(ds)), replace=False)
    print(f"\nVerifying {len(indices)} random samples:\n")
    for idx in indices:
        ds.verify_sample(int(idx))

    # ── Dataloader test ───────────────────────────────────────────────────
    from torch.utils.data import DataLoader
    loader = DataLoader(ds, batch_size=2, shuffle=True, num_workers=0)
    batch  = next(iter(loader))

    print("\n── Batch shapes ──────────────────────────────────────────")
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k:15s}: {tuple(v.shape)}  dtype={v.dtype}")
        else:
            print(f"  {k:15s}: {v}")

    # ── Visualise ────────────────────────────────────────────────────────
    save_dir = Path(args.save_dir) if args.save_dir else Path(cfg.paths.results) / "debug"
    save_dir.mkdir(parents=True, exist_ok=True)

    B    = batch["image"].shape[0]
    fig, axes = plt.subplots(B, 2, figsize=(8, 4 * B))
    if B == 1:
        axes = axes[np.newaxis, :]

    for i in range(B):
        img_np = batch["image"][i, 0].numpy()
        msk_np = batch["mask"][i].numpy()
        img_n  = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)

        axes[i, 0].imshow(img_n, cmap="gray")
        axes[i, 0].set_title(
            f"{batch['patient_id'][i]}  slice={batch['slice_idx'][i].item()}",
            fontsize=9,
        )
        axes[i, 0].axis("off")

        axes[i, 1].imshow(img_n, cmap="gray")
        axes[i, 1].imshow(
            np.ma.masked_where(msk_np < 0.5, msk_np),
            cmap="Reds", alpha=0.6,
        )
        axes[i, 1].set_title("GT Mask overlay", fontsize=9)
        axes[i, 1].axis("off")

    fig.suptitle(f"Debug — {args.split} split samples", fontsize=12)
    fig.tight_layout()
    out = save_dir / f"debug_{args.split}.png"
    fig.savefig(out, dpi=100)
    plt.close(fig)
    print(f"\nDebug visualisation saved → {out}")


if __name__ == "__main__":
    main()
