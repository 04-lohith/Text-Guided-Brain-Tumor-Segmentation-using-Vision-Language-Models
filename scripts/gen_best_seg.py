"""
scripts/gen_best_seg.py
Generates segmentation_best.png — top-6 highest-Dice tumor slices from Model 3.
"""
import sys, os, torch
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.config_loader import load_config
from data.dataloader import build_dataloaders
from models import build_model
from training.trainer import load_checkpoint

cfg    = load_config(None)
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Device: {device}")

_, val_loader = build_dataloaders(cfg)

model = build_model("model3_cross_attention", cfg)
load_checkpoint(model, "checkpoints/model3_cross_attention_best.pth", device=device)
model.to(device).eval()

# ── Collect best samples ──────────────────────────────────────────────────
best = []

with torch.no_grad():
    for batch in val_loader:
        batch_dev = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
        logits = model(batch_dev)
        probs  = torch.sigmoid(logits).squeeze(1)
        preds  = (probs >= 0.5).float()
        gts    = batch_dev["mask"].float()
        imgs   = batch["image"].squeeze(1).cpu().numpy()

        for i in range(preds.shape[0]):
            gt   = gts[i].cpu().numpy()
            pred = preds[i].cpu().numpy()
            img  = imgs[i]

            if gt.sum() < 50:          # skip near-empty slices
                continue
            tp = ((gt == 1) & (pred == 1)).sum()
            fp = ((gt == 0) & (pred == 1)).sum()
            fn = ((gt == 1) & (pred == 0)).sum()
            dice = (2 * tp) / (2 * tp + fp + fn + 1e-6)

            if dice > 0.75:
                best.append((float(dice), img, gt, pred))

        if len(best) >= 40:
            break

best.sort(key=lambda x: -x[0])
best = best[:6]
print(f"Found {len(best)} samples. Best Dice = {best[0][0]:.4f}")

# ── Plot ──────────────────────────────────────────────────────────────────
n = len(best)
fig, axes = plt.subplots(n, 4, figsize=(14, 3.0 * n))
fig.suptitle(
    "Model 3 — Cross-Attention VLM: Representative Segmentation Results\n"
    "Green = True Positive  |  Red = False Positive  |  Yellow = False Negative",
    fontsize=12, fontweight="bold"
)

cols = ["FLAIR MRI", "Ground Truth", "Prediction", "Overlay"]
for c, title in enumerate(cols):
    axes[0, c].set_title(title, fontsize=10, fontweight="bold", pad=6)

for row, (dice, img, gt, pred) in enumerate(best):
    img_n = (img - img.min()) / (img.max() - img.min() + 1e-8)

    # Build RGBA overlay: TP=green, FP=red, FN=yellow
    overlay = np.zeros((*img_n.shape, 4), dtype=np.float32)
    tp_mask = (gt == 1) & (pred == 1)
    fp_mask = (gt == 0) & (pred == 1)
    fn_mask = (gt == 1) & (pred == 0)
    overlay[tp_mask] = [0.0, 0.9, 0.0, 0.7]   # green
    overlay[fp_mask] = [0.9, 0.0, 0.0, 0.7]   # red
    overlay[fn_mask] = [1.0, 1.0, 0.0, 0.7]   # yellow

    axes[row, 0].imshow(img_n, cmap="gray")
    axes[row, 1].imshow(img_n, cmap="gray")
    axes[row, 1].imshow(np.ma.masked_where(gt == 0, gt), cmap="Greens", alpha=0.6)
    axes[row, 2].imshow(img_n, cmap="gray")
    axes[row, 2].imshow(np.ma.masked_where(pred == 0, pred), cmap="Blues", alpha=0.6)
    axes[row, 3].imshow(img_n, cmap="gray")
    axes[row, 3].imshow(overlay)

    axes[row, 0].set_ylabel(f"Dice = {dice:.3f}", fontsize=9,
                            rotation=0, labelpad=65, va="center")
    for c in range(4):
        axes[row, c].axis("off")

plt.tight_layout()
out = Path("results/plots/model3_cross_attention/segmentation_best.png")
fig.savefig(out, dpi=130, bbox_inches="tight")
plt.close(fig)
print(f"Saved → {out}")
