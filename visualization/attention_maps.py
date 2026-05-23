"""
visualization/attention_maps.py
─────────────────────────────────────────────────────────────────────────────
Cross-attention explainability for Model 3 (CrossAttentionFusionModel).

Extracts, reshapes, and visualises the attention weights stored in
model.cross_fusion.blocks[-1].last_attn_weights after every forward pass.

Cross-attention shape (TRUE token-wise design):
  (B, n_heads, N_patches, seq_len)
    B         : batch size
    n_heads   : number of attention heads (8)
    N_patches : number of CLIP image patches (ViT-B/32 → 49 patches)
    seq_len   : number of BERT text tokens (128)

For spatial visualisation we mean-pool over seq_len first →
  (B, n_heads, N_patches)  then reshape to a patch grid.

Outputs (all saved as .png):
  attention_avg_heads.png   – averaged across all heads
  attention_per_head.png    – 8 individual head maps
  attention_head_entropy.py – head diversity / entropy bar chart
─────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
import torch.nn.functional as F


# ─── Patch grid constants (ViT-B/32 on 224×224 after CLIP internal resize) ─
# CLIP ViT-B/32 patch size = 32 px, input = 224×224  →  7×7 = 49 patches
# NOTE: These are only used as documentation; all reshape logic is dynamic.
PATCH_SIZE = 32
PATCH_GRID = 7          # 224 // 32
N_PATCHES  = 49         # 7 * 7


def _extract_attn_weights(model) -> Optional[torch.Tensor]:
    """
    Pull the stored attention weights from the model's last forward pass.

    Returns
    -------
    attn : (B, n_heads, N_patches, 1)  or None if model isn't Model 3.
    """
    if not hasattr(model, "get_attn_weights"):
        return None
    return model.get_attn_weights()


def _attn_to_spatial(
    attn: torch.Tensor,             # (B, n_heads, N_patches, 1)
    img_h: int = 128,
    img_w: int = 128,
    patch_grid: int = None,         # auto-detected from N if None
) -> np.ndarray:
    """
    Reshape patch-level attention into a (B, n_heads, img_h, img_w) heatmap.

    The patch grid size is computed dynamically from the actual number of
    patches (N) so this works regardless of CLIP's internal resize behaviour.
    ViT-B/32 resizes inputs to 224×224 → 7×7 = 49 patches.

    Steps
    -----
    1.  Squeeze text dimension:  (B, H, N, 1) → (B, H, N)
    2.  Compute patch grid:      pg = int(round(sqrt(N)))
    3.  Reshape to patch grid:   (B, H, N) → (B, H, pg, pg)
    4.  Interpolate to img size: (B, H, pg, pg) → (B, H, img_h, img_w)
    5.  Normalise per head per sample → [0, 1]
    """
    attn = attn.squeeze(-1)                          # (B, H, N)
    B, nH, N = attn.shape

    # Dynamically detect the square patch grid from N
    pg = int(round(N ** 0.5))
    if pg * pg != N:
        # Non-square: use rectangular grid (rare for standard ViTs)
        import math
        pg_h = int(math.ceil(N ** 0.5))
        pg_w = int(math.ceil(N / pg_h))
        # Pad N to pg_h * pg_w if needed
        pad = pg_h * pg_w - N
        if pad > 0:
            attn = torch.cat([attn, attn.new_zeros(B, nH, pad)], dim=-1)
        attn_2d = attn.view(B, nH, pg_h, pg_w)
    else:
        pg_h = pg_w = pg
        attn_2d = attn.view(B, nH, pg_h, pg_w)      # (B, H, pg, pg)

    # Upsample to image resolution
    attn_up = F.interpolate(
        attn_2d.float(),
        size=(img_h, img_w),
        mode="bilinear",
        align_corners=False,
    )                                                 # (B, H, img_h, img_w)

    maps = attn_up.cpu().numpy()

    # Per-head per-sample normalisation → [0, 1]
    for b in range(B):
        for h in range(nH):
            mn, mx = maps[b, h].min(), maps[b, h].max()
            maps[b, h] = (maps[b, h] - mn) / (mx - mn + 1e-8)

    return maps


# ── Public API ──────────────────────────────────────────────────────────────

def plot_attention_avg_heads(
    model,
    images:      torch.Tensor,      # (B, 1, H, W) — already on CPU
    masks_gt:    torch.Tensor,      # (B, H, W)
    save_path:   Path,
    patient_ids: Optional[List[str]] = None,
    model_name:  str = "Model 3",
    max_samples: int = 6,
) -> None:
    """
    Plot cross-attention maps averaged across all heads.

    Panels per row:
      MRI | Attention Map | MRI + Attention overlay | GT Mask
    """
    attn = _extract_attn_weights(model)
    if attn is None:
        print("  [attention_maps] No attention weights found – skipping.")
        return

    H, W   = images.shape[-2:]
    maps   = _attn_to_spatial(attn, img_h=H, img_w=W)  # (B, n_heads, H, W)
    avg_map = maps.mean(axis=1)                          # (B, H, W)

    B = min(images.shape[0], max_samples, avg_map.shape[0])

    fig = plt.figure(figsize=(18, 4.0 * B), facecolor="#0d0d0d")
    gs  = gridspec.GridSpec(B, 4, figure=fig, hspace=0.04, wspace=0.04)

    col_titles = ["FLAIR MRI", "Cross-Attn (avg heads)", "Attn Overlay", "Ground Truth"]
    for col, t in enumerate(col_titles):
        ax = fig.add_subplot(gs[0, col])
        ax.set_title(t, fontsize=11, fontweight="bold", color="white", pad=8)
        ax.axis("off")

    for i in range(B):
        img_np = images[i, 0].numpy()
        img_n  = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)
        amap   = avg_map[i]
        gt_np  = masks_gt[i].numpy() if isinstance(masks_gt, torch.Tensor) \
                 else masks_gt[i]
        pid    = patient_ids[i] if patient_ids else f"Sample {i}"

        # MRI
        ax = fig.add_subplot(gs[i, 0])
        ax.imshow(img_n, cmap="gray")
        ax.set_ylabel(pid, fontsize=7, color="white", rotation=0, labelpad=65, va="center")
        ax.axis("off")

        # Attention map
        ax = fig.add_subplot(gs[i, 1])
        im = ax.imshow(amap, cmap="plasma", vmin=0, vmax=1)
        ax.axis("off")
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        div = make_axes_locatable(ax)
        cax = div.append_axes("right", size="4%", pad=0.04)
        cb  = plt.colorbar(im, cax=cax)
        cb.ax.tick_params(labelsize=6, colors="white")
        cb.outline.set_edgecolor("white")

        # Overlay
        ax = fig.add_subplot(gs[i, 2])
        ax.imshow(img_n, cmap="gray")
        ax.imshow(amap, cmap="plasma", alpha=0.6, vmin=0, vmax=1)
        ax.axis("off")

        # GT Mask
        ax = fig.add_subplot(gs[i, 3])
        ax.imshow(img_n, cmap="gray")
        ax.imshow(
            np.ma.masked_where(gt_np < 0.5, gt_np),
            cmap="Greens", alpha=0.65, vmin=0, vmax=1,
        )
        ax.axis("off")

    fig.suptitle(
        f"{model_name} — Cross-Attention Maps (Averaged Heads)\n"
        "Bright regions = image patches most attended to by the radiological text",
        fontsize=12, color="white", y=1.02,
    )
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight", dpi=140,
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Attention avg-heads figure saved → {save_path}")


def plot_attention_per_head(
    model,
    images:      torch.Tensor,
    save_path:   Path,
    patient_ids: Optional[List[str]] = None,
    model_name:  str = "Model 3",
    sample_idx:  int = 0,           # which batch sample to visualise
) -> None:
    """
    Plot each attention head separately for a single sample.

    Layout: n_heads columns, rows for MRI + per-head maps.
    """
    attn = _extract_attn_weights(model)
    if attn is None:
        print("  [attention_maps] No attention weights found – skipping.")
        return

    H, W  = images.shape[-2:]
    maps  = _attn_to_spatial(attn, img_h=H, img_w=W)   # (B, n_heads, H, W)
    b     = min(sample_idx, maps.shape[0] - 1)
    n_heads = maps.shape[1]
    img_np  = images[b, 0].numpy()
    img_n   = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)
    pid     = patient_ids[b] if patient_ids else f"Sample {b}"

    # Entropy per head (lower = more focused)
    # Handle both (B, n_heads, N_patches) and (B, n_heads, N_patches, seq_len)
    attn_b = attn[b]                                  # (n_heads, N, seq_len) or (n_heads, N)
    if attn_b.dim() == 3:
        attn_b = attn_b.mean(dim=-1)                  # → (n_heads, N_patches)
    attn_np = attn_b.cpu().numpy()                    # (n_heads, N_patches)
    p       = attn_np / (attn_np.sum(-1, keepdims=True) + 1e-10)
    entropy = -(p * np.log(p + 1e-10)).sum(-1)        # (n_heads,)

    ncols = n_heads
    fig, axes = plt.subplots(2, ncols, figsize=(3 * ncols, 7),
                              facecolor="#0d0d0d")
    fig.suptitle(
        f"{model_name} — Per-Head Cross-Attention\n{pid}",
        fontsize=11, color="white", y=1.02,
    )

    for h in range(n_heads):
        # Top row: MRI + attention overlay
        ax = axes[0, h]
        ax.imshow(img_n, cmap="gray")
        ax.imshow(maps[b, h], cmap="plasma", alpha=0.6, vmin=0, vmax=1)
        ax.set_title(f"Head {h+1}\nH={entropy[h]:.2f}", fontsize=8,
                     color="white")
        ax.axis("off")

        # Bottom row: pure attention map
        ax = axes[1, h]
        ax.imshow(maps[b, h], cmap="plasma", vmin=0, vmax=1)
        ax.axis("off")

    # Annotate entropy ranking
    best_head = int(entropy.argmin())
    axes[0, best_head].set_title(
        f"Head {best_head+1}\nH={entropy[best_head]:.2f} ★ (most focused)",
        fontsize=8, color="#ffd700",
    )

    for ax in axes.flat:
        for spine in ax.spines.values():
            spine.set_visible(False)

    fig.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight", dpi=140,
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Per-head attention figure saved → {save_path}")


def plot_attention_head_entropy(
    model,
    save_path:  Path,
    model_name: str = "Model 3",
) -> None:
    """
    Bar chart of per-head entropy (lower = more spatially focused head).
    """
    attn = _extract_attn_weights(model)
    if attn is None:
        print("  [attention_maps] No attention weights – skipping entropy plot.")
        return

    # Batch-average entropy
    # Handle both (B, n_heads, N_patches) and (B, n_heads, N_patches, seq_len)
    attn_t = attn
    if attn_t.dim() == 4:
        attn_t = attn_t.mean(dim=-1)               # → (B, n_heads, N_patches)
    attn_np = attn_t.cpu().numpy()                 # (B, n_heads, N_patches)
    p       = attn_np / (attn_np.sum(-1, keepdims=True) + 1e-10)
    entropy = -(p * np.log(p + 1e-10)).sum(-1)     # (B, n_heads)
    mean_H  = entropy.mean(0)                       # (n_heads,)
    std_H   = entropy.std(0)
    n_heads = mean_H.shape[0]

    fig, ax = plt.subplots(figsize=(10, 4), facecolor="#0d0d0d")
    x_pos = np.arange(n_heads)
    colors = plt.cm.plasma(np.linspace(0.2, 0.9, n_heads))
    bars = ax.bar(x_pos, mean_H, yerr=std_H, color=colors,
                  capsize=5, edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Attention Head", fontsize=11, color="white")
    ax.set_ylabel("Entropy (lower = more focused)", fontsize=11, color="white")
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"H{i+1}" for i in range(n_heads)], color="white")
    ax.tick_params(colors="white")
    ax.set_title(f"{model_name} — Attention Head Entropy", fontsize=12, color="white")
    ax.set_facecolor("#0d0d0d")
    for spine in ax.spines.values():
        spine.set_color("#555555")
    ax.grid(axis="y", alpha=0.3, color="#555555")

    # Annotate best head
    best = int(mean_H.argmin())
    ax.annotate(f"Most focused (H{best+1})",
                xy=(best, mean_H[best]),
                xytext=(best + 0.5, mean_H[best] + std_H[best] + 0.05),
                arrowprops=dict(arrowstyle="->", color="#ffd700"),
                color="#ffd700", fontsize=9)

    fig.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight", dpi=140,
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Head entropy figure saved → {save_path}")
