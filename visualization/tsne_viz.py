"""
visualization/tsne_viz.py
─────────────────────────────────────────────────────────────────────────────
t-SNE visualisation of image and text embeddings to demonstrate
multimodal alignment.

Steps:
  1. Extract image embeddings from CLIP encoder (global feat, 256-dim)
  2. Extract text embeddings (from precomputed .npy, pooled to 768-dim)
  3. Project both to same dim if needed
  4. Concatenate and run t-SNE in 2D
  5. Plot coloured by modality and by patient
─────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import warnings


def extract_embeddings(
    model,
    loader: DataLoader,
    device: str = "cpu",
    max_batches: int = 50,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Extract image and text embeddings from the dataloader.

    For Model 1 (no image encoder), we use U-Net bottleneck features.
    For Models 2 and 3 we use the CLIP global feature + text projection.

    Returns
    -------
    img_embs : (N, D_img)
    txt_embs : (N, D_txt)
    pids     : list of patient IDs
    """
    model.eval()
    has_image_encoder = hasattr(model, "image_encoder")

    img_list, txt_list, pid_list = [], [], []

    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= max_batches:
                break
            batch_dev = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            if has_image_encoder:
                img_feat, _ = model.image_encoder(batch_dev["image"])  # (B, D)
                txt_feat    = model.text_proj(batch_dev["text_emb"])    # (B, D)
            else:
                # Model 1: use raw normalised image intensities (mean per slice)
                img_feat = batch_dev["image"].view(batch_dev["image"].shape[0], -1).mean(-1, keepdim=True).expand(-1, 256)
                txt_feat = batch_dev["text_emb"][:, :256]               # first 256 dims

            img_list.append(img_feat.cpu().numpy())
            txt_list.append(txt_feat.cpu().numpy())
            pid_list.extend(batch_dev["patient_id"] if "patient_id" in batch_dev else ["?"] * img_feat.shape[0])

    img_embs = np.concatenate(img_list, axis=0)
    txt_embs = np.concatenate(txt_list, axis=0)
    return img_embs, txt_embs, pid_list


def plot_tsne(
    img_embs:   np.ndarray,         # (N, D)
    txt_embs:   np.ndarray,         # (N, D)
    patient_ids: List[str],
    save_path:  Path,
    model_name: str = "model",
    perplexity: int = 30,
    n_iter:     int = 1000,
) -> None:
    """
    Compute t-SNE on concatenated image + text embeddings and plot.

    Colours:
      - Orange dots  = image embeddings
      - Blue crosses = text embeddings
      Paired samples (same patient) are connected by a thin grey line.
    """
    N = img_embs.shape[0]

    # Ensure same dimensionality
    D = min(img_embs.shape[1], txt_embs.shape[1])
    img_embs = img_embs[:, :D]
    txt_embs = txt_embs[:, :D]

    # Scale
    scaler  = StandardScaler()
    combined = scaler.fit_transform(np.concatenate([img_embs, txt_embs], axis=0))

    # t-SNE
    perplexity = min(perplexity, N // 2 - 1)
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        max_iter=n_iter,
        random_state=42,
        init="pca",
        learning_rate="auto",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        coords = tsne.fit_transform(combined)   # (2*N, 2)

    img_coords = coords[:N]
    txt_coords = coords[N:]

    # ── Assign per-patient colour ──────────────────────────────────────────
    unique_pids = list(dict.fromkeys(patient_ids))  # preserve order
    try:
        cmap = matplotlib.colormaps.get_cmap("tab20")
    except AttributeError:
        cmap = plt.cm.get_cmap("tab20")  # fallback for matplotlib < 3.7
    n_colors = max(len(unique_pids), 1)
    pid_to_colour = {pid: cmap(i / n_colors) for i, pid in enumerate(unique_pids)}
    colours = [pid_to_colour[p] for p in patient_ids]

    # ── Plot ──────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: modality separation
    ax = axes[0]
    ax.scatter(img_coords[:, 0], img_coords[:, 1],
               c="#e07b54", s=20, alpha=0.6, label="Image emb")
    ax.scatter(txt_coords[:, 0], txt_coords[:, 1],
               c="#5470c6", s=20, marker="x", alpha=0.6, label="Text emb")
    ax.set_title("t-SNE by Modality", fontsize=12)
    ax.legend()
    ax.set_xlabel("t-SNE dim 1")
    ax.set_ylabel("t-SNE dim 2")
    ax.spines[["top", "right"]].set_visible(False)

    # Right: patient-coloured pairs with connection lines
    ax = axes[1]
    for ic, tc, col in zip(img_coords, txt_coords, colours):
        ax.plot([ic[0], tc[0]], [ic[1], tc[1]],
                color=col, alpha=0.2, linewidth=0.7)
    ax.scatter(img_coords[:, 0], img_coords[:, 1],
               c=colours, s=20, alpha=0.7, marker="o", label="Image")
    ax.scatter(txt_coords[:, 0], txt_coords[:, 1],
               c=colours, s=20, alpha=0.7, marker="^", label="Text")
    ax.set_title("t-SNE by Patient (lines = same patient)", fontsize=12)
    ax.legend(handles=[
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="grey",
                   markersize=8, label="Image"),
        plt.Line2D([0], [0], marker="^", color="w", markerfacecolor="grey",
                   markersize=8, label="Text"),
    ])
    ax.set_xlabel("t-SNE dim 1")
    ax.set_ylabel("t-SNE dim 2")
    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle(f"{model_name} — Multimodal Embedding Alignment (t-SNE)", fontsize=13)
    fig.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight", dpi=120)
    plt.close(fig)
    print(f"t-SNE figure saved → {save_path}")
