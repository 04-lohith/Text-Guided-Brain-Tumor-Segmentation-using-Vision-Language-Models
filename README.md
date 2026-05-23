# Text-Guided Brain Tumor Segmentation using Vision-Language Models

> **Research-grade multimodal deep learning framework** — integrating FLAIR MRI with radiology text reports for pixel-wise brain tumor segmentation via CLIP + BioClinicalBERT + U-Net with cross-attention fusion.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Architecture](#2-architecture)
   - [Model 1 — Image-Only U-Net Baseline](#model-1--image-only-u-net-baseline)
   - [Model 2 — Concatenation Fusion](#model-2--concatenation-fusion)
   - [Model 3 — Cross-Attention VLM](#model-3--cross-attention-vlm-recommended)
3. [Results](#3-results)
4. [Directory Structure](#4-directory-structure)
5. [Setup & Installation](#5-setup--installation)
6. [Dataset Preparation](#6-dataset-preparation)
7. [Training](#7-training)
8. [Evaluation](#8-evaluation)
9. [Inference](#9-inference)
10. [Ablation Study](#10-ablation-study)
11. [Visualisations & Explainability](#11-visualisations--explainability)
12. [MacBook Optimisation Strategies](#12-macbook-optimisation-strategies)

---

## 1. Project Overview

This project develops a **multimodal deep learning framework** for pixel-wise brain tumor segmentation by jointly learning from:

- **FLAIR MRI volumes** (BraTS 2020) — 3D volumes sliced axially into 2D images (128×128)
- **Radiology text reports** (TextBraTS) — one report per patient describing tumor location, edema extent, signal characteristics, and structural involvement; tokenized with BioClinicalBERT (max 128 tokens)

### Dataset

| Split | Patients | Slices |
|---|---|---|
| Train | 258 | 33,024 |
| Validation | 86 | 11,008 |

Ground-truth masks are collapsed to **binary** (tumor vs. background).

### Evaluation Metrics

| Metric | Formula |
|---|---|
| Dice | 2·TP / (2·TP + FP + FN) |
| IoU | TP / (TP + FP + FN) |
| Precision | TP / (TP + FP) |
| Recall | TP / (TP + FN) |
| HD95 | 95th-percentile Hausdorff Distance (mm) — lower is better |

### Key Features

- Three model variants enabling a full **ablation study** (image-only → global fusion → cross-attention)
- **CLIP ViT-B/32** + **BioClinicalBERT** as frozen pretrained encoders
- **FiLM conditioning** of U-Net decoder with text features at every resolution
- Token-wise **multi-head cross-attention** for spatially-resolved text-image alignment
- Full training pipeline (early stopping, cosine LR schedule, gradient accumulation)
- Explainability: **Grad-CAM** saliency maps, **t-SNE** embedding visualisation
- MacBook-optimised: MPS acceleration, no GPU required

---

## 2. Architecture

### Model 1 — Image-Only U-Net Baseline

```
FLAIR slice (128×128)
       ↓
   U-Net Encoder (4 levels)
   [32 → 64 → 128 → 256 → 512 bottleneck]
   Double-Conv (3×3, BN, ReLU) + MaxPool 2×2
       ↓
   U-Net Decoder (bilinear up-sampling + skip connections)
       ↓
Segmentation Mask (128×128), sigmoid → threshold 0.5
```

**Design choices:**
- **Input:** Single-channel FLAIR slice (128×128)
- **Loss:** Focal Tversky Loss (α=0.3, β=0.7, γ=0.75) — weighted to penalise false negatives more than false positives
- **Optimizer:** AdamW, LR = 3×10⁻⁵ with cosine schedule
- **Trainable parameters:** 12.6 M
- **Text conditioning:** None — purely image-driven

This model serves as the **image-only baseline**. It must infer tumor boundaries from pixel intensities alone, which leads to over-segmentation (high recall, lower precision) in ambiguous FLAIR signal regions.

---

### Model 2 — Concatenation Fusion

```
FLAIR slice → CLIP ViT-B/32 (frozen) → global CLS image feat (512-d)
                                                           ↘
Text report → BioClinicalBERT (frozen) → pooled CLS (768-d)
                                → Linear projection (256-d)
                                                           ↙
                     Concatenate → FiLM(γ, β) parameters
                           ↓
FLAIR slice → U-Net (FiLM-conditioned at every decoder level) → Mask
```

**FiLM conditioning:**
```
FiLM(h) = γ ⊙ h + β,   γ, β ∈ ℝᶜ
```
Applied at the bottleneck and all 4 up-sampling stages, allowing text to modulate spatial features at multiple resolutions.

**Key limitation:** The text is pooled to a **single global vector** — every spatial position in the image receives identical text conditioning regardless of anatomical location.

- **Trainable parameters:** 14.1 M

---

### Model 3 — Cross-Attention VLM *(recommended)*

```
FLAIR slice → CLIP ViT-B/32 (last block unfrozen)
           → patch tokens V ∈ ℝ^(B × 49 × 768)   [7×7 spatial patches]
                                                         ↓  Q = V·Wq
Text → BioClinicalBERT (frozen) → token sequence T ∈ ℝ^(B × 128 × 768)
                                                         ↓  K,V = T·Wk, T·Wv

                Multi-Head Cross-Attention (8 heads, dₖ=96)
                Attn(Q,K,V) = softmax(QKᵀ / √dₖ) · V
                A ∈ ℝ^(B × 8 × 49 × 128) stored for visualisation
                                    ↓
                          mean-pool → fused (256-d)
                                    ↓
              FiLM conditioning → U-Net decoder → Segmentation Mask
```

**Training loss:**
```
L = L_FocalTversky + λ·H(A),   λ = 0.001
```
The entropy term `H(A)` penalises uniform attention to encourage cross-head specialisation.

**Differential learning rates** (prevent catastrophic forgetting):

| Component | LR |
|---|---|
| U-Net | 3×10⁻⁵ |
| FiLM layers | 1×10⁻⁵ |
| Cross-attention | 5×10⁻⁶ |
| CLIP final block | 1×10⁻⁶ |

- **Trainable parameters:** 22.3 M
- **Best checkpoint:** Epoch 12 (of 18 total, 3-epoch warmup, early stopping patience = 6)

Each image patch token attends over all 128 text tokens, enabling the model to focus on tumor-related image regions that correspond to specific clinical descriptors (e.g., "hyperintense FLAIR signal in the right temporal lobe").

---

## 3. Results

### Individual Model Results

| Model | Dice | IoU | Precision | Recall | HD95 (mm) |
|---|---|---|---|---|---|
| **M1: Image-Only U-Net** | 0.696 | 0.649 | 0.710 | 0.906 | 10.13 |
| **M2: Concat Fusion** | 0.732 | 0.687 | 0.772 | 0.886 | 9.70 |
| **M3: Cross-Attn VLM** | **0.828** | **0.786** | **0.897** | **0.874** | **8.15** |

### Gains vs. Baseline

| Metric | M3 vs. M1 | M3 vs. M2 |
|---|---|---|
| Dice | +13.2% | +9.6% |
| IoU | +21.1% | +14.4% |
| Precision | +26.3% | +16.2% |
| Recall | −3.5% | −1.3% |
| HD95 | −19.5% | −16.0% |

### Key Findings

- **M1 → M2 (+3.6 Dice):** Even a pooled, spatially uniform text embedding improves segmentation. The global semantic prior from the radiology report reduces false positives (+6.2 precision points).
- **M2 → M3 (+9.6 Dice):** Token-wise cross-attention provides significantly richer conditioning — each image patch can attend differentially to specific clinical descriptors.
- **HD95 trend (10.13 → 9.70 → 8.15 mm):** A consistent downward trend confirms text conditioning progressively sharpens tumor boundary delineation — clinically significant for radiation planning.
- **Recall trade-off:** M3 shows a slight recall decrease (−3.5% vs. M1). This is intentional: improved precision eliminates over-aggressive predictions, yielding a favourable precision-recall rebalance confirmed by Dice/IoU gains.
- **Explainability note:** Attention maps are near-uniform across all 8 heads (entropy ≈ 3.89 nats; max = log 128 ≈ 4.85 nats), indicating that gains stem from **semantic FiLM modulation** rather than explicit spatial token-region grounding. Performance improvement and interpretability are separate; future work should address the grounding gap through contrastive pre-training and attention rollout.

---

## 4. Directory Structure

```
brain_tumor_vlm/
├── configs/
│   ├── config.yaml              # Master configuration
│   └── config_loader.py         # YAML loader with DotDict
├── data/
│   ├── dataset.py               # BraTSMultimodalDataset
│   └── dataloader.py            # DataLoader factory
├── models/
│   ├── unet.py                  # U-Net with FiLM conditioning
│   ├── image_encoder.py         # CLIP ViT-B/32 wrapper
│   ├── text_encoder.py          # TextProjectionHead (768 → 256)
│   ├── fusion.py                # ConcatFusion + CrossAttnFusion
│   ├── model1_image_only.py     # Model 1 definition
│   ├── model2_concat_fusion.py  # Model 2 definition
│   └── model3_cross_attention.py# Model 3 definition
├── training/
│   ├── losses.py                # Dice, BCE+Dice, Focal Tversky
│   └── trainer.py               # Full training engine
├── evaluation/
│   └── metrics.py               # MetricAccumulator + all metrics
├── visualization/
│   ├── plot_training.py         # Loss/Dice/IoU curves
│   ├── plot_segmentation.py     # Prediction overlays (TP/FP/FN)
│   ├── gradcam.py               # Grad-CAM saliency maps
│   ├── tsne_viz.py              # t-SNE embedding alignment
│   └── error_analysis.py        # Dice distribution + worst cases
├── scripts/
│   ├── train.py                 # Training entry-point
│   ├── evaluate.py              # Evaluation + visualisation
│   ├── ablation.py              # Multi-model ablation study
│   ├── gen_best_seg.py          # Generate best segmentation samples
│   ├── train_all.sh             # Shell script to train all models
│   └── debug_dataset.py         # Dataset debugging utilities
├── inference/
│   └── infer.py                 # Single-patient inference pipeline
├── utils/
│   └── logging_utils.py
├── report/
│   └── main.tex                 # Full LaTeX project report
├── checkpoints/                 # Saved model weights (not tracked)
├── logs/                        # TensorBoard logs (not tracked)
├── results/
│   ├── plots/                   # Training curves, visualisations
│   └── tables/                  # CSV performance tables
├── environment.yml              # Conda environment spec
└── requirements.txt
```

---

## 5. Setup & Installation

### Prerequisites

- macOS (Apple Silicon or Intel) — MPS acceleration supported
- Python 3.10
- Anaconda or Miniconda

### Create and activate Conda environment

```bash
conda create -n brain_tumor_vlm python=3.10 -y
conda activate brain_tumor_vlm
```

### Install PyTorch (CPU/MPS for MacBook)

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### Install all dependencies

```bash
cd brain_tumor_vlm/
pip install -r requirements.txt
pip install git+https://github.com/openai/CLIP.git
```

### Verify installation

```bash
python -c "import torch, clip, transformers; print('All OK')"
```

Alternatively, use the provided Conda environment file:

```bash
conda env create -f environment.yml
conda activate brain_tumor_vlm
```

---

## 6. Dataset Preparation

### Required datasets (not included in this repo)

| Dataset | Source | Notes |
|---|---|---|
| BraTS 2020 FLAIR | [Kaggle / Synapse](https://www.kaggle.com/datasets/awsaf49/brats2020-training-data) | FLAIR volumes as `.npy` |
| TextBraTS | [TextBraTS GitHub](https://github.com/gt-hit/TextBraTS) | BioClinicalBERT embeddings |

### Expected directory layout (outside this repo)

```
<project_root>/
├── FLAIR_BRATS2020/
│   ├── train/
│   │   ├── images/image_X.npy     # (128, 128, 128) float64 volumes
│   │   └── masks/mask_X.npy       # (128, 128, 128, 4) float32 one-hot
│   └── val/
│       ├── images/
│       └── masks/
└── TextBRats/TextBraTSData/
    └── BraTS20_Training_XXX/
        ├── *_flair_text.npy        # (1, 128, 768) BERT token embeddings
        └── *_flair_text.txt        # Raw radiology report
```

> **Critical mapping:** `image_X.npy` corresponds to patient `BraTS20_Training_{X+1:03d}` (zero-indexed images → 1-indexed text folders).

### Debug dataset loading

```bash
conda activate brain_tumor_vlm
cd brain_tumor_vlm/
python scripts/debug_dataset.py --split train --n-samples 4
python scripts/debug_dataset.py --split val
```

---

## 7. Training

All scripts must be run from the `brain_tumor_vlm/` directory.

```bash
conda activate brain_tumor_vlm
cd brain_tumor_vlm/
```

### Train individual models

```bash
# Model 1 — Image-Only Baseline (fastest, ~12.6M params)
python scripts/train.py --model model1_image_only

# Model 2 — Concatenation Fusion (~14.1M params)
python scripts/train.py --model model2_concat_fusion

# Model 3 — Cross-Attention VLM (~22.3M params)
python scripts/train.py --model model3_cross_attention
```

### Train all models sequentially

```bash
bash scripts/train_all.sh
```

### Resume from checkpoint

```bash
python scripts/train.py --model model1_image_only \
  --resume checkpoints/model1_image_only_ep020.pth
```

### Monitor with TensorBoard

```bash
tensorboard --logdir logs/
```

### Key hyperparameters (edit `configs/config.yaml`)

| Parameter | Default | Notes |
|---|---|---|
| `training.num_epochs` | 50 | M3 best at epoch 12 |
| `training.lr` | 1e-4 | AdamW base LR |
| `training.gradient_accumulation_steps` | 4 | Effective batch = 2×4 = 8 |
| `training.early_stopping_patience` | 10 | Epochs without improvement |
| `dataloader.train_batch_size` | 2 | Keep low for MacBook |
| `loss.type` | `dice_bce` | Options: `dice`, `dice_bce`, `focal_tversky` |

---

## 8. Evaluation

```bash
# Metrics only + segmentation visualisation
python scripts/evaluate.py \
  --model model3_cross_attention \
  --checkpoint checkpoints/model3_cross_attention_best.pth

# Full evaluation with all visualisations (Grad-CAM, t-SNE, Dice distribution)
python scripts/evaluate.py \
  --model model3_cross_attention \
  --checkpoint checkpoints/model3_cross_attention_best.pth \
  --all-viz
```

Output metrics: **Dice**, **IoU**, **Precision**, **Recall**, **HD95**

> **Note on text generation metrics:** ROUGE/BLEU measure n-gram overlap between *generated* and *reference* text. In this project, text is used exclusively as a **conditioning input** — no text is generated. Therefore, ROUGE/BLEU are not applicable; segmentation metrics fully capture model performance.

---

## 9. Inference

Run inference on a single patient:

```bash
python inference/infer.py \
  --model model3_cross_attention \
  --checkpoint checkpoints/model3_cross_attention_best.pth \
  --image "../FLAIR_BRATS2020/train/images/image_0.npy" \
  --text "../TextBRats/TextBraTSData/BraTS20_Training_001/BraTS20_Training_001_flair_text.npy" \
  --output results/inference/patient_001_pred.npy
```

---

## 10. Ablation Study

Train all three models first, then run:

```bash
python scripts/ablation.py
```

Skip models whose checkpoints are missing:

```bash
python scripts/ablation.py --skip-missing
```

**Outputs:**

| File | Description |
|---|---|
| `results/plots/ablation/ablation_dice.png` | Dice comparison bar chart |
| `results/plots/ablation/ablation_iou.png` | IoU comparison bar chart |
| `results/plots/ablation/ablation_hd95.png` | HD95 comparison bar chart |
| `results/plots/ablation/ablation_radar.png` | Radar chart — all metrics |
| `results/tables/ablation_results.csv` | Full results table |

---

## 11. Visualisations & Explainability

All visualisations are generated automatically during evaluation with `--all-viz`.

| Plot | Description |
|---|---|
| `*_training_summary.png` | Loss + Dice + IoU training curves per epoch |
| `segmentation_samples.png` | FLAIR \| GT \| Prediction \| Overlay (Green=TP, Red=FP, Yellow=FN) |
| `segmentation_best.png` | Top-6 predictions by Dice (all ≥ 0.90) for M3 |
| `gradcam.png` | Grad-CAM saliency maps on encoder/decoder features |
| `tsne.png` | t-SNE of image+text embeddings showing multimodal alignment |
| `dice_distribution.png` | Per-slice Dice histogram |
| `worst_predictions.png` | Top-N failure cases visualised |

### Grad-CAM Notes

- **Model 1:** Activation broadly distributed across the brain (no text guidance).
- **Model 2:** Activation slightly more concentrated around tumor regions.
- **Model 3:** Near-zero activation on empty slices; diffuse but tumor-focused on tumor slices. Consistent with multi-scale decoder feature integration via FiLM.

### t-SNE Notes

- **Model 2:** Text embeddings (pooled CLS) cluster tightly, reflecting linguistic homogeneity within the BraTS cohort.
- **Model 3:** Image embeddings distributed; text embeddings collapse to a near-single point (reflects pooled representation — token-level variance used in cross-attention is not captured in the pooled visualisation).

---

## 12. MacBook Optimisation Strategies

| Strategy | Implementation |
|---|---|
| **MPS acceleration** | Automatic — uses Apple Silicon GPU via `torch.device("mps")` |
| **Frozen encoders** | CLIP and BERT weights frozen; only projection heads + U-Net trained |
| **Gradient accumulation** | `gradient_accumulation_steps=4` → effective batch 8 with physical batch 2 |
| **Lightweight batch size** | `train_batch_size=2` |
| **Precomputed text embeddings** | `.npy` embeddings loaded directly (no BERT inference during training) |
| **2D slicing** | 3D volumes sliced to 2D (avoids 3D conv memory overhead) |
| **Small U-Net** | `base_ch=32, depth=4` (~12.6M params for image-only model) |
| **num_workers=0** | Avoids macOS multiprocessing fork issues |
| **No AMP** | MPS doesn't fully support `torch.autocast` yet — removed for stability |

---

## Citation

If you use this code for research, please cite the BraTS 2020 dataset:

```bibtex
@article{brats2020,
  title   = {Identifying the Best Machine Learning Algorithms for Brain Tumor Segmentation},
  author  = {Menze, B. H. and others},
  journal = {arXiv preprint arXiv:1811.02629},
  year    = {2018}
}
```

---

## License

This project is for academic and research purposes.
