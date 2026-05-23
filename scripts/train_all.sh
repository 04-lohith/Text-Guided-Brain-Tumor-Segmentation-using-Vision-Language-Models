#!/usr/bin/env bash
# =============================================================================
# train_all.sh
# Sequential training of all 3 model variants.
#
# Epochs:
#   Model 1 (Image-Only U-Net)     : 20  (from config.yaml)
#   Model 2 (Concat Fusion)        : 20  (from config.yaml)
#   Model 3 (Cross-Attn Fusion)    : 15  (--epochs override)
#
# Usage (from inside brain_tumor_vlm/):
#   bash scripts/train_all.sh
# =============================================================================

set -e  # exit immediately if any command fails

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "============================================================"
echo "  Brain Tumor VLM — Full Training Pipeline"
echo "  Project root : $PROJECT_ROOT"
echo "  Start time   : $(date)"
echo "============================================================"

# ── Model 1: Image-Only U-Net (20 epochs) ────────────────────────────────────
echo ""
echo ">>> [1/3] Training model1_image_only  (20 epochs)"
echo "------------------------------------------------------------"
python "$SCRIPT_DIR/train.py" --model model1_image_only
echo ">>> model1_image_only DONE"

# ── Model 2: Concatenation Fusion (20 epochs) ────────────────────────────────
echo ""
echo ">>> [2/3] Training model2_concat_fusion  (20 epochs)"
echo "------------------------------------------------------------"
python "$SCRIPT_DIR/train.py" --model model2_concat_fusion
echo ">>> model2_concat_fusion DONE"

# ── Model 3: Cross-Attention Fusion (15 epochs) ──────────────────────────────
echo ""
echo ">>> [3/3] Training model3_cross_attention  (15 epochs)"
echo "------------------------------------------------------------"
python "$SCRIPT_DIR/train.py" --model model3_cross_attention --epochs 15
echo ">>> model3_cross_attention DONE"

# ── Ablation Study ───────────────────────────────────────────────────────────
echo ""
echo ">>> Running ablation study..."
echo "------------------------------------------------------------"
python "$SCRIPT_DIR/ablation.py" --skip-missing
echo ">>> Ablation DONE"

echo ""
echo "============================================================"
echo "  ALL DONE.  $(date)"
echo "  Checkpoints : $PROJECT_ROOT/checkpoints/"
echo "  Results     : $PROJECT_ROOT/results/"
echo "============================================================"
