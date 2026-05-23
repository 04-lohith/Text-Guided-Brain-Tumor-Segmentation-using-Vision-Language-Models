"""Visualization package."""
from .plot_training import plot_training_curves, save_history, load_history
from .plot_segmentation import visualise_predictions
from .gradcam import GradCAM, GradCAMMode, save_gradcam_figures
from .attention_maps import (
    plot_attention_avg_heads, plot_attention_per_head, plot_attention_head_entropy,
)
from .tsne_viz import extract_embeddings, plot_tsne
from .error_analysis import (
    collect_per_slice_metrics,
    plot_dice_distribution,
    plot_worst_predictions,
)

__all__ = [
    "plot_training_curves", "save_history", "load_history",
    "visualise_predictions",
    "GradCAM", "GradCAMMode", "save_gradcam_figures",
    "plot_attention_avg_heads", "plot_attention_per_head", "plot_attention_head_entropy",
    "extract_embeddings", "plot_tsne",
    "collect_per_slice_metrics", "plot_dice_distribution", "plot_worst_predictions",
]
