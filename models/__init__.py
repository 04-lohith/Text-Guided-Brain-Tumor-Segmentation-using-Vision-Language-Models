"""Models package."""
from .model1_image_only import ImageOnlyUNet
from .model2_concat_fusion import ConcatFusionModel
from .model3_cross_attention import CrossAttentionFusionModel
from .unet import UNet
from .fusion import ConcatFusion, CrossAttnFusion
from .image_encoder import CLIPImageEncoder
from .text_encoder import TextProjectionHead


def build_model(model_name: str, cfg=None):
    """
    Factory function to instantiate a model by name.

    Parameters
    ----------
    model_name : str
        One of 'model1_image_only', 'model2_concat_fusion',
        'model3_cross_attention'.
    cfg : DotDict
        Project configuration.

    Returns
    -------
    nn.Module
    """
    registry = {
        "model1_image_only":      ImageOnlyUNet,
        "model2_concat_fusion":   ConcatFusionModel,
        "model3_cross_attention": CrossAttentionFusionModel,
    }
    if model_name not in registry:
        raise ValueError(
            f"Unknown model '{model_name}'. "
            f"Choose from: {list(registry.keys())}"
        )
    model = registry[model_name](cfg=cfg)
    n_params = model.count_parameters()
    print(f"[Model] {model_name}  |  trainable params: {n_params:,}")
    return model


__all__ = [
    "ImageOnlyUNet",
    "ConcatFusionModel",
    "CrossAttentionFusionModel",
    "UNet",
    "ConcatFusion",
    "CrossAttnFusion",
    "CLIPImageEncoder",
    "TextProjectionHead",
    "build_model",
]
