"""Model registry and factory.

Instantiate any registered model from config.model.name.
"""

from __future__ import annotations

import logging

from config import ModelConfig
from models.base import BaseModel
from models.res_attn_unet import ResAttnUNet3D

logger = logging.getLogger(__name__)

# Registry maps config keys to model classes
_REGISTRY: dict[str, type[BaseModel]] = {
    "res_attn_unet": ResAttnUNet3D,
}

# Lazy imports for optional heavy models
_LAZY_REGISTRY: dict[str, str] = {
    "attn_unet": "models.attn_unet.AttnUNet3D",
    "unetr": "models.unetr.UNETR",
    "swin_unetr": "models.swin_unetr.SwinUNETR",
    "mednext": "models.mednext.MedNeXt",
}


def build_model(cfg: ModelConfig) -> BaseModel:
    """Instantiate a model from the config.

    Args:
        cfg: Model configuration with 'name' field.

    Returns:
        Instantiated BaseModel.

    Raises:
        ValueError: If model name is not registered.
    """
    name = cfg.name

    # Try eager registry first
    if name in _REGISTRY:
        model_cls = _REGISTRY[name]
        model = model_cls(
            in_channels=cfg.in_channels,
            out_channels=cfg.out_channels,
            features=cfg.features,
            dropout=cfg.dropout,
        )
        logger.info(
            "Built model '%s' | params=%s",
            name, f"{model.get_num_parameters():,}",
        )
        return model

    # Try lazy import
    if name in _LAZY_REGISTRY:
        import importlib  # noqa: PLC0415
        module_path, class_name = _LAZY_REGISTRY[name].rsplit(".", 1)
        try:
            module = importlib.import_module(module_path)
            model_cls = getattr(module, class_name)
            model = model_cls(
                in_channels=cfg.in_channels,
                out_channels=cfg.out_channels,
                features=cfg.features,
                dropout=cfg.dropout,
            )
            logger.info("Built model '%s' | params=%s", name, f"{model.get_num_parameters():,}")
            return model
        except ImportError as e:
            raise ImportError(
                f"Model '{name}' requires additional dependencies: {e}"
            ) from e

    available = list(_REGISTRY.keys()) + list(_LAZY_REGISTRY.keys())
    raise ValueError(
        f"Unknown model: '{name}'. Available: {available}"
    )
