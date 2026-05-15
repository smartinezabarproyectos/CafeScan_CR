from __future__ import annotations

from src.core.exceptions import ModelNotFoundError

REGISTRY = {
    "efficientnet_b0": ("src.models.classical.efficientnet", "EfficientNetB0"),
    "resnet50":        ("src.models.classical.resnet",       "ResNet50"),
    "vit":             ("src.models.classical.vit",          "ViTSmall"),
    "mobilenet":       ("src.models.classical.mobilenet",    "MobileNetV3"),
}

def build_model(name: str, **kwargs):
    if name not in REGISTRY:
        raise ModelNotFoundError(f"'{name}' not in registry. Available: {list(REGISTRY)}")
    import importlib
    module_path, class_name = REGISTRY[name]
    cls = getattr(importlib.import_module(module_path), class_name)
    return cls(**kwargs)

def list_models(kind: str = "all") -> list[str]:
    return list(REGISTRY)
