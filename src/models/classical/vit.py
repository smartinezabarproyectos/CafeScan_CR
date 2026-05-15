from __future__ import annotations

import timm
import torch.nn as nn

from src.models.base_models import BaseModel


class ViTSmall(BaseModel):
    """ViT-Small/16 fine-tuned for 5-class coffee disease classification.

    Uses ViT-S/16 (22M params) instead of ViT-B/16 (86M) — more tractable
    on CPU and sufficient for 224x224 leaf images.
    """

    def __init__(self, num_classes: int = 5, pretrained: bool = True, drop_rate: float = 0.1):
        super().__init__()
        self.backbone = timm.create_model(
            "vit_small_patch16_224",
            pretrained=pretrained,
            num_classes=0,
            drop_rate=drop_rate,
        )
        in_features = self.backbone.num_features
        self.head = nn.Sequential(
            nn.Dropout(drop_rate),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.head(features)

    def _backbone_and_head(self):
        return self.backbone, self.head
