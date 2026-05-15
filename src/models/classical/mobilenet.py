from __future__ import annotations

import torch
import timm
import torch.nn as nn

from src.models.base_models import BaseModel

class MobileNetV3(BaseModel):

    def __init__(self, num_classes: int = 5, pretrained: bool = True, drop_rate: float = 0.2):
        super().__init__()
        self.backbone = timm.create_model(
            "mobilenetv3_large_100",
            pretrained=pretrained,
            num_classes=0,
            drop_rate=drop_rate,
        )

        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224)
            in_features = self.backbone(dummy).shape[1]
        self.head = nn.Sequential(
            nn.Dropout(drop_rate),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.head(features)

    def _backbone_and_head(self):
        return self.backbone, self.head
