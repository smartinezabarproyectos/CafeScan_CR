from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class BaseModel(ABC, nn.Module):
    """Contract all DL models must satisfy."""

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ...

    def param_groups(self, lr: float, backbone_lr_mult: float = 0.1) -> list[dict]:
        """Differential LR: backbone gets lr*mult, head gets lr."""
        backbone, head = self._backbone_and_head()
        return [
            {"params": backbone.parameters(), "lr": lr * backbone_lr_mult},
            {"params": head.parameters(),     "lr": lr},
        ]

    @abstractmethod
    def _backbone_and_head(self) -> tuple[nn.Module, nn.Module]:
        """Return (backbone, classification_head) for differential LR."""
        ...

    def freeze_backbone(self) -> None:
        backbone, _ = self._backbone_and_head()
        for p in backbone.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self) -> None:
        backbone, _ = self._backbone_and_head()
        for p in backbone.parameters():
            p.requires_grad = True

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def num_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
