from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from src.data.label_mapper import UNIFIED_LABELS


def compute_class_weights(dataset: Dataset, num_classes: int = 5) -> torch.Tensor:
    """Inverse-frequency class weights from a dataset's label distribution."""
    counts = torch.zeros(num_classes)
    for _, label in dataset:
        counts[label] += 1
    weights = counts.sum() / (num_classes * counts.clamp(min=1))
    return weights


def compute_class_weights_from_records(records, indices, num_classes: int = 5) -> torch.Tensor:
    """Faster version working directly on TaggedRecord lists + index subset."""
    counts = torch.zeros(num_classes)
    for i in indices:
        counts[records[i].label] += 1
    weights = counts.sum() / (num_classes * counts.clamp(min=1))
    return weights


class LabelSmoothingCrossEntropy(nn.Module):
    """Cross-entropy with label smoothing + optional class weights."""

    def __init__(
        self,
        smoothing: float = 0.1,
        weight: torch.Tensor | None = None,
        reduction: str = "mean",
    ):
        super().__init__()
        self.smoothing = smoothing
        self.weight = weight
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        n_classes = logits.size(-1)
        log_probs = F.log_softmax(logits, dim=-1)

        # smooth targets
        with torch.no_grad():
            smooth = torch.full_like(log_probs, self.smoothing / (n_classes - 1))
            smooth.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)

        loss = -(smooth * log_probs)

        if self.weight is not None:
            w = self.weight.to(logits.device)[targets]
            loss = loss.sum(dim=-1) * w
        else:
            loss = loss.sum(dim=-1)

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss
