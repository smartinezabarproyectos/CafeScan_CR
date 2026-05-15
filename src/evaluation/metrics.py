from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src.data.label_mapper import UNIFIED_LABELS

def compute_metrics(
    y_true: list[int] | np.ndarray,
    y_pred: list[int] | np.ndarray,
    class_names: list[str] = UNIFIED_LABELS,
) -> dict:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    return {
        "accuracy":         float(accuracy_score(y_true, y_pred)),
        "macro_f1":         float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1":      float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "macro_precision":  float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall":     float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "per_class_f1":     f1_score(y_true, y_pred, average=None, zero_division=0).tolist(),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "report":           classification_report(y_true, y_pred, target_names=class_names, zero_division=0),
    }

@torch.no_grad()
def evaluate_loader(model, loader, device) -> tuple[list[int], list[int]]:
    model.eval()
    all_true, all_pred = [], []
    for imgs, labels in loader:
        imgs = imgs.to(device)
        logits = model(imgs)
        preds = logits.argmax(dim=1).cpu().tolist()
        all_true.extend(labels.tolist())
        all_pred.extend(preds)
    return all_true, all_pred
