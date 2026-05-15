from __future__ import annotations

from pathlib import Path

from torch.utils.data import ConcatDataset

from .bracol import BracolDataset
from .jmuben import JMuBENDataset
from .label_mapper import UNIFIED_LABELS


def build_unified_dataset(data_root: str | Path, transform=None) -> ConcatDataset:
    """Combine BRACOL + JMuBEN + JMuBEN2 into one dataset.

    Expected structure:
        data_root/
            bracol/
            jmuben/
            jmuben2/
    """
    root = Path(data_root)
    parts = []

    bracol_root = root / "bracol"
    if bracol_root.exists():
        parts.append(BracolDataset(bracol_root, transform=transform))

    for name in ("jmuben", "jmuben2"):
        sub = root / name
        if sub.exists():
            parts.append(JMuBENDataset(sub, transform=transform))

    if not parts:
        raise FileNotFoundError(f"No datasets found under {root}")

    return ConcatDataset(parts)


def class_names() -> list[str]:
    return UNIFIED_LABELS
