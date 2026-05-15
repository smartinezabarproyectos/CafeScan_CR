from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset


class BaseDataset(ABC, Dataset):
    """Contract all dataset loaders must satisfy."""

    def __init__(self, root: str | Path, transform=None):
        self.root = Path(root)
        self.transform = transform
        self.samples: list[tuple[Path, int]] = []  # (img_path, label_idx)
        self._load()

    @abstractmethod
    def _load(self) -> None:
        """Populate self.samples."""

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label

    def class_counts(self) -> dict[int, int]:
        from collections import Counter
        return dict(Counter(lbl for _, lbl in self.samples))
