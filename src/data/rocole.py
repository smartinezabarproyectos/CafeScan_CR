from __future__ import annotations

from pathlib import Path

from .base_dataset import BaseDataset
from .label_mapper import label_to_idx

_IMG_EXTS = {".jpg", ".jpeg", ".png"}

class RoCoLeDataset(BaseDataset):

    def _load(self) -> None:
        for class_dir in sorted(self.root.iterdir()):
            if not class_dir.is_dir():
                continue
            try:
                idx = label_to_idx(class_dir.name)
            except ValueError:
                continue
            for img_path in sorted(class_dir.glob("**/*")):
                if img_path.suffix.lower() in _IMG_EXTS:
                    self.samples.append((img_path, idx))
