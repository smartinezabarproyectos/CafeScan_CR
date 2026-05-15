from __future__ import annotations

from pathlib import Path

from .base_dataset import BaseDataset
from .label_mapper import label_to_idx

# RoCoLe — Robusta Coffee Leaf dataset (Ecuador)
# DOI: 10.17632/c5yvn32j78.2
# Classes: healthy, rust (no cercospora/miner/phoma)
# Currently excluded from the unified corpus: only 2 of 5 classes present,
# and images are Robusta (C. canephora) vs Arabica in other datasets —
# potential domain shift that would require separate analysis.
_IMG_EXTS = {".jpg", ".jpeg", ".png"}


class RoCoLeDataset(BaseDataset):
    """RoCoLe — Robusta Coffee Leaf dataset (Ecuador).

    root should point to data/raw/rocole/
    Expected structure: root/<class_folder>/*.jpg
    """

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
