from __future__ import annotations

from pathlib import Path

from .base_dataset import BaseDataset
from .label_mapper import label_to_idx

# JMuBEN (Cercospora, Rust, Phoma) + JMuBEN2 (Healthy, Miner)
# After extraction the structure is flat class folders under root.
# root should point to data/raw/jmuben/ (or jmuben2/)


class JMuBENDataset(BaseDataset):
    """JMuBEN + JMuBEN2 — Arabica Coffee Leaf dataset (Kenya).

    DOI JMuBEN:  10.17632/t2r6rszp5c.1  | Cercospora, Rust, Phoma
    DOI JMuBEN2: 10.17632/tgv3zb82nd.1  | Healthy, Miner
    root should point to data/raw/jmuben/ or data/raw/jmuben2/
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
                if img_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
                    self.samples.append((img_path, idx))
