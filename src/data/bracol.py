from __future__ import annotations

import pandas as pd
from pathlib import Path

from .base_dataset import BaseDataset
from .label_mapper import LABEL_TO_IDX

# predominant_stress codes in dataset.csv
_STRESS_MAP: dict[int, str] = {
    0: "healthy",
    1: "miner",
    2: "rust",
    3: "phoma",
    4: "cercospora",
    # 5 = multi-stress (co-infection) — excluded, no single predominant class
}

# BRACOL unzips to a nested structure:
# data/raw/bracol/coffee-datasets/coffee-datasets/leaf/
_LEAF_SUBDIRS = [
    "coffee-datasets/coffee-datasets/leaf",
    "coffee-datasets/leaf",
    "leaf",
    ".",
]


def _find_leaf_root(root: Path) -> Path:
    for sub in _LEAF_SUBDIRS:
        candidate = root / sub
        if (candidate / "dataset.csv").exists():
            return candidate
    raise FileNotFoundError(
        f"Could not find dataset.csv under {root}. "
        "Tried: " + ", ".join(_LEAF_SUBDIRS)
    )


class BracolDataset(BaseDataset):
    """BRACOL — Brazilian Arabica Coffee Leaf dataset.

    DOI: 10.17632/yy2k5y8mxg.1  |  ~1,685 usable images  |  5 classes
    root should point to data/raw/bracol/

    Labels come from dataset.csv (predominant_stress column).
    Class 5 (multi-stress co-infection) is excluded.
    """

    def _load(self) -> None:
        leaf_root = _find_leaf_root(self.root)
        csv_path = leaf_root / "dataset.csv"
        img_dir = leaf_root / "images"

        df = pd.read_csv(csv_path)

        for _, row in df.iterrows():
            stress = int(row["predominant_stress"])
            if stress not in _STRESS_MAP:
                continue  # skip class 5 (multi-stress)

            label_name = _STRESS_MAP[stress]
            label_idx = LABEL_TO_IDX[label_name]

            img_path = img_dir / f"{int(row['id'])}.jpg"
            if img_path.exists():
                self.samples.append((img_path, label_idx))
