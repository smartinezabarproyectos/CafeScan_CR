from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Sequence

import numpy as np
from torch.utils.data import Dataset, Subset

from .bracol import BracolDataset
from .jmuben import JMuBENDataset
from .label_mapper import UNIFIED_LABELS
from .transforms import train_transforms, val_transforms, test_transforms

DATASET_TAGS = {
    "bracol": 0,
    "jmuben": 1,
    "jmuben2": 2,
}

class TaggedRecord:
    __slots__ = ("path", "label", "tag")

    def __init__(self, path: Path, label: int, tag: int):
        self.path = path
        self.label = label
        self.tag = tag

_RECORDS_CACHE: dict[str, list[TaggedRecord]] = {}

def _collect_records(data_root: str | Path) -> list[TaggedRecord]:
    key = str(Path(data_root).resolve())
    if key in _RECORDS_CACHE:
        return _RECORDS_CACHE[key]

    root = Path(data_root)
    records: list[TaggedRecord] = []

    bracol = BracolDataset(root / "bracol")
    for path, lbl in bracol.samples:
        records.append(TaggedRecord(path, lbl, DATASET_TAGS["bracol"]))

    for name in ("jmuben", "jmuben2"):
        sub = root / name
        if sub.exists():
            ds = JMuBENDataset(sub)
            tag = DATASET_TAGS[name]
            for path, lbl in ds.samples:
                records.append(TaggedRecord(path, lbl, tag))

    valid = _filter_valid(records)
    _RECORDS_CACHE[key] = valid
    return valid

def _filter_valid(records: list[TaggedRecord]) -> list[TaggedRecord]:
    from PIL import Image, ImageFile, UnidentifiedImageError
    ImageFile.LOAD_TRUNCATED_IMAGES = True

    valid, bad_paths = [], []
    for rec in records:
        try:
            with Image.open(rec.path) as img:
                img.verify()
            valid.append(rec)
        except (OSError, UnidentifiedImageError, Exception):
            bad_paths.append(rec.path)

    if bad_paths:
        print(f"[data] Removed {len(bad_paths)} unreadable images:", flush=True)
        for p in bad_paths:
            print(f"  [BAD] {p}", flush=True)

    return valid

def stratified_split(
    data_root: str | Path,
    ratios: tuple[float, float, float] = (0.70, 0.15, 0.15),
    seed: int = 42,
    size: int = 224,
) -> tuple[Dataset, Dataset, Dataset]:
    assert abs(sum(ratios) - 1.0) < 1e-6, "Ratios must sum to 1"
    rng = np.random.default_rng(seed)

    records = _collect_records(data_root)

    groups: dict[tuple[int, int], list[int]] = defaultdict(list)
    for i, rec in enumerate(records):
        groups[(rec.label, rec.tag)].append(i)

    train_idx, val_idx, test_idx = [], [], []
    for (lbl, tag), idxs in groups.items():
        arr = np.array(idxs)
        rng.shuffle(arr)
        n = len(arr)
        n_train = int(n * ratios[0])
        n_val   = int(n * ratios[1])
        train_idx.extend(arr[:n_train].tolist())
        val_idx.extend(arr[n_train:n_train + n_val].tolist())
        test_idx.extend(arr[n_train + n_val:].tolist())

    train_ds = _IndexedDataset(records, train_idx, train_transforms(size))
    val_ds   = _IndexedDataset(records, val_idx,   val_transforms(size))
    test_ds  = _IndexedDataset(records, test_idx,  test_transforms(size))

    return train_ds, val_ds, test_ds

def split_summary(ds: "_IndexedDataset") -> dict:
    from collections import Counter
    label_counts = Counter(ds.records[i].label for i in ds.indices)
    tag_counts   = Counter(ds.records[i].tag   for i in ds.indices)
    return {
        "total": len(ds.indices),
        "by_class": {UNIFIED_LABELS[k]: v for k, v in sorted(label_counts.items())},
        "by_source": {k: v for k, v in sorted(tag_counts.items())},
    }

class _IndexedDataset(Dataset):

    def __init__(self, records: list[TaggedRecord], indices: list[int], transform):
        self.records = records
        self.indices = indices
        self.transform = transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        from PIL import Image, ImageFile
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        rec = self.records[self.indices[i]]
        img = Image.open(rec.path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, rec.label
