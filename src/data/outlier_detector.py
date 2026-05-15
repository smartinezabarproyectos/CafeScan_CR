from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image

from .bracol import BracolDataset
from .jmuben import JMuBENDataset
from .label_mapper import UNIFIED_LABELS

BLUR_THRESHOLD = 25.0
MIN_DIM_PX = 64
MIN_FILE_BYTES = 2_048
MAX_DARK_MEAN = 20.0

@dataclass
class OutlierReport:
    total_scanned: int = 0
    blurry: list[str] = field(default_factory=list)
    too_small: list[str] = field(default_factory=list)
    too_dark: list[str] = field(default_factory=list)
    tiny_file: list[str] = field(default_factory=list)
    unreadable: list[str] = field(default_factory=list)

    def all_flagged(self) -> set[str]:
        return (
            set(self.blurry)
            | set(self.too_small)
            | set(self.too_dark)
            | set(self.tiny_file)
            | set(self.unreadable)
        )

    def summary(self) -> str:
        flagged = len(self.all_flagged())
        lines = [
            f"Scanned : {self.total_scanned}",
            f"Flagged : {flagged} ({flagged/max(self.total_scanned,1)*100:.1f}%)",
            f"  blurry     : {len(self.blurry)}",
            f"  too small  : {len(self.too_small)}",
            f"  too dark   : {len(self.too_dark)}",
            f"  tiny file  : {len(self.tiny_file)}",
            f"  unreadable : {len(self.unreadable)}",
        ]
        return "\n".join(lines)

    def to_dataframe(self) -> pd.DataFrame:
        records = []
        for path in self.all_flagged():
            reasons = []
            if path in self.blurry:       reasons.append("blur")
            if path in self.too_small:    reasons.append("small_dim")
            if path in self.too_dark:     reasons.append("dark")
            if path in self.tiny_file:    reasons.append("tiny_file")
            if path in self.unreadable:   reasons.append("unreadable")
            records.append({"path": path, "reasons": ",".join(reasons)})
        return pd.DataFrame(records)

def _collect_all_paths(data_root: Path) -> list[Path]:
    paths = []
    bracol = BracolDataset(data_root / "bracol")
    paths.extend(p for p, _ in bracol.samples)
    for name in ("jmuben", "jmuben2"):
        sub = data_root / name
        if sub.exists():
            ds = JMuBENDataset(sub)
            paths.extend(p for p, _ in ds.samples)
    return paths

def detect_outliers(
    data_root: str | Path,
    blur_threshold: float = BLUR_THRESHOLD,
    min_dim: int = MIN_DIM_PX,
    min_bytes: int = MIN_FILE_BYTES,
    max_dark: float = MAX_DARK_MEAN,
    verbose: bool = True,
) -> OutlierReport:
    root = Path(data_root)
    paths = _collect_all_paths(root)
    report = OutlierReport(total_scanned=len(paths))

    for i, img_path in enumerate(paths):
        path_str = str(img_path)

        if verbose and i % 5000 == 0:
            print(f"  [{i}/{len(paths)}] scanning...")

        try:
            if img_path.stat().st_size < min_bytes:
                report.tiny_file.append(path_str)
                continue
        except OSError:
            report.unreadable.append(path_str)
            continue

        gray = cv2.imread(path_str, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            report.unreadable.append(path_str)
            continue

        h, w = gray.shape

        if w < min_dim or h < min_dim:
            report.too_small.append(path_str)

        if gray.mean() < max_dark:
            report.too_dark.append(path_str)

        lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        if lap_var < blur_threshold:
            report.blurry.append(path_str)

    return report
