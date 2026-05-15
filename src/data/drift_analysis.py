from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
_SAMPLE_SIZE = 500          # images per dataset for statistics
_RESIZE = (128, 128)        # standardize before feature extraction


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def _load_sample(paths: list[Path], n: int, seed: int = 42) -> np.ndarray:
    """Load n random images, resize, return (n, 3*H*W) float32 array."""
    rng = np.random.default_rng(seed)
    chosen = rng.choice(len(paths), size=min(n, len(paths)), replace=False)
    feats = []
    for i in chosen:
        try:
            arr = np.array(
                Image.open(paths[i]).convert("RGB").resize(_RESIZE),
                dtype=np.float32,
            ) / 255.0
            feats.append(arr.flatten())
        except Exception:
            pass
    return np.stack(feats) if feats else np.empty((0, 3 * _RESIZE[0] * _RESIZE[1]))


def _collect_paths(root: Path) -> list[Path]:
    return [p for p in root.rglob("*") if p.suffix.lower() in _IMG_EXTS]


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def _rgb_stats(paths: list[Path], n: int = _SAMPLE_SIZE) -> dict:
    """Per-channel mean and std from a random sample."""
    rng = np.random.default_rng(0)
    chosen = rng.choice(len(paths), size=min(n, len(paths)), replace=False)
    means = {"r": [], "g": [], "b": []}
    for i in chosen:
        try:
            arr = np.array(
                Image.open(paths[i]).convert("RGB").resize(_RESIZE),
                dtype=np.float32,
            ) / 255.0
            means["r"].append(arr[:, :, 0].mean())
            means["g"].append(arr[:, :, 1].mean())
            means["b"].append(arr[:, :, 2].mean())
        except Exception:
            pass
    return {
        ch: {"mean": float(np.mean(v)), "std": float(np.std(v))}
        for ch, v in means.items()
    }


# ---------------------------------------------------------------------------
# Maximum Mean Discrepancy (linear kernel, unbiased estimator)
# ---------------------------------------------------------------------------

def _mmd_linear(X: np.ndarray, Y: np.ndarray) -> float:
    """Unbiased MMD² with linear kernel — O(n²) but fast on flat features."""
    n, m = len(X), len(Y)
    if n == 0 or m == 0:
        return float("nan")

    XX = (X @ X.T).mean()
    YY = (Y @ Y.T).mean()
    XY = (X @ Y.T).mean()
    return float(XX - 2 * XY + YY)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_drift_analysis(
    data_root: str | Path,
    sample_size: int = _SAMPLE_SIZE,
    output_dir: str | Path | None = None,
) -> pd.DataFrame:
    """Compare RGB statistics and MMD between BRACOL, JMuBEN, JMuBEN2.

    Returns a DataFrame with pairwise MMD scores and per-dataset RGB stats.
    """
    root = Path(data_root)
    datasets = {
        "bracol":  root / "bracol" / "coffee-datasets" / "coffee-datasets" / "leaf" / "images",
        "jmuben":  root / "jmuben",
        "jmuben2": root / "jmuben2",
    }

    print("Collecting paths...")
    paths = {}
    for name, droot in datasets.items():
        if droot.exists():
            p = _collect_paths(droot)
            paths[name] = p
            print(f"  {name}: {len(p)} images found")

    # RGB stats
    print("\nComputing RGB statistics...")
    stats_rows = []
    for name, p in paths.items():
        s = _rgb_stats(p, sample_size)
        stats_rows.append({
            "dataset": name,
            "R_mean": s["r"]["mean"], "R_std": s["r"]["std"],
            "G_mean": s["g"]["mean"], "G_std": s["g"]["std"],
            "B_mean": s["b"]["mean"], "B_std": s["b"]["std"],
        })
    stats_df = pd.DataFrame(stats_rows).set_index("dataset")
    print(stats_df.round(4).to_string())

    # MMD
    print("\nExtracting features for MMD (this may take a minute)...")
    features = {}
    for name, p in paths.items():
        features[name] = _load_sample(p, sample_size)
        print(f"  {name}: {features[name].shape}")

    names = list(features.keys())
    mmd_rows = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            score = _mmd_linear(features[a], features[b])
            mmd_rows.append({"pair": f"{a} vs {b}", "MMD2": round(score, 6)})
            print(f"  MMD² ({a} vs {b}) = {score:.6f}")

    mmd_df = pd.DataFrame(mmd_rows)

    if output_dir:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        stats_df.to_csv(out / "drift_rgb_stats.csv")
        mmd_df.to_csv(out / "drift_mmd.csv", index=False)
        print(f"\nResults saved to {out}")

    return mmd_df
