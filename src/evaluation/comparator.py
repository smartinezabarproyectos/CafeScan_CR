from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data.label_mapper import UNIFIED_LABELS

def load_all_results(results_dir: str | Path) -> pd.DataFrame:
    root = Path(results_dir)
    rows = []
    for model_dir in sorted(root.iterdir()):
        metrics_file = model_dir / "val_metrics.json"
        if not metrics_file.exists():
            continue
        with open(metrics_file) as f:
            m = json.load(f)
        row = {
            "model": model_dir.name,
            "accuracy": m.get("accuracy", 0),
            "macro_f1": m.get("macro_f1", 0),
            "weighted_f1": m.get("weighted_f1", 0),
            "macro_precision": m.get("macro_precision", 0),
            "macro_recall": m.get("macro_recall", 0),
        }

        for i, cls in enumerate(UNIFIED_LABELS):
            pcf = m.get("per_class_f1", [])
            row[f"f1_{cls}"] = pcf[i] if i < len(pcf) else 0.0
        rows.append(row)
    return pd.DataFrame(rows).set_index("model")

def plot_comparison(df: pd.DataFrame, save_path: str | Path | None = None) -> None:
    metrics = ["accuracy", "macro_f1", "weighted_f1"]
    n = len(metrics)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))

    colors = plt.cm.tab10(np.linspace(0, 1, len(df)))
    for ax, metric in zip(axes, metrics):
        bars = ax.barh(df.index, df[metric], color=colors)
        ax.set_xlim(0, 1)
        ax.set_title(metric.replace("_", " ").title())
        ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=9)

    plt.suptitle("Model Comparison — Val Set", fontsize=14, y=1.02)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()

def plot_per_class_f1(df: pd.DataFrame, save_path: str | Path | None = None) -> None:
    cls_cols = [f"f1_{c}" for c in UNIFIED_LABELS]
    subset = df[cls_cols].rename(columns=lambda c: c.replace("f1_", ""))

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(UNIFIED_LABELS))
    width = 0.8 / len(subset)
    colors = plt.cm.tab10(np.linspace(0, 1, len(subset)))

    for i, (model, row) in enumerate(subset.iterrows()):
        ax.bar(x + i * width, row.values, width, label=model, color=colors[i])

    ax.set_xticks(x + width * len(subset) / 2)
    ax.set_xticklabels(UNIFIED_LABELS)
    ax.set_ylabel("F1 Score")
    ax.set_title("Per-Class F1 — All Models")
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left")
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()

def print_summary_table(df: pd.DataFrame) -> None:
    cols = ["accuracy", "macro_f1", "weighted_f1", "macro_precision", "macro_recall"]
    print(df[cols].round(4).to_string())
