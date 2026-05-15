from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

def plot_training_history(history: list[dict], save_path: str | Path | None = None) -> None:
    epochs = [r["epoch"] for r in history]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, [r["train_loss"] for r in history], label="train loss")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(epochs, [r["val_macro_f1"] for r in history], label="val macro-F1")
    axes[1].plot(epochs, [r["val_accuracy"] for r in history], label="val accuracy")
    axes[1].set_title("Validation metrics")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()

def plot_class_distribution(counts: dict[str, int], title: str = "", save_path: str | Path | None = None) -> None:
    labels = list(counts.keys())
    values = list(counts.values())
    colors = ["#2ecc71", "#e74c3c", "#3498db", "#f39c12", "#9b59b6"]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(labels, values, color=colors[: len(labels)])
    ax.bar_label(bars)
    ax.set_title(title or "Class distribution")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
