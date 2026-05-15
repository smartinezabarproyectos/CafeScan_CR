"""final_eval.py — Evaluacion completa en test set para todos los modelos.

Genera:
  - results/<model>/test_metrics.json
  - results/<model>/confusion_matrix.png
  - results/<model>/gradcam_<class>.png  (una imagen por clase)
  - results/figures/test_comparison.png
  - results/figures/test_per_class_f1.png
  - results/tables/test_summary.csv

Usage:
    python scripts/final_eval.py
    python scripts/final_eval.py --models efficientnet_b0 vit
    python scripts/final_eval.py --no_gradcam
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — saves figures without opening windows
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.core.config import TrainingConfig
from src.core.registry import build_model, list_models
from src.data.label_mapper import UNIFIED_LABELS
from src.data.splitter import stratified_split
from src.data.transforms import test_transforms
from src.evaluation.confusion_matrix import plot_confusion_matrix
from src.evaluation.metrics import compute_metrics, evaluate_loader
from src.utils.io import load_checkpoint
from src.utils.seed import set_seed


CKPT_DIR = Path("results/checkpoints")
RESULTS_DIR = Path("results")
FIG_DIR = RESULTS_DIR / "figures"
TABLE_DIR = RESULTS_DIR / "tables"


# ---------------------------------------------------------------------------
# Grad-CAM target layer per model
# ---------------------------------------------------------------------------

def _get_gradcam_layer(model, model_name: str):
    """Return the last conv/feature layer suitable for Grad-CAM."""
    if model_name == "efficientnet_b0":
        # Last block before global pooling
        blocks = list(model.backbone.children())
        for layer in reversed(blocks):
            if any(isinstance(m, torch.nn.Conv2d) for m in layer.modules()):
                return layer
        return blocks[-1]
    elif model_name == "mobilenet":
        # MobileNetV3: use last inverted residual block — avoids the inplace
        # HardSwish issue that lives in conv_head / act2 expansion layers
        return model.backbone.blocks[-1]
    elif model_name == "resnet50":
        return model.backbone.layer4
    elif model_name == "vit":
        return None  # attention-based; standard Grad-CAM not applicable
    return None


# ---------------------------------------------------------------------------
# Grad-CAM implementation (inline to avoid import of deleted quantum_viz)
# ---------------------------------------------------------------------------

class _GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.gradients = None
        self.activations = None
        target_layer.register_forward_hook(self._fwd_hook)
        target_layer.register_full_backward_hook(self._bwd_hook)

    def _fwd_hook(self, m, inp, out):
        # clone avoids issues when the layer output is a view modified inplace downstream
        self.activations = out.detach().clone()

    def _bwd_hook(self, m, gin, gout):
        # clone prevents "view modified inplace" errors in models with HardSwish / inplace ops
        self.gradients = gout[0].detach().clone()

    @torch.enable_grad()
    def __call__(self, x: torch.Tensor, class_idx: int | None = None) -> np.ndarray:
        self.model.eval()
        x = x.unsqueeze(0) if x.dim() == 3 else x
        x = x.clone().requires_grad_(True)
        logits = self.model(x)
        if class_idx is None:
            class_idx = logits.argmax(dim=1).item()
        self.model.zero_grad()
        logits[0, class_idx].backward()
        grads = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (grads * self.activations).sum(dim=1).squeeze()
        cam = torch.relu(cam).cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam

    def overlay(self, x: torch.Tensor, class_idx: int | None = None,
                save_path: Path | None = None) -> None:
        import cv2
        cam = self(x, class_idx)
        img = x.squeeze().permute(1, 2, 0).cpu().numpy()
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)

        heatmap = cv2.resize(cam, (img.shape[1], img.shape[0]))
        heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0
        overlay = 0.5 * img + 0.4 * heatmap

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(img);       axes[0].set_title("Original")
        axes[1].imshow(cam, cmap="jet"); axes[1].set_title("GradCAM")
        axes[2].imshow(np.clip(overlay, 0, 1)); axes[2].set_title("Overlay")
        for ax in axes:
            ax.axis("off")
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()


# ---------------------------------------------------------------------------
# Sample one image per class from test_ds
# ---------------------------------------------------------------------------

def _sample_per_class(test_ds, n_classes: int) -> dict[int, torch.Tensor]:
    """Return {class_idx: image_tensor} — first found per class."""
    from src.data.transforms import test_transforms as tf
    transform = tf(224)
    samples: dict[int, torch.Tensor] = {}
    for i in range(len(test_ds)):
        img, label = test_ds[i]
        if label not in samples:
            samples[label] = img
        if len(samples) == n_classes:
            break
    return samples


# ---------------------------------------------------------------------------
# Per-model evaluation
# ---------------------------------------------------------------------------

def eval_model(
    model_name: str,
    test_loader: DataLoader,
    test_ds,
    device: torch.device,
    run_gradcam: bool,
) -> dict:
    print(f"\n{'='*50}")
    print(f"Evaluating: {model_name}")
    print(f"{'='*50}")

    ckpt = CKPT_DIR / f"{model_name}_best.pt"
    if not ckpt.exists():
        print(f"  [SKIP] checkpoint not found: {ckpt}")
        return {}

    model = build_model(model_name, num_classes=len(UNIFIED_LABELS), pretrained=False)
    load_checkpoint(model, ckpt, device=str(device))
    model.to(device).eval()

    y_true, y_pred = evaluate_loader(model, test_loader, device)
    metrics = compute_metrics(y_true, y_pred)

    out_dir = RESULTS_DIR / model_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save test metrics
    summary = {k: v for k, v in metrics.items() if k != "report"}
    with open(out_dir / "test_metrics.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(out_dir / "test_classification_report.txt", "w", encoding="utf-8") as f:
        f.write(metrics["report"])

    # Confusion matrix
    plot_confusion_matrix(
        y_true, y_pred,
        title=f"{model_name} — Test Set",
        save_path=out_dir / "confusion_matrix.png",
    )
    plt.close("all")

    print(f"  Accuracy   : {metrics['accuracy']:.4f}")
    print(f"  Macro-F1   : {metrics['macro_f1']:.4f}")
    print(f"  Weighted-F1: {metrics['weighted_f1']:.4f}")
    print(f"\n{metrics['report']}")

    # Grad-CAM
    if run_gradcam:
        target_layer = _get_gradcam_layer(model, model_name)
        if target_layer is not None:
            cam = _GradCAM(model, target_layer)
            samples = _sample_per_class(test_ds, len(UNIFIED_LABELS))
            for cls_idx, img_tensor in samples.items():
                cls_name = UNIFIED_LABELS[cls_idx].replace(" ", "_")
                save_path = out_dir / f"gradcam_{cls_name}.png"
                try:
                    cam.overlay(img_tensor.to(device), class_idx=cls_idx, save_path=save_path)
                    print(f"  GradCAM saved: {save_path.name}")
                except Exception as e:
                    print(f"  GradCAM failed for class {cls_name}: {e}")
        else:
            print(f"  [INFO] GradCAM skipped for {model_name} (ViT — attention maps not supported)")

    return {
        "model": model_name,
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "weighted_f1": metrics["weighted_f1"],
        "macro_precision": metrics["macro_precision"],
        "macro_recall": metrics["macro_recall"],
        **{f"f1_{UNIFIED_LABELS[i]}": v for i, v in enumerate(metrics["per_class_f1"])},
    }


# ---------------------------------------------------------------------------
# Comparison figures
# ---------------------------------------------------------------------------

def _plot_test_comparison(df: pd.DataFrame) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    metrics = ["accuracy", "macro_f1", "weighted_f1"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    colors = plt.cm.tab10(np.linspace(0, 1, len(df)))

    for ax, metric in zip(axes, metrics):
        bars = ax.barh(df["model"], df[metric], color=colors)
        ax.set_xlim(0, 1)
        ax.set_title(metric.replace("_", " ").title())
        ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=9)

    plt.suptitle("Model Comparison — Test Set", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "test_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nFigura guardada: {FIG_DIR / 'test_comparison.png'}")


def _plot_per_class_f1(df: pd.DataFrame) -> None:
    cls_cols = [f"f1_{c}" for c in UNIFIED_LABELS]
    subset = df.set_index("model")[cls_cols].rename(columns=lambda c: c.replace("f1_", ""))

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(UNIFIED_LABELS))
    width = 0.8 / len(subset)
    colors = plt.cm.tab10(np.linspace(0, 1, len(subset)))

    for i, (model, row) in enumerate(subset.iterrows()):
        ax.bar(x + i * width, row.values, width, label=model, color=colors[i])

    ax.set_xticks(x + width * len(subset) / 2)
    ax.set_xticklabels(UNIFIED_LABELS, rotation=15, ha="right")
    ax.set_ylabel("F1 Score")
    ax.set_ylim(0, 1.05)
    ax.set_title("Per-Class F1 — Test Set")
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "test_per_class_f1.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Figura guardada: {FIG_DIR / 'test_per_class_f1.png'}")


# ---------------------------------------------------------------------------
# Overfitting check: val vs test
# ---------------------------------------------------------------------------

def _check_overfitting(test_df: pd.DataFrame, threshold: float = 0.05) -> None:
    """Load val_metrics.json for each model and compare with test macro_f1.

    Flags models where val - test gap exceeds threshold (default 5 pp).
    Saves results/tables/val_vs_test.csv.
    """
    print("\n" + "="*60)
    print("OVERFITTING CHECK — Val vs Test Macro-F1")
    print("="*60)

    rows = []
    for _, r in test_df.iterrows():
        model = r["model"]
        val_file = RESULTS_DIR / model / "val_metrics.json"
        if not val_file.exists():
            print(f"  [{model}] val_metrics.json not found — skipping")
            continue
        with open(val_file) as f:
            val = json.load(f)
        val_f1  = val.get("macro_f1", 0.0)
        test_f1 = r["macro_f1"]
        gap     = val_f1 - test_f1
        flag    = "OVERFIT?" if gap > threshold else "OK"
        rows.append({
            "model":   model,
            "val_f1":  round(val_f1,  4),
            "test_f1": round(test_f1, 4),
            "gap":     round(gap,     4),
            "status":  flag,
        })
        symbol = "!!" if flag == "OVERFIT?" else "  "
        print(f"  {symbol} {model:<20} val={val_f1:.4f}  test={test_f1:.4f}  gap={gap:+.4f}  [{flag}]")

    if rows:
        cmp_df = pd.DataFrame(rows)
        cmp_path = TABLE_DIR / "val_vs_test.csv"
        TABLE_DIR.mkdir(parents=True, exist_ok=True)
        cmp_df.to_csv(cmp_path, index=False)
        print(f"\n  Tabla guardada: {cmp_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Final test-set evaluation for all models")
    p.add_argument("--models", nargs="+", default=None,
                   help="Models to evaluate (default: all trained)")
    p.add_argument("--data_root", default="data/raw")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no_gradcam", action="store_true")
    args = p.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    config = TrainingConfig(data_root=args.data_root, seed=args.seed)
    _, _, test_ds = stratified_split(
        config.data_root,
        ratios=(config.train_ratio, config.val_ratio, config.test_ratio),
        seed=args.seed,
        size=config.img_size,
    )
    print(f"Test set: {len(test_ds)} samples")

    test_loader = DataLoader(
        test_ds,
        batch_size=64,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=config.num_workers > 0,
    )

    models_to_eval = args.models or [
        m for m in list_models()
        if (CKPT_DIR / f"{m}_best.pt").exists()
    ]

    rows = []
    for name in models_to_eval:
        row = eval_model(name, test_loader, test_ds, device, run_gradcam=not args.no_gradcam)
        if row:
            rows.append(row)

    if not rows:
        print("No models evaluated.")
        return

    df = pd.DataFrame(rows).sort_values("macro_f1", ascending=False)

    # Save CSV summary
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = TABLE_DIR / "test_summary.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nCSV guardado: {csv_path}")

    # Print final table
    print("\n" + "="*60)
    print("FINAL TEST SET RESULTS")
    print("="*60)
    cols = ["model", "accuracy", "macro_f1", "weighted_f1", "macro_precision", "macro_recall"]
    print(df[cols].round(4).to_string(index=False))

    # Val vs Test overfitting check
    _check_overfitting(df)

    # Figures
    _plot_test_comparison(df)
    _plot_per_class_f1(df)

    print("\n[OK] Evaluacion completa.")
    print(f"  Resultados en: {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
