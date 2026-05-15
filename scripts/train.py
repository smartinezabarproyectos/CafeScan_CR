from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.config import TrainingConfig
from src.utils.seed import set_seed

MODEL_REGISTRY = {
    "efficientnet_b0": ("src.models.classical.efficientnet", "EfficientNetB0"),
    "resnet50":        ("src.models.classical.resnet",       "ResNet50"),
    "vit":             ("src.models.classical.vit",          "ViTSmall"),
    "mobilenet":       ("src.models.classical.mobilenet",    "MobileNetV3"),
}

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",       required=True, choices=list(MODEL_REGISTRY))
    p.add_argument("--data_root",   default="data/raw")
    p.add_argument("--epochs",      type=int,   default=30)
    p.add_argument("--batch_size",  type=int,   default=32)
    p.add_argument("--lr",          type=float, default=1e-4)
    p.add_argument("--num_workers", type=int,   default=8)
    p.add_argument("--seed",        type=int,   default=42)
    p.add_argument("--no_pretrain", action="store_true")
    return p.parse_args()

def main():
    args = parse_args()
    set_seed(args.seed)

    config = TrainingConfig(
        data_root=args.data_root,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    module_path, class_name = MODEL_REGISTRY[args.model]
    import importlib
    module = importlib.import_module(module_path)
    ModelClass = getattr(module, class_name)
    model = ModelClass(num_classes=config.num_classes, pretrained=not args.no_pretrain)

    from src.training.classical_trainer import build_trainer
    trainer, test_ds = build_trainer(model, args.model, config)
    trainer.fit()

    import torch
    from torch.utils.data import DataLoader
    from src.evaluation.metrics import compute_metrics, evaluate_loader

    test_loader = DataLoader(test_ds, batch_size=config.batch_size * 2,
                             shuffle=False, num_workers=config.num_workers)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    y_true, y_pred = evaluate_loader(model, test_loader, device)
    test_metrics = compute_metrics(y_true, y_pred)

    import json
    from pathlib import Path
    out = Path(config.results_dir) / args.model / "test_metrics.json"
    summary = {k: v for k, v in test_metrics.items() if k != "report"}
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nTest set results:")
    print(f"  Accuracy  : {test_metrics['accuracy']:.4f}")
    print(f"  Macro-F1  : {test_metrics['macro_f1']:.4f}")
    print(test_metrics["report"])

if __name__ == "__main__":
    main()
