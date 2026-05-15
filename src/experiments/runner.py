from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.core.config import TrainingConfig
from src.core.registry import list_models
from src.evaluation.comparator import load_all_results, plot_comparison, plot_per_class_f1, print_summary_table
from src.training.classical_trainer import build_trainer
from src.utils.seed import set_seed

ALL_MODELS = list_models()

def run_model(name: str, config: TrainingConfig) -> None:
    from src.core.registry import build_model

    model = build_model(name, num_classes=config.num_classes, pretrained=True)
    trainer, _ = build_trainer(model, name, config)
    trainer.fit()

def main():
    p = argparse.ArgumentParser(description="Train QuantumCafe-CR models")
    p.add_argument("--models", nargs="+", default=None,
                   help="Model names to train. Defaults to all models.")
    p.add_argument("--data_root", default="data/raw")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--weight_decay", type=float, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--patience", type=int, default=None)
    p.add_argument("--label_smoothing", type=float, default=None)
    args = p.parse_args()

    set_seed(args.seed)

    cfg_kwargs: dict = dict(
        data_root=args.data_root,
        epochs=args.epochs,
        seed=args.seed,
    )
    if args.lr is not None:
        cfg_kwargs["lr"] = args.lr
    if args.weight_decay is not None:
        cfg_kwargs["weight_decay"] = args.weight_decay
    if args.batch_size is not None:
        cfg_kwargs["batch_size"] = args.batch_size
    if args.patience is not None:
        cfg_kwargs["patience"] = args.patience
    if args.label_smoothing is not None:
        cfg_kwargs["label_smoothing"] = args.label_smoothing

    config = TrainingConfig(**cfg_kwargs)

    models_to_run = args.models if args.models else ALL_MODELS

    for name in models_to_run:
        print(f"\n{'#'*60}\n# {name}\n{'#'*60}", flush=True)
        run_model(name, config)

    print("\n\n=== FINAL COMPARISON ===", flush=True)
    results_path = Path(config.results_dir)
    df = load_all_results(results_path)
    if not df.empty:
        print_summary_table(df)
        fig_dir = results_path / "figures"
        fig_dir.mkdir(parents=True, exist_ok=True)
        plot_comparison(df, save_path=fig_dir / "model_comparison.png")
        plot_per_class_f1(df, save_path=fig_dir / "per_class_f1.png")

if __name__ == "__main__":
    main()
