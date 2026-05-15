from __future__ import annotations

import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.core.config import TrainingConfig
from src.evaluation.metrics import compute_metrics, evaluate_loader
from src.training.callbacks import EarlyStopping, ModelCheckpoint


class BaseTrainer:
    """Training loop shared by all classical DL models."""

    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        model_name: str,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler=None,
    ):
        self.model = model
        self.config = config
        self.model_name = model_name
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.scaler = torch.amp.GradScaler("cuda", enabled=config.amp and self.device.type == "cuda")

        self.early_stop = EarlyStopping(
            patience=config.patience, min_delta=config.min_delta, mode="max"
        )
        self.checkpoint = ModelCheckpoint(
            config.checkpoint_dir, model_name, mode="max"
        )

        self.history: list[dict] = []
        Path(config.results_dir).mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def fit(self) -> dict:
        print(f"\n{'='*60}")
        print(f"Training {self.model_name} on {self.device}")
        print(f"  Params: {self.model.num_params():,} total | {self.model.num_trainable_params():,} trainable")
        print(f"  Epochs: {self.config.epochs}  |  Batch: {self.config.batch_size}  |  Patience: {self.config.patience}")
        print(f"{'='*60}\n")

        for epoch in range(1, self.config.epochs + 1):
            t0 = time.time()
            train_loss = self._train_epoch()
            val_metrics = self._val_epoch()
            elapsed = time.time() - t0

            val_f1 = val_metrics["macro_f1"]
            saved = self.checkpoint.step(val_f1, self.model)
            stop = self.early_stop.step(val_f1)

            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_f1)
                else:
                    self.scheduler.step()

            row = {
                "epoch": epoch,
                "train_loss": round(train_loss, 4),
                "val_accuracy": round(val_metrics["accuracy"], 4),
                "val_macro_f1": round(val_f1, 4),
                "val_weighted_f1": round(val_metrics["weighted_f1"], 4),
                "elapsed_s": round(elapsed, 1),
            }
            self.history.append(row)

            flag = " *" if saved else ""
            print(
                f"Epoch {epoch:3d}/{self.config.epochs} | "
                f"loss {train_loss:.4f} | "
                f"acc {val_metrics['accuracy']:.4f} | "
                f"macro-F1 {val_f1:.4f}{flag} | "
                f"{elapsed:.0f}s"
            )

            if stop:
                print(f"\nEarly stopping at epoch {epoch} (best macro-F1={self.early_stop.best:.4f})")
                break

        # Load best and evaluate on val
        self.checkpoint.load_best(self.model)
        y_true, y_pred = evaluate_loader(self.model, self.val_loader, self.device)
        final_metrics = compute_metrics(y_true, y_pred)

        self._save_results(final_metrics)
        self._print_final(final_metrics)
        return final_metrics

    def fit_hpo(self, trial) -> float:
        """Training loop with Optuna pruning support. Returns best val macro-F1."""
        import optuna

        best_f1 = 0.0
        for epoch in range(1, self.config.epochs + 1):
            self._train_epoch()
            val_metrics = self._val_epoch()
            val_f1 = val_metrics["macro_f1"]
            best_f1 = max(best_f1, val_f1)

            trial.report(val_f1, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

            print(
                f"  [t{trial.number}] ep {epoch}/{self.config.epochs} "
                f"f1={val_f1:.4f} best={best_f1:.4f}",
                flush=True,
            )

        return best_f1

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _train_epoch(self) -> float:
        self.model.train()
        total_loss = 0.0
        amp_enabled = self.config.amp and self.device.type == "cuda"
        for imgs, labels in self.train_loader:
            imgs, labels = imgs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=amp_enabled):
                logits = self.model(imgs)
                loss = self.criterion(logits, labels)
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            total_loss += loss.item() * imgs.size(0)
        return total_loss / len(self.train_loader.dataset)

    def _val_epoch(self) -> dict:
        y_true, y_pred = evaluate_loader(self.model, self.val_loader, self.device)
        return compute_metrics(y_true, y_pred)

    def _save_results(self, metrics: dict) -> None:
        out_dir = Path(self.config.results_dir) / self.model_name
        out_dir.mkdir(parents=True, exist_ok=True)

        # History
        with open(out_dir / "history.json", "w") as f:
            json.dump(self.history, f, indent=2)

        # Final metrics (exclude heavy 'report' string for JSON)
        summary = {k: v for k, v in metrics.items() if k != "report"}
        with open(out_dir / "val_metrics.json", "w") as f:
            json.dump(summary, f, indent=2)

        # Full classification report
        with open(out_dir / "classification_report.txt", "w") as f:
            f.write(metrics["report"])

    def _print_final(self, metrics: dict) -> None:
        print(f"\n{'='*60}")
        print(f"Final results — {self.model_name}")
        print(f"{'='*60}")
        print(f"  Accuracy    : {metrics['accuracy']:.4f}")
        print(f"  Macro-F1    : {metrics['macro_f1']:.4f}")
        print(f"  Weighted-F1 : {metrics['weighted_f1']:.4f}")
        print(f"\n{metrics['report']}")
