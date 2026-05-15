from __future__ import annotations

from pathlib import Path

import torch


class EarlyStopping:
    """Stop training when monitored metric stops improving."""

    def __init__(self, patience: int = 8, min_delta: float = 1e-4, mode: str = "max"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best = float("-inf") if mode == "max" else float("inf")
        self.counter = 0
        self.triggered = False

    def step(self, value: float) -> bool:
        """Return True if training should stop."""
        improved = (
            value > self.best + self.min_delta
            if self.mode == "max"
            else value < self.best - self.min_delta
        )
        if improved:
            self.best = value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.triggered = True
        return self.triggered


class ModelCheckpoint:
    """Save best model by monitored metric."""

    def __init__(self, checkpoint_dir: str, model_name: str, mode: str = "max"):
        self.dir = Path(checkpoint_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path = self.dir / f"{model_name}_best.pt"
        self.mode = mode
        self.best = float("-inf") if mode == "max" else float("inf")

    def step(self, value: float, model: torch.nn.Module) -> bool:
        """Save checkpoint if value is best so far. Returns True if saved."""
        improved = value > self.best if self.mode == "max" else value < self.best
        if improved:
            self.best = value
            torch.save(model.state_dict(), self.path)
            return True
        return False

    def load_best(self, model: torch.nn.Module) -> None:
        model.load_state_dict(torch.load(self.path, map_location="cpu"))
