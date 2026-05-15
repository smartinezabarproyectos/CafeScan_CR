from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TrainingConfig:
    # Data
    data_root: str = "data/raw"
    num_classes: int = 5
    img_size: int = 224
    num_workers: int = 2   # Windows multiprocessing overhead — 2 es el sweet spot

    # Training loop
    epochs: int = 30
    batch_size: int = 64   # GPU can handle 64 comfortably with 8GB VRAM
    seed: int = 42

    # Optimizer
    lr: float = 1e-4
    backbone_lr_mult: float = 0.1   # backbone gets lr * this
    weight_decay: float = 1e-4

    # Scheduler
    scheduler: str = "cosine"       # "cosine" | "step"
    warmup_epochs: int = 2

    # Regularization
    label_smoothing: float = 0.1
    use_weighted_loss: bool = True

    # Early stopping
    patience: int = 8               # epochs without val macro-F1 improvement
    min_delta: float = 1e-4

    # Checkpointing
    checkpoint_dir: str = "results/checkpoints"
    results_dir: str = "results"

    # Mixed precision — enabled on GPU (Blackwell SM 12.0 supports bf16/fp16)
    amp: bool = True

    # Split ratios
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
