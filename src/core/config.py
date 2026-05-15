from __future__ import annotations

from dataclasses import dataclass, field

@dataclass
class TrainingConfig:

    data_root: str = "data/raw"
    num_classes: int = 5
    img_size: int = 224
    num_workers: int = 2

    epochs: int = 30
    batch_size: int = 64
    seed: int = 42

    lr: float = 1e-4
    backbone_lr_mult: float = 0.1
    weight_decay: float = 1e-4

    scheduler: str = "cosine"
    warmup_epochs: int = 2

    label_smoothing: float = 0.1
    use_weighted_loss: bool = True

    patience: int = 8
    min_delta: float = 1e-4

    checkpoint_dir: str = "results/checkpoints"
    results_dir: str = "results"

    amp: bool = True

    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
