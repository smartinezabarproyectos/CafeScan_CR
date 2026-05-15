from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from src.core.config import TrainingConfig
from src.data.splitter import stratified_split, split_summary
from src.models.base_models import BaseModel
from src.training.base_trainer import BaseTrainer
from src.training.losses import LabelSmoothingCrossEntropy, compute_class_weights_from_records


def build_trainer(
    model: BaseModel,
    model_name: str,
    config: TrainingConfig,
) -> BaseTrainer:
    """Wire up data loaders, optimizer, scheduler, and loss for a classical model."""

    train_ds, val_ds, test_ds = stratified_split(
        config.data_root,
        ratios=(config.train_ratio, config.val_ratio, config.test_ratio),
        seed=config.seed,
        size=config.img_size,
    )

    # Print split summary
    for name, ds in [("train", train_ds), ("val", val_ds), ("test", test_ds)]:
        s = split_summary(ds)
        print(f"{name}: {s['total']} samples | {s['by_class']}")

    # Class weights from train split
    weight = None
    if config.use_weighted_loss:
        weight = compute_class_weights_from_records(
            train_ds.records, train_ds.indices, config.num_classes
        )
        print(f"Class weights: {weight.tolist()}")

    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=config.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.batch_size * 2,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=config.num_workers > 0,
    )

    criterion = LabelSmoothingCrossEntropy(
        smoothing=config.label_smoothing,
        weight=weight,
    )

    optimizer = torch.optim.AdamW(
        model.param_groups(config.lr, config.backbone_lr_mult),
        weight_decay=config.weight_decay,
    )

    # Cosine schedule with linear warmup
    steps_per_epoch = len(train_loader)
    warmup_steps = config.warmup_epochs * steps_per_epoch
    total_steps = config.epochs * steps_per_epoch

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + torch.cos(torch.tensor(3.14159 * progress)).item())

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    return BaseTrainer(
        model=model,
        config=config,
        model_name=model_name,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
    ), test_ds
