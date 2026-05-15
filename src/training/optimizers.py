from __future__ import annotations

import torch
import torch.nn as nn

def build_adamw(model: nn.Module, lr: float, backbone_lr_mult: float, weight_decay: float):
    return torch.optim.AdamW(
        model.param_groups(lr, backbone_lr_mult),
        weight_decay=weight_decay,
    )

def build_cosine_schedule(optimizer, epochs: int, steps_per_epoch: int, warmup_epochs: int = 2):
    warmup_steps = warmup_epochs * steps_per_epoch
    total_steps = epochs * steps_per_epoch

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        import math
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
