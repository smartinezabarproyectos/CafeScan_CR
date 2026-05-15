from __future__ import annotations

import torchvision.transforms as T

# Corpus-specific normalization stats (computed from 2000-image sample in EDA)
# Fall back to ImageNet if corpus stats not yet computed
CORPUS_MEAN = [0.4755, 0.6139, 0.4104]  # computed from 3000-image sample of unified corpus
CORPUS_STD  = [0.2139, 0.1834, 0.2165]

_NORMALIZE = T.Normalize(mean=CORPUS_MEAN, std=CORPUS_STD)


def train_transforms(size: int = 224) -> T.Compose:
    return T.Compose([
        T.RandomResizedCrop(size, scale=(0.7, 1.0)),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.RandomRotation(30),
        T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
        T.ToTensor(),
        _NORMALIZE,
        T.RandomErasing(p=0.2, scale=(0.02, 0.15)),
    ])


def val_transforms(size: int = 224) -> T.Compose:
    return T.Compose([
        T.Resize(int(size * 1.14)),   # slight oversize then center-crop
        T.CenterCrop(size),
        T.ToTensor(),
        _NORMALIZE,
    ])


def test_transforms(size: int = 224) -> T.Compose:
    """Identical to val — no randomness."""
    return val_transforms(size)


def update_normalization_stats(mean: list[float], std: list[float]) -> None:
    """Call after running scripts/compute_stats.py with the real corpus stats."""
    global CORPUS_MEAN, CORPUS_STD, _NORMALIZE
    CORPUS_MEAN = mean
    CORPUS_STD = std
    _NORMALIZE = T.Normalize(mean=mean, std=std)
