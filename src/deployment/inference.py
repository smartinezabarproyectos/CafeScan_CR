from __future__ import annotations

from pathlib import Path

import torch
import torchvision.transforms as T
from PIL import Image

from src.data.label_mapper import UNIFIED_LABELS
from src.data.transforms import test_transforms
from src.utils.io import load_checkpoint


class Predictor:
    """Single-image inference wrapper used by API and Streamlit app."""

    def __init__(self, model: torch.nn.Module, checkpoint_path: str | Path, img_size: int = 224):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = load_checkpoint(model, checkpoint_path, device=str(self.device))
        self.model.to(self.device).eval()
        self.transform = test_transforms(img_size)

    @torch.no_grad()
    def predict(self, image: Image.Image) -> dict:
        x = self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)
        logits = self.model(x)
        probs = torch.softmax(logits, dim=-1).squeeze().cpu().tolist()
        pred_idx = int(torch.argmax(logits).item())
        return {
            "class": UNIFIED_LABELS[pred_idx],
            "confidence": round(probs[pred_idx], 4),
            "probabilities": {cls: round(p, 4) for cls, p in zip(UNIFIED_LABELS, probs)},
        }

    @torch.no_grad()
    def predict_path(self, path: str | Path) -> dict:
        return self.predict(Image.open(path))
