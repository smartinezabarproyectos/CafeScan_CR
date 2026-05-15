from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn


class GradCAM:
    """Gradient-weighted Class Activation Mapping for CNN-based models.

    Usage:
        cam = GradCAM(model, target_layer=model.backbone.blocks[-1])
        heatmap = cam(img_tensor, class_idx=2)
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.gradients: torch.Tensor | None = None
        self.activations: torch.Tensor | None = None

        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def __call__(self, x: torch.Tensor, class_idx: int | None = None) -> np.ndarray:
        self.model.eval()
        x = x.unsqueeze(0) if x.dim() == 3 else x
        x.requires_grad_(True)

        logits = self.model(x)
        if class_idx is None:
            class_idx = logits.argmax(dim=1).item()

        self.model.zero_grad()
        logits[0, class_idx].backward()

        grads = self.gradients.mean(dim=(2, 3), keepdim=True)  # GAP over spatial dims
        cam = (grads * self.activations).sum(dim=1).squeeze()
        cam = torch.relu(cam).cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam

    def overlay(
        self,
        x: torch.Tensor,
        class_idx: int | None = None,
        save_path: str | Path | None = None,
    ) -> None:
        cam = self(x, class_idx)
        img = x.squeeze().permute(1, 2, 0).cpu().numpy()
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)

        heatmap = cv2.resize(cam, (img.shape[1], img.shape[0]))
        heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0
        overlay = 0.5 * img + 0.4 * heatmap

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(img)
        axes[0].set_title("Original")
        axes[1].imshow(cam, cmap="jet")
        axes[1].set_title("GradCAM")
        axes[2].imshow(overlay)
        axes[2].set_title("Overlay")
        for ax in axes:
            ax.axis("off")
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()
