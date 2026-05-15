from __future__ import annotations

from pathlib import Path

import torch

from src.utils.io import load_checkpoint

def export_onnx(
    model: torch.nn.Module,
    checkpoint_path: str | Path,
    output_path: str | Path,
    img_size: int = 224,
    opset: int = 17,
) -> None:
    device = torch.device("cpu")
    load_checkpoint(model, checkpoint_path, device="cpu")
    model.to(device).eval()

    dummy = torch.randn(1, 3, img_size, img_size)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model, dummy, str(output_path),
        input_names=["image"],
        output_names=["logits"],
        dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=opset,
    )
    print(f"ONNX model saved to {output_path}")
