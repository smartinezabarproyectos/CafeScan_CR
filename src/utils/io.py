from __future__ import annotations

import json
from pathlib import Path

import torch


def save_json(data: dict, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_json(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_checkpoint(model: torch.nn.Module, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def load_checkpoint(model: torch.nn.Module, path: str | Path, device: str = "cpu") -> torch.nn.Module:
    state = torch.load(path, map_location=device)
    model.load_state_dict(state)
    return model
