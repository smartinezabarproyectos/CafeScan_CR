"""FastAPI inference endpoint.

Usage:
    uvicorn src.deployment.api:app --reload
"""
from __future__ import annotations

import io
from pathlib import Path

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image

from src.core.registry import build_model
from src.deployment.inference import Predictor

app = FastAPI(title="QuantumCafe-CR API", version="1.0")

_PREDICTOR: Predictor | None = None


def get_predictor() -> Predictor:
    global _PREDICTOR
    if _PREDICTOR is None:
        model = build_model("efficientnet_b0", num_classes=5, pretrained=False)
        ckpt = Path("results/checkpoints/efficientnet_b0_best.pt")
        _PREDICTOR = Predictor(model, ckpt)
    return _PREDICTOR


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))
    result = get_predictor().predict(image)
    return JSONResponse(result)
