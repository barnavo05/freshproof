"""HTTP API: send a photo, get a grade, the evidence, and a manual-check flag."""

from __future__ import annotations

import base64
import io

from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, ImageOps

from .overlay import render_overlay
from .predictor import Predictor

DEFAULT_MAX_BYTES = 5 * 1024 * 1024


def create_app(
    predictor: Predictor,
    abstain_threshold: float = 0.6,
    max_bytes: int = DEFAULT_MAX_BYTES,
) -> FastAPI:
    if not 0.0 <= abstain_threshold <= 1.0:
        raise ValueError("abstain_threshold must be between 0 and 1")

    app = FastAPI(title="FreshProof")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/scan")
    async def scan(file: UploadFile = File(...)) -> dict:
        data = await file.read(max_bytes + 1)
        if len(data) > max_bytes:
            raise HTTPException(status_code=413, detail="image is too large")
        try:
            image = Image.open(io.BytesIO(data))
            image.load()
            # Phone photos may be stored sideways with a rotation flag.
            image = ImageOps.exif_transpose(image).convert("RGB")
        except (OSError, ValueError, SyntaxError, Image.DecompressionBombError):
            raise HTTPException(status_code=400, detail="file is not a valid image")

        prediction = predictor.predict(image)
        overlay = None
        if prediction.heatmap is not None:
            overlay = base64.b64encode(render_overlay(image, prediction.heatmap)).decode("ascii")

        return {
            "produce_type": prediction.produce_type,
            "grade": prediction.grade,
            "confidence": round(prediction.confidence, 4),
            "needs_manual_check": prediction.confidence < abstain_threshold,
            "days_left": prediction.days_left,
            "evidence_overlay_png_base64": overlay,
        }

    return app
