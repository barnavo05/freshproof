"""Local labeling tool: paint the defect on a photo, pick a grade, save.

Runs on your own computer. Photos are read from <data_root>/raw, masks are
written to <data_root>/masks and rows go into <data_root>/labels.csv.
"""

from __future__ import annotations

import base64
import binascii
import io
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from PIL import Image
from pydantic import BaseModel

from freshproof.data.images import list_photos, oriented_size
from freshproof.data.labels import load_records, upsert_record
from freshproof.data.schema import Grade, LightCondition, PhotoRecord

DEFAULT_PRODUCE = ["tomato", "banana", "potato", "onion", "mango", "apple", "orange", "capsicum"]
MAX_MASK_BYTES = 8 * 1024 * 1024
INDEX_FILE = Path(__file__).with_name("index.html")


class LabelPayload(BaseModel):
    produce_type: str
    grade: Grade
    light: LightCondition
    phone_model: str
    item_id: Optional[str] = None
    days_until_spoiled: Optional[float] = None
    labeler: str = "unknown"
    # PNG as base64 or a data URL. The alpha channel marks the defect.
    # null means "no defect mask", and any existing mask is removed.
    mask_png_base64: Optional[str] = None


def _decode_mask(data: str, size: tuple[int, int]) -> Optional[Image.Image]:
    payload = data.split(",", 1)[1] if data.startswith("data:") else data
    try:
        raw = base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError):
        raise HTTPException(status_code=400, detail="mask is not valid base64")
    if len(raw) > MAX_MASK_BYTES:
        raise HTTPException(status_code=413, detail="mask is too large")
    try:
        with Image.open(io.BytesIO(raw)) as mask:
            rgba = mask.convert("RGBA")
    except (OSError, ValueError, SyntaxError):
        raise HTTPException(status_code=400, detail="mask is not a valid image")
    painted = np.asarray(rgba)[..., 3] > 127
    if not painted.any():
        return None
    resized = Image.fromarray((painted * 255).astype(np.uint8)).resize(size, Image.NEAREST)
    return resized if np.asarray(resized).any() else None


def create_labeling_app(data_root: str | Path) -> FastAPI:
    root = Path(data_root)
    raw_dir = root / "raw"
    masks_dir = root / "masks"
    labels_path = root / "labels.csv"

    app = FastAPI(title="FreshProof labeler")

    def photo_path(photo_id: str) -> Path:
        paths = list_photos(raw_dir)
        if photo_id not in paths:
            raise HTTPException(status_code=404, detail="photo not found")
        return paths[photo_id]

    def records_by_id() -> dict[str, PhotoRecord]:
        if not labels_path.exists():
            return {}
        try:
            return {r.photo_id: r for r in load_records(labels_path)}
        except ValueError as err:
            raise HTTPException(status_code=500, detail=f"labels.csv problem: {err}")

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return INDEX_FILE.read_text(encoding="utf-8")

    @app.get("/api/options")
    def options() -> dict:
        seen = {r.produce_type for r in records_by_id().values()}
        return {
            "produce_types": sorted(seen | set(DEFAULT_PRODUCE)),
            "grades": [g.value for g in Grade],
            "lights": [light.value for light in LightCondition],
        }

    @app.get("/api/photos")
    def photos() -> list[dict]:
        labeled = records_by_id()
        return [
            {"photo_id": photo_id, "labeled": photo_id in labeled}
            for photo_id in list_photos(raw_dir)
        ]

    @app.get("/api/photos/{photo_id}/image")
    def image(photo_id: str) -> FileResponse:
        return FileResponse(photo_path(photo_id))

    @app.get("/api/photos/{photo_id}/label")
    def get_label(photo_id: str) -> dict:
        photo_path(photo_id)
        record = records_by_id().get(photo_id)
        has_mask = (masks_dir / f"{photo_id}.png").exists()
        if record is None:
            return {"label": None, "has_mask": has_mask}
        return {
            "label": {
                "photo_id": record.photo_id,
                "produce_type": record.produce_type,
                "grade": record.grade.value,
                "light": record.light.value,
                "phone_model": record.phone_model,
                "item_id": record.item_id,
                "days_until_spoiled": record.days_until_spoiled,
                "labeler": record.labeler,
            },
            "has_mask": has_mask,
        }

    @app.get("/api/photos/{photo_id}/mask")
    def get_mask(photo_id: str) -> FileResponse:
        photo_path(photo_id)
        mask_file = masks_dir / f"{photo_id}.png"
        if not mask_file.exists():
            raise HTTPException(status_code=404, detail="no mask for this photo")
        return FileResponse(mask_file, media_type="image/png")

    @app.post("/api/photos/{photo_id}/label")
    def save_label(photo_id: str, payload: LabelPayload) -> dict:
        path = photo_path(photo_id)
        if not payload.phone_model.strip():
            raise HTTPException(status_code=400, detail="phone_model is required")

        mask_image = None
        if payload.mask_png_base64:
            mask_image = _decode_mask(payload.mask_png_base64, oriented_size(path))

        record = PhotoRecord(
            photo_id=photo_id,
            produce_type=payload.produce_type.strip().lower(),
            grade=payload.grade,
            light=payload.light,
            phone_model=payload.phone_model.strip(),
            defect_mask_path=f"masks/{photo_id}.png" if mask_image is not None else None,
            days_until_spoiled=payload.days_until_spoiled,
            labeler=payload.labeler.strip() or "unknown",
            item_id=(payload.item_id or "").strip() or None,
        )
        try:
            record.validate()
        except ValueError as err:
            raise HTTPException(status_code=400, detail=str(err))

        mask_file = masks_dir / f"{photo_id}.png"
        if mask_image is not None:
            masks_dir.mkdir(parents=True, exist_ok=True)
            mask_image.save(mask_file)
        elif mask_file.exists():
            mask_file.unlink()

        try:
            upsert_record(labels_path, record)
        except ValueError as err:
            raise HTTPException(status_code=500, detail=f"labels.csv problem: {err}")
        return {"saved": True, "has_mask": mask_image is not None}

    return app
