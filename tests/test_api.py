import base64
import io

import numpy as np
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
pytest.importorskip("multipart")

from fastapi.testclient import TestClient  # noqa: E402
from PIL import Image  # noqa: E402

from freshproof.service.api import create_app  # noqa: E402
from freshproof.service.overlay import render_overlay  # noqa: E402
from freshproof.service.predictor import Prediction, StubPredictor  # noqa: E402


def _png_bytes(color=(200, 200, 200), size=(40, 30), dark_patch=False):
    img = Image.new("RGB", size, color)
    if dark_patch:
        for x in range(10, 30):
            for y in range(5, 25):
                img.putpixel((x, y), (20, 10, 5))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


class _FixedPredictor:
    def __init__(self, confidence):
        self.confidence = confidence

    def predict(self, image):
        return Prediction("tomato", "B", self.confidence, days_left=3.0, heatmap=None)


def _post(client, data, name="p.png", content_type="image/png"):
    return client.post("/scan", files={"file": (name, data, content_type)})


def test_health():
    client = TestClient(create_app(StubPredictor()))
    assert client.get("/health").json() == {"status": "ok"}


def test_clean_image_gets_grade_a_with_overlay():
    client = TestClient(create_app(StubPredictor()))
    body = _post(client, _png_bytes()).json()
    assert body["grade"] == "A"
    overlay = Image.open(io.BytesIO(base64.b64decode(body["evidence_overlay_png_base64"])))
    assert overlay.size == (40, 30)


def test_dark_patch_lowers_grade():
    client = TestClient(create_app(StubPredictor()))
    body = _post(client, _png_bytes(dark_patch=True)).json()
    assert body["grade"] in {"C", "reject"}


def test_low_confidence_flags_manual_check():
    client = TestClient(create_app(_FixedPredictor(0.4), abstain_threshold=0.6))
    body = _post(client, _png_bytes()).json()
    assert body["needs_manual_check"] is True
    assert body["days_left"] == 3.0
    assert body["evidence_overlay_png_base64"] is None


def test_high_confidence_does_not_flag():
    client = TestClient(create_app(_FixedPredictor(0.9), abstain_threshold=0.6))
    assert _post(client, _png_bytes()).json()["needs_manual_check"] is False


def test_rotated_phone_photo_overlay_is_upright():
    exif = Image.Exif()
    exif[0x0112] = 6
    buf = io.BytesIO()
    Image.new("RGB", (40, 30), (200, 200, 200)).save(buf, format="JPEG", exif=exif)
    client = TestClient(create_app(StubPredictor()))
    body = _post(client, buf.getvalue(), "p.jpg", "image/jpeg").json()
    overlay = Image.open(io.BytesIO(base64.b64decode(body["evidence_overlay_png_base64"])))
    assert overlay.size == (30, 40)


def test_non_image_rejected():
    client = TestClient(create_app(StubPredictor()))
    response = _post(client, b"this is not an image", "x.txt", "text/plain")
    assert response.status_code == 400


def test_oversized_upload_rejected():
    client = TestClient(create_app(StubPredictor(), max_bytes=100))
    response = _post(client, _png_bytes(size=(200, 200)))
    assert response.status_code == 413


def test_bad_threshold_rejected():
    with pytest.raises(ValueError):
        create_app(StubPredictor(), abstain_threshold=1.5)


def test_overlay_resizes_small_heatmap():
    img = Image.new("RGB", (40, 30), (100, 100, 100))
    heat = np.zeros((4, 4))
    heat[1:3, 1:3] = 1.0
    out = Image.open(io.BytesIO(render_overlay(img, heat)))
    assert out.size == (40, 30)


def test_overlay_rejects_non_2d_heatmap():
    with pytest.raises(ValueError):
        render_overlay(Image.new("RGB", (4, 4)), np.zeros((2, 2, 2)))
