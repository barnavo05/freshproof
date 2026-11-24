import base64
import io

import numpy as np
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402
from PIL import Image  # noqa: E402

from freshproof.data import load_records  # noqa: E402
from freshproof.labeling.app import create_labeling_app  # noqa: E402


def _mask_png(size=(30, 20), box=(5, 5, 15, 12)):
    """A painted-looking mask: opaque pixels inside the box, transparent elsewhere."""
    rgba = np.zeros((size[1], size[0], 4), dtype=np.uint8)
    x0, y0, x1, y1 = box
    rgba[y0:y1, x0:x1] = (255, 255, 255, 255)
    buf = io.BytesIO()
    Image.fromarray(rgba, "RGBA").save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _payload(**kw):
    base = {
        "produce_type": "Tomato",
        "grade": "B",
        "light": "daylight",
        "phone_model": "phoneA",
        "item_id": "t1",
        "days_until_spoiled": 3,
        "labeler": "me",
        "mask_png_base64": _mask_png(),
    }
    base.update(kw)
    return base


@pytest.fixture
def root(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    Image.new("RGB", (60, 40), (200, 40, 40)).save(raw / "a.jpg")
    Image.new("RGB", (60, 40), (30, 160, 60)).save(raw / "b.png")
    return tmp_path


@pytest.fixture
def client(root):
    return TestClient(create_labeling_app(root))


def test_index_page_served(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "FreshProof" in response.text


def test_lists_photos_unlabeled(client):
    assert client.get("/api/photos").json() == [
        {"photo_id": "a", "labeled": False},
        {"photo_id": "b", "labeled": False},
    ]


def test_options_include_defaults(client):
    body = client.get("/api/options").json()
    assert "tomato" in body["produce_types"]
    assert body["grades"] == ["A", "B", "C", "reject"]


def test_save_writes_label_and_mask_at_photo_size(client, root):
    response = client.post("/api/photos/a/label", json=_payload())
    assert response.status_code == 200
    assert response.json()["has_mask"] is True

    records = load_records(root / "labels.csv")
    assert len(records) == 1
    assert records[0].produce_type == "tomato"
    assert records[0].defect_mask_path == "masks/a.png"

    mask = np.asarray(Image.open(root / "masks" / "a.png"))
    assert mask.shape == (40, 60)  # same size as the photo, not the 30x20 canvas
    assert set(np.unique(mask)) == {0, 255}

    assert client.get("/api/photos").json()[0]["labeled"] is True


def test_saved_label_can_be_read_back(client):
    client.post("/api/photos/a/label", json=_payload())
    body = client.get("/api/photos/a/label").json()
    assert body["label"]["grade"] == "B"
    assert body["label"]["days_until_spoiled"] == 3
    assert body["has_mask"] is True
    assert client.get("/api/photos/a/mask").status_code == 200


def test_resave_updates_instead_of_duplicating(client, root):
    client.post("/api/photos/a/label", json=_payload(grade="B"))
    client.post("/api/photos/a/label", json=_payload(grade="C"))
    records = load_records(root / "labels.csv")
    assert len(records) == 1
    assert records[0].grade.value == "C"


def test_grade_a_with_mask_is_rejected_and_nothing_saved(client, root):
    response = client.post("/api/photos/a/label", json=_payload(grade="A"))
    assert response.status_code == 400
    assert not (root / "labels.csv").exists()
    assert not (root / "masks" / "a.png").exists()


def test_grade_a_without_mask_is_fine(client, root):
    response = client.post(
        "/api/photos/a/label", json=_payload(grade="A", mask_png_base64=None)
    )
    assert response.status_code == 200
    assert load_records(root / "labels.csv")[0].defect_mask_path is None


def test_null_mask_removes_existing_mask(client, root):
    client.post("/api/photos/a/label", json=_payload())
    assert (root / "masks" / "a.png").exists()
    client.post("/api/photos/a/label", json=_payload(mask_png_base64=None))
    assert not (root / "masks" / "a.png").exists()
    assert client.get("/api/photos/a/mask").status_code == 404


def test_empty_painting_counts_as_no_mask(client, root):
    empty = _mask_png(box=(0, 0, 0, 0))
    response = client.post("/api/photos/a/label", json=_payload(mask_png_base64=empty))
    assert response.status_code == 200
    assert response.json()["has_mask"] is False


def test_missing_phone_or_produce_rejected(client):
    assert client.post("/api/photos/a/label", json=_payload(phone_model=" ")).status_code == 400
    assert client.post("/api/photos/a/label", json=_payload(produce_type=" ")).status_code == 400


def test_invalid_grade_rejected(client):
    assert client.post("/api/photos/a/label", json=_payload(grade="Z")).status_code == 422


def test_bad_mask_data_rejected(client):
    assert client.post(
        "/api/photos/a/label", json=_payload(mask_png_base64="not base64!!")
    ).status_code == 400
    junk = base64.b64encode(b"just text").decode()
    assert client.post(
        "/api/photos/a/label", json=_payload(mask_png_base64=junk)
    ).status_code == 400


def test_unknown_photo_and_path_tricks_give_404(client):
    assert client.get("/api/photos/zzz/image").status_code == 404
    assert client.get("/api/photos/..%2Flabels/image").status_code == 404
    assert client.post("/api/photos/zzz/label", json=_payload()).status_code == 404


def test_rotated_phone_photo_gets_upright_mask(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    exif = Image.Exif()
    exif[0x0112] = 6
    Image.new("RGB", (60, 40), (200, 40, 40)).save(raw / "r.jpg", exif=exif)
    client = TestClient(create_labeling_app(tmp_path))
    response = client.post("/api/photos/r/label", json=_payload(mask_png_base64=_mask_png((20, 30), (2, 2, 10, 10))))
    assert response.status_code == 200
    mask = np.asarray(Image.open(tmp_path / "masks" / "r.png"))
    assert mask.shape == (60, 40)  # upright: 40 wide, 60 tall


def test_corrupt_labels_file_reports_clear_error(root):
    (root / "labels.csv").write_text("photo_id,produce_type\nx,tomato\n")
    client = TestClient(create_labeling_app(root))
    response = client.get("/api/photos")
    assert response.status_code == 500
    assert "labels.csv" in response.json()["detail"]
