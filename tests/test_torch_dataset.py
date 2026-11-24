import pytest

pytest.importorskip("torch")
pytest.importorskip("torchvision")

from PIL import Image  # noqa: E402

from freshproof.data import Grade, LightCondition, PhotoRecord  # noqa: E402
from freshproof.data.torch_dataset import PhotoDataset, find_image  # noqa: E402


def _record(photo_id="p1", produce="tomato", grade=Grade.B):
    return PhotoRecord(
        photo_id=photo_id,
        produce_type=produce,
        grade=grade,
        light=LightCondition.DAYLIGHT,
        phone_model="phone",
    )


def _make_image(root, name):
    raw = root / "raw"
    raw.mkdir(exist_ok=True)
    Image.new("RGB", (80, 60), (200, 40, 40)).save(raw / f"{name}.jpg")


def test_item_has_expected_shape_and_labels(tmp_path):
    _make_image(tmp_path, "p1")
    ds = PhotoDataset([_record()], tmp_path, ["banana", "tomato"], image_size=32)
    image, produce, grade = ds[0]
    assert tuple(image.shape) == (3, 32, 32)
    assert produce == 1
    assert grade == list(Grade).index(Grade.B)


def test_missing_image_raises(tmp_path):
    (tmp_path / "raw").mkdir()
    with pytest.raises(FileNotFoundError):
        find_image(tmp_path / "raw", "nope")


def test_unknown_produce_type_rejected(tmp_path):
    with pytest.raises(ValueError):
        PhotoDataset([_record(produce="mango")], tmp_path, ["tomato"])


def test_missing_photo_is_reported_up_front(tmp_path):
    (tmp_path / "raw").mkdir()
    with pytest.raises(FileNotFoundError, match="p1"):
        PhotoDataset([_record()], tmp_path, ["tomato"])


def test_rotated_phone_photo_is_used_upright(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    exif = Image.Exif()
    exif[0x0112] = 6  # stored sideways, should display rotated
    Image.new("RGB", (80, 40), (200, 40, 40)).save(raw / "p1.jpg", exif=exif)
    ds = PhotoDataset([_record()], tmp_path, ["tomato"], image_size=32)
    image, _, _ = ds[0]
    assert tuple(image.shape) == (3, 32, 32)
