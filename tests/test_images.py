import pytest
from PIL import Image

from freshproof.data.images import find_image, list_photos, load_oriented, oriented_size


def _save_with_orientation(path, size, orientation):
    exif = Image.Exif()
    exif[0x0112] = orientation
    Image.new("RGB", size, (10, 120, 30)).save(path, exif=exif)


def test_list_photos_finds_supported_files_only(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    Image.new("RGB", (4, 4)).save(raw / "b.jpg")
    Image.new("RGB", (4, 4)).save(raw / "a.png")
    Image.new("RGB", (4, 4)).save(raw / "c.JPG")
    (raw / "notes.txt").write_text("hi")
    assert list(list_photos(raw)) == ["a", "b", "c"]


def test_list_photos_missing_folder_is_empty(tmp_path):
    assert list_photos(tmp_path / "nope") == {}


def test_find_image_and_missing(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    Image.new("RGB", (4, 4)).save(raw / "p1.jpg")
    assert find_image(raw, "p1").name == "p1.jpg"
    with pytest.raises(FileNotFoundError):
        find_image(raw, "nope")


def test_rotated_photo_is_made_upright(tmp_path):
    path = tmp_path / "rot.jpg"
    _save_with_orientation(path, (60, 40), 6)
    assert load_oriented(path).size == (40, 60)
    assert oriented_size(path) == (40, 60)


def test_normal_orientation_unchanged(tmp_path):
    path = tmp_path / "ok.jpg"
    _save_with_orientation(path, (60, 40), 1)
    assert load_oriented(path).size == (60, 40)
    assert oriented_size(path) == (60, 40)
