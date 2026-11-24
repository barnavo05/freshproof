import numpy as np
from PIL import Image

from freshproof.data import Grade, group_split, load_records
from freshproof.synthetic import generate_dataset, grade_from_fraction

SEVERITY = {Grade.A: 0, Grade.B: 1, Grade.C: 2, Grade.REJECT: 3}


def _make(tmp_path, **kw):
    kw.setdefault("items_per_type", 3)
    kw.setdefault("days", 5)
    kw.setdefault("size", 64)
    return generate_dataset(tmp_path, **kw)


def test_dataset_loads_with_real_loader(tmp_path):
    labels = _make(tmp_path)
    records = load_records(labels)
    assert len(records) == 3 * 3 * 5  # types x items x days


def test_masks_exist_only_for_defective_photos(tmp_path):
    records = load_records(_make(tmp_path))
    for rec in records:
        if rec.grade is Grade.A:
            assert rec.defect_mask_path is None
        else:
            assert (tmp_path / rec.defect_mask_path).exists()


def test_mask_matches_image_size_and_is_binary(tmp_path):
    records = load_records(_make(tmp_path))
    rec = next(r for r in records if r.defect_mask_path)
    image = Image.open(tmp_path / "raw" / f"{rec.photo_id}.jpg")
    mask = np.asarray(Image.open(tmp_path / rec.defect_mask_path))
    assert mask.shape == (image.height, image.width)
    assert set(np.unique(mask)) <= {0, 255}


def test_items_get_worse_and_days_left_shrink(tmp_path):
    records = load_records(_make(tmp_path))
    by_item = {}
    for rec in records:
        by_item.setdefault(rec.item_id, []).append(rec)
    for photos in by_item.values():
        photos.sort(key=lambda r: r.photo_id)
        severity = [SEVERITY[r.grade] for r in photos]
        assert severity == sorted(severity)
        days_left = [r.days_until_spoiled for r in photos]
        assert days_left == sorted(days_left, reverse=True)
        assert photos[0].grade is Grade.A


def test_same_seed_same_labels(tmp_path):
    a = _make(tmp_path / "a", seed=3).read_bytes()
    b = _make(tmp_path / "b", seed=3).read_bytes()
    assert a == b


def test_synthetic_data_works_with_item_split(tmp_path):
    records = load_records(_make(tmp_path, items_per_type=6))
    split = group_split(records, holdout_fraction=0.2, val_fraction=0.2)
    train = {r.item_id for r in split.train}
    holdout = {r.item_id for r in split.holdout}
    assert train and holdout and not (train & holdout)


def test_grade_thresholds():
    assert grade_from_fraction(0) is Grade.A
    assert grade_from_fraction(0.01) is Grade.B
    assert grade_from_fraction(0.1) is Grade.C
    assert grade_from_fraction(0.3) is Grade.REJECT
