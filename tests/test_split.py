import pytest

from freshproof.data import Grade, LightCondition, PhotoRecord, group_split


def _records(n_items=20, photos_per_item=5):
    out = []
    for i in range(n_items):
        for d in range(photos_per_item):
            out.append(
                PhotoRecord(
                    photo_id=f"i{i}_d{d}",
                    produce_type="tomato",
                    grade=Grade.B,
                    light=LightCondition.DAYLIGHT,
                    phone_model="phone",
                    item_id=f"item{i}",
                )
            )
    return out


def _groups(records):
    return {r.group_id for r in records}


def test_no_item_appears_in_two_splits():
    split = group_split(_records())
    train, val, hold = _groups(split.train), _groups(split.val), _groups(split.holdout)
    assert not (train & val)
    assert not (train & hold)
    assert not (val & hold)


def test_every_photo_is_assigned_once():
    records = _records()
    split = group_split(records)
    ids = [r.photo_id for r in split.train + split.val + split.holdout]
    assert sorted(ids) == sorted(r.photo_id for r in records)


def test_same_seed_gives_same_split():
    a = group_split(_records(), seed=7)
    b = group_split(_records(), seed=7)
    assert [r.photo_id for r in a.holdout] == [r.photo_id for r in b.holdout]


def test_different_seed_changes_split():
    a = group_split(_records(), seed=1)
    b = group_split(_records(), seed=2)
    assert [r.photo_id for r in a.holdout] != [r.photo_id for r in b.holdout]


def test_sizes_follow_fractions():
    split = group_split(_records(n_items=20), holdout_fraction=0.2, val_fraction=0.1)
    assert len(_groups(split.holdout)) == 4
    assert len(_groups(split.val)) == 2
    assert len(_groups(split.train)) == 14


def test_bad_fractions_rejected():
    with pytest.raises(ValueError):
        group_split(_records(), holdout_fraction=0.6, val_fraction=0.5)
    with pytest.raises(ValueError):
        group_split(_records(), holdout_fraction=-0.1)
