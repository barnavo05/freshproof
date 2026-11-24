import pytest

from freshproof.data import Grade, LightCondition, PhotoRecord


def _rec(**kw):
    base = dict(
        photo_id="p1",
        produce_type="tomato",
        grade=Grade.B,
        light=LightCondition.DAYLIGHT,
        phone_model="test-phone",
    )
    base.update(kw)
    return PhotoRecord(**base)


def test_group_id_prefers_item_id():
    assert _rec(item_id="tomato_07").group_id == "tomato_07"


def test_group_id_falls_back_to_photo_id():
    assert _rec().group_id == "p1"


def test_negative_shelf_life_rejected():
    with pytest.raises(ValueError):
        _rec(days_until_spoiled=-1).validate()


def test_grade_a_with_mask_rejected():
    with pytest.raises(ValueError):
        _rec(grade=Grade.A, defect_mask_path="m.png").validate()
