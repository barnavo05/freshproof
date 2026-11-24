import pytest

from freshproof.data import Grade, LightCondition, load_records

HEADER = (
    "photo_id,produce_type,grade,light,phone_model,item_id,"
    "defect_mask_path,days_until_spoiled,labeler\n"
)


def _write(tmp_path, body, header=HEADER):
    path = tmp_path / "labels.csv"
    path.write_text(header + body, encoding="utf-8")
    return path


def test_loads_valid_rows(tmp_path):
    path = _write(
        tmp_path,
        "p1,Tomato,B,daylight,phoneA,t1,masks/p1.png,3.5,me\n"
        "p2,tomato,A,shade,phoneA,,,,\n",
    )
    records = load_records(path)
    assert len(records) == 2
    assert records[0].produce_type == "tomato"
    assert records[0].grade is Grade.B
    assert records[0].light is LightCondition.DAYLIGHT
    assert records[0].days_until_spoiled == 3.5
    assert records[1].item_id is None
    assert records[1].defect_mask_path is None
    assert records[1].labeler == "unknown"


def test_missing_column_is_reported(tmp_path):
    path = _write(tmp_path, "", header="photo_id,produce_type\n")
    with pytest.raises(ValueError, match="missing columns"):
        load_records(path)


def test_bad_grade_names_the_line(tmp_path):
    path = _write(tmp_path, "p1,tomato,Z,daylight,phoneA,,,,\n")
    with pytest.raises(ValueError, match="line 2"):
        load_records(path)


def test_duplicate_photo_id_rejected(tmp_path):
    path = _write(
        tmp_path,
        "p1,tomato,B,daylight,phoneA,,,,\n" "p1,tomato,B,daylight,phoneA,,,,\n",
    )
    with pytest.raises(ValueError, match="duplicate"):
        load_records(path)


def test_grade_a_with_mask_is_caught_on_load(tmp_path):
    path = _write(tmp_path, "p1,tomato,A,daylight,phoneA,,m.png,,\n")
    with pytest.raises(ValueError, match="line 2"):
        load_records(path)
