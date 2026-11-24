"""Read data/labels.csv into validated PhotoRecord objects."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Optional

from .schema import Grade, LightCondition, PhotoRecord

REQUIRED_COLUMNS = ("photo_id", "produce_type", "grade", "light", "phone_model")


def _blank_to_none(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    value = value.strip()
    return value or None


def _parse_row(row: dict[str, str]) -> PhotoRecord:
    days = _blank_to_none(row.get("days_until_spoiled"))
    return PhotoRecord(
        photo_id=row["photo_id"].strip(),
        produce_type=row["produce_type"].strip().lower(),
        grade=Grade(row["grade"].strip()),
        light=LightCondition(row["light"].strip()),
        phone_model=row["phone_model"].strip(),
        defect_mask_path=_blank_to_none(row.get("defect_mask_path")),
        days_until_spoiled=float(days) if days is not None else None,
        labeler=_blank_to_none(row.get("labeler")) or "unknown",
        item_id=_blank_to_none(row.get("item_id")),
    )


def load_records(csv_path: str | Path) -> list[PhotoRecord]:
    """Load and validate every row. Errors name the file line that is wrong."""
    path = Path(csv_path)
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        columns = reader.fieldnames or []
        missing = [c for c in REQUIRED_COLUMNS if c not in columns]
        if missing:
            raise ValueError(f"{path.name} is missing columns: {', '.join(missing)}")

        records: list[PhotoRecord] = []
        seen: set[str] = set()
        for line_no, row in enumerate(reader, start=2):
            try:
                record = _parse_row(row)
                record.validate()
            except (ValueError, KeyError) as err:
                raise ValueError(f"{path.name} line {line_no}: {err}") from err
            if record.photo_id in seen:
                raise ValueError(
                    f"{path.name} line {line_no}: duplicate photo_id {record.photo_id}"
                )
            seen.add(record.photo_id)
            records.append(record)
    return records
