"""Read data/labels.csv into validated PhotoRecord objects."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Optional

from .schema import Grade, LightCondition, PhotoRecord

REQUIRED_COLUMNS = ("photo_id", "produce_type", "grade", "light", "phone_model")
LABEL_COLUMNS = (
    "photo_id", "produce_type", "grade", "light", "phone_model",
    "item_id", "defect_mask_path", "days_until_spoiled", "labeler",
)


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


def _record_to_row(record: PhotoRecord) -> dict[str, str]:
    days = record.days_until_spoiled
    return {
        "photo_id": record.photo_id,
        "produce_type": record.produce_type,
        "grade": record.grade.value,
        "light": record.light.value,
        "phone_model": record.phone_model,
        "item_id": record.item_id or "",
        "defect_mask_path": record.defect_mask_path or "",
        "days_until_spoiled": "" if days is None else f"{days:g}",
        "labeler": record.labeler,
    }


def write_records(csv_path: str | Path, records: list[PhotoRecord]) -> None:
    """Write all records. Goes through a temp file so a crash cannot corrupt labels.csv."""
    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=LABEL_COLUMNS)
        writer.writeheader()
        for record in records:
            writer.writerow(_record_to_row(record))
    temp.replace(path)


def upsert_record(csv_path: str | Path, record: PhotoRecord) -> None:
    """Add the record, or replace the existing row with the same photo_id."""
    path = Path(csv_path)
    existing = load_records(path) if path.exists() else []
    merged = [record if r.photo_id == record.photo_id else r for r in existing]
    if not any(r.photo_id == record.photo_id for r in existing):
        merged.append(record)
    write_records(path, merged)
