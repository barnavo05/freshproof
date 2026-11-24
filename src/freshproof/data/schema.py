"""Label schema for the FreshProof dataset.

One PhotoRecord describes one photo. Keeping this strict from day one makes
the dataset releasable and the experiments reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class Grade(str, Enum):
    A = "A"
    B = "B"
    C = "C"
    REJECT = "reject"


class LightCondition(str, Enum):
    DAYLIGHT = "daylight"
    SHADE = "shade"
    TUBE_LIGHT = "tube_light"
    SHOP_LIGHT = "shop_light"
    EVENING = "evening"
    OTHER = "other"


@dataclass(frozen=True)
class PhotoRecord:
    photo_id: str
    produce_type: str
    grade: Grade
    light: LightCondition
    phone_model: str
    # Path to a binary mask of the true defect area. None means no visible defect.
    defect_mask_path: Optional[str] = None
    # Ground truth from daily re-photography. None if not tracked.
    days_until_spoiled: Optional[float] = None
    # Which labeler produced this record, for agreement checks.
    labeler: str = "unknown"

    def validate(self) -> None:
        if not self.photo_id:
            raise ValueError("photo_id is required")
        if not self.produce_type:
            raise ValueError("produce_type is required")
        if self.days_until_spoiled is not None and self.days_until_spoiled < 0:
            raise ValueError("days_until_spoiled cannot be negative")
        if self.grade is Grade.A and self.defect_mask_path is not None:
            raise ValueError("a grade A item should not have a defect mask")
