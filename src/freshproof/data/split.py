"""Train, validation and holdout splits that keep each item in one split.

Daily photos of the same tomato look almost identical. If some land in train
and others in holdout, the scores look better than they really are. Splitting
by item (group_id) prevents that leak.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from .schema import PhotoRecord


@dataclass(frozen=True)
class Split:
    train: list[PhotoRecord]
    val: list[PhotoRecord]
    holdout: list[PhotoRecord]


def group_split(
    records: list[PhotoRecord],
    holdout_fraction: float = 0.15,
    val_fraction: float = 0.15,
    seed: int = 42,
) -> Split:
    if holdout_fraction < 0 or val_fraction < 0:
        raise ValueError("fractions must not be negative")
    if holdout_fraction + val_fraction >= 1:
        raise ValueError("holdout and validation fractions must sum to less than 1")

    groups: dict[str, list[PhotoRecord]] = {}
    for record in records:
        groups.setdefault(record.group_id, []).append(record)

    group_ids = sorted(groups)
    random.Random(seed).shuffle(group_ids)

    total = len(group_ids)
    n_holdout = round(total * holdout_fraction)
    n_val = min(round(total * val_fraction), total - n_holdout)

    holdout_ids = group_ids[:n_holdout]
    val_ids = group_ids[n_holdout : n_holdout + n_val]
    train_ids = group_ids[n_holdout + n_val :]

    def gather(ids: list[str]) -> list[PhotoRecord]:
        return [rec for gid in ids for rec in groups[gid]]

    return Split(train=gather(train_ids), val=gather(val_ids), holdout=gather(holdout_ids))
