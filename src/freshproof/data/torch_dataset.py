"""PyTorch dataset that reads photos listed in labels.csv.

Photos are looked up by photo_id inside <root>/raw with a .jpg, .jpeg or .png
extension, and opened upright (phone rotation applied). Import this module only
when torch is installed.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import Dataset
from torchvision import transforms

from .images import find_image, list_photos, load_oriented
from .schema import Grade, PhotoRecord

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

__all__ = ["PhotoDataset", "build_transform", "find_image"]


def build_transform(image_size: int, train: bool) -> transforms.Compose:
    steps: list = [transforms.Resize((image_size, image_size))]
    if train:
        # Mild changes only. Strong colour shifts would hide real spoilage signs.
        steps += [
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
        ]
    steps += [transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    return transforms.Compose(steps)


class PhotoDataset(Dataset):
    def __init__(
        self,
        records: list[PhotoRecord],
        root: str | Path,
        produce_types: list[str],
        image_size: int = 224,
        train: bool = False,
    ):
        self.records = records
        self.raw_dir = Path(root) / "raw"
        self.produce_index = {name: i for i, name in enumerate(produce_types)}
        self.grade_index = {grade: i for i, grade in enumerate(Grade)}
        self.transform = build_transform(image_size, train)

        unknown = {r.produce_type for r in records} - set(self.produce_index)
        if unknown:
            raise ValueError(f"produce types not in the class list: {sorted(unknown)}")

        self.paths = list_photos(self.raw_dir)
        missing = [r.photo_id for r in records if r.photo_id not in self.paths]
        if missing:
            shown = ", ".join(missing[:5])
            raise FileNotFoundError(
                f"{len(missing)} photos listed in labels.csv are not in {self.raw_dir}: {shown}"
            )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, int]:
        record = self.records[index]
        image = self.transform(load_oriented(self.paths[record.photo_id]))
        return (
            image,
            self.produce_index[record.produce_type],
            self.grade_index[record.grade],
        )
