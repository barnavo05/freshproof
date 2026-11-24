"""PyTorch dataset that reads photos listed in labels.csv.

Photos are looked up by photo_id inside <root>/raw with a .jpg, .jpeg or .png
extension. Import this module only when torch is installed.
"""

from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from .schema import Grade, PhotoRecord

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def find_image(raw_dir: Path, photo_id: str) -> Path:
    for ext in IMAGE_EXTENSIONS:
        candidate = raw_dir / f"{photo_id}{ext}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"no image found for photo_id {photo_id} in {raw_dir}")


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

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, int]:
        record = self.records[index]
        path = find_image(self.raw_dir, record.photo_id)
        with Image.open(path) as img:
            image = self.transform(img.convert("RGB"))
        return (
            image,
            self.produce_index[record.produce_type],
            self.grade_index[record.grade],
        )
