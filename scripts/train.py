"""Train the baseline classifier.

Run from the repo root:
    PYTHONPATH=src python scripts/train.py --data-root data --epochs 10

Splits are made by item and saved next to the model, so any result can be
reproduced. The holdout set is only touched when you pass --eval-holdout, and
you should do that once, at the very end.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader

from freshproof.data import group_split, load_records
from freshproof.data.torch_dataset import PhotoDataset
from freshproof.models.classifier import MultiHeadClassifier


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--data-root", default=None, help="folder with raw/ and labels.csv")
    p.add_argument("--out", default="runs/baseline")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--val-fraction", type=float, default=0.15)
    p.add_argument("--pretrained", action="store_true", help="start from ImageNet weights")
    p.add_argument("--eval-holdout", action="store_true", help="score the holdout set once at the end")
    return p.parse_args()


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    correct_produce = correct_grade = total = 0
    for images, produce, grade in loader:
        images, produce, grade = images.to(device), produce.to(device), grade.to(device)
        produce_logits, grade_logits = model(images)
        correct_produce += (produce_logits.argmax(1) == produce).sum().item()
        correct_grade += (grade_logits.argmax(1) == grade).sum().item()
        total += images.size(0)
    if total == 0:
        return {"produce_acc": float("nan"), "grade_acc": float("nan")}
    return {"produce_acc": correct_produce / total, "grade_acc": correct_grade / total}


def main() -> None:
    args = parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    data_root = Path(args.data_root or cfg["data"]["root"])
    image_size = cfg["data"]["image_size"]

    records = load_records(data_root / "labels.csv")
    split = group_split(
        records,
        holdout_fraction=cfg["data"]["holdout_fraction"],
        val_fraction=args.val_fraction,
        seed=cfg["data"]["seed"],
    )
    if not split.train or not split.val:
        raise SystemExit("not enough data: train and validation splits must not be empty")

    produce_types = sorted({r.produce_type for r in records})
    num_grades = len(cfg["grades"])

    def loader(recs, train):
        ds = PhotoDataset(recs, data_root, produce_types, image_size, train=train)
        return DataLoader(ds, batch_size=args.batch_size, shuffle=train)

    train_loader = loader(split.train, True)
    val_loader = loader(split.val, False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiHeadClassifier(len(produce_types), num_grades, args.pretrained).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "split.json").write_text(
        json.dumps(
            {name: [r.photo_id for r in getattr(split, name)] for name in ("train", "val", "holdout")},
            indent=2,
        ),
        encoding="utf-8",
    )
    (out / "classes.json").write_text(
        json.dumps({"produce_types": produce_types, "grades": cfg["grades"]}, indent=2),
        encoding="utf-8",
    )

    best = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for images, produce, grade in train_loader:
            images, produce, grade = images.to(device), produce.to(device), grade.to(device)
            optimizer.zero_grad()
            produce_logits, grade_logits = model(images)
            loss = loss_fn(produce_logits, produce) + loss_fn(grade_logits, grade)
            loss.backward()
            optimizer.step()
            running += loss.item() * images.size(0)
        metrics = evaluate(model, val_loader, device)
        print(
            f"epoch {epoch:02d} loss {running / len(split.train):.4f} "
            f"val produce {metrics['produce_acc']:.3f} grade {metrics['grade_acc']:.3f}"
        )
        if metrics["grade_acc"] > best:
            best = metrics["grade_acc"]
            torch.save(model.state_dict(), out / "best.pt")

    if args.eval_holdout:
        model.load_state_dict(torch.load(out / "best.pt", map_location=device))
        final = evaluate(model, loader(split.holdout, False), device)
        (out / "holdout_metrics.json").write_text(json.dumps(final, indent=2), encoding="utf-8")
        print(f"holdout produce {final['produce_acc']:.3f} grade {final['grade_acc']:.3f}")


if __name__ == "__main__":
    main()
