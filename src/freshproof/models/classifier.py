"""Baseline model: one ResNet18 backbone, two heads (produce type and grade)."""

from __future__ import annotations

import torch
from torch import nn
from torchvision import models


class MultiHeadClassifier(nn.Module):
    def __init__(self, num_produce: int, num_grades: int, pretrained: bool = False):
        super().__init__()
        if num_produce < 1 or num_grades < 1:
            raise ValueError("num_produce and num_grades must be at least 1")
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        backbone = models.resnet18(weights=weights)
        features = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.produce_head = nn.Linear(features, num_produce)
        self.grade_head = nn.Linear(features, num_grades)

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(images)
        return self.produce_head(features), self.grade_head(features)
