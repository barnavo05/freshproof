"""What the API needs from a model, plus a simple stand-in for demos and tests.

The real trained model will implement the same Predictor interface, so the API
does not change when the model does.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class Prediction:
    produce_type: str
    grade: str
    confidence: float
    days_left: Optional[float] = None
    # 2D array in 0..1 showing where the grade came from. May be any size.
    heatmap: Optional[np.ndarray] = None


class Predictor(Protocol):
    def predict(self, image: Image.Image) -> Prediction: ...


class StubPredictor:
    """Demo only. Grades by the share of dark pixels. Not a real quality model."""

    DARK_LEVEL = 70
    THRESHOLDS = (0.01, 0.05, 0.15)
    GRADES = ("A", "B", "C", "reject")

    def predict(self, image: Image.Image) -> Prediction:
        gray = np.asarray(image.convert("L"), dtype=float)
        dark = gray < self.DARK_LEVEL
        fraction = float(dark.mean())
        grade = self.GRADES[sum(fraction >= t for t in self.THRESHOLDS)]
        margin = min(abs(fraction - t) for t in self.THRESHOLDS)
        confidence = float(min(0.99, 0.5 + margin * 10))
        return Prediction(
            produce_type="unknown",
            grade=grade,
            confidence=confidence,
            days_left=None,
            heatmap=dark.astype(float),
        )
