"""
Framework-agnostic data structures shared across the whole application.
"""

from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime
from typing import List


@dataclass(frozen=True)
class ClassPrediction:
    """One disease class with its predicted probability."""
    label: str
    confidence: float  # 0.0 - 1.0


@dataclass(frozen=True)
class PredictionResult:
    """
    The structured output of a single inference run. Frozen (immutable)
    because a result is a historical fact once computed — it should be
    stored and displayed, never mutated.
    """
    image_path: Path
    predictions: List[ClassPrediction]   # sorted, highest confidence first
    inference_time_ms: float
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def top_prediction(self) -> ClassPrediction:
        return self.predictions[0]

    @property
    def is_confident(self) -> bool:
        from config import CONFIDENCE_THRESHOLD
        return self.top_prediction.confidence >= CONFIDENCE_THRESHOLD