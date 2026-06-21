"""
Unit tests for InferencePipeline using fake ImageSource/InferenceEngine.
Zero Pi hardware, zero trained model required — that's the entire point
of coding against interfaces instead of concrete classes.
"""

from pathlib import Path
from typing import List

import numpy as np
import pytest
from PIL import Image

from core.interfaces import ImageSource, InferenceEngine
from core.models import ClassPrediction
from core.pipeline import InferencePipeline


class FakeImageSource(ImageSource):
    """Returns a path to a pre-made test image instead of touching hardware."""

    def __init__(self, image_path: Path):
        self._image_path = image_path

    def get_image(self) -> Path:
        return self._image_path


class FakeInferenceEngine(InferenceEngine):
    """Returns a hardcoded prediction instead of running a real model."""

    def load(self) -> None:
        pass

    def predict(self, image: np.ndarray) -> List[ClassPrediction]:
        assert image.shape == (1, 224, 224, 3), f"unexpected shape: {image.shape}"
        return [
            ClassPrediction(label="Diabetic Retinopathy", confidence=0.87),
            ClassPrediction(label="Normal", confidence=0.10),
        ]


@pytest.fixture
def sample_image(tmp_path: Path) -> Path:
    """Creates a throwaway 300x300 RGB JPEG as test input."""
    path = tmp_path / "sample.jpg"
    Image.new("RGB", (300, 300), color=(120, 60, 60)).save(path)
    return path


def test_pipeline_returns_sorted_top_prediction(sample_image: Path):
    pipeline = InferencePipeline(
        image_source=FakeImageSource(sample_image),
        inference_engine=FakeInferenceEngine(),
    )

    result = pipeline.run()

    assert result.top_prediction.label == "Diabetic Retinopathy"
    assert result.top_prediction.confidence == pytest.approx(0.87)
    assert result.is_confident is True
    assert result.inference_time_ms >= 0
    assert result.image_path == sample_image