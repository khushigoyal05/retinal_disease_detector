"""
Abstract interfaces that decouple the rest of the app from concrete
hardware/library implementations (Dependency Inversion Principle).
"""

from abc import ABC, abstractmethod
from pathlib import Path
import numpy as np

from core.models import PredictionResult


class ImageSource(ABC):
    """
    Anything that can produce an image for the pipeline: the Pi Camera,
    a file-upload dialog, or — in tests — a fake source with no hardware.
    """

    @abstractmethod
    def get_image(self) -> Path:
        """
        Acquire an image and return the filesystem path it was saved to.
        Returning a path (not raw pixels) keeps a permanent, auditable
        record of every image the system has ever processed.
        """
        raise NotImplementedError

from typing import List
from core.models import ClassPrediction   # add this import, remove PredictionResult import

class InferenceEngine(ABC):
    @abstractmethod
    def load(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def predict(self, image: np.ndarray) -> List[ClassPrediction]:
        """Run inference and return predictions, sorted by confidence descending."""
        raise NotImplementedError