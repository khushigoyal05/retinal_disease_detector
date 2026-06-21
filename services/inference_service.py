"""Concrete InferenceEngine implementation backed by a TFLite model.

On the Raspberry Pi this uses the lightweight tflite-runtime package.
On a dev machine where only the full TensorFlow package is installed
(needed anyway for training), it transparently falls back to
tensorflow.lite's interpreter, which exposes an identical API.
"""

from pathlib import Path
from typing import List, Optional

import numpy as np

try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:  # tflite-runtime isn't installed (e.g. dev Windows box)
    from tensorflow.lite.python.interpreter import Interpreter

import config
from core.interfaces import InferenceEngine
from core.models import ClassPrediction


class TFLiteInferenceEngine(InferenceEngine):
    """Runs a trained .tflite model and returns per-class confidence scores."""

    def __init__(self, model_path: Path = config.MODEL_PATH) -> None:
        self._model_path = model_path
        self._interpreter: Optional[Interpreter] = None
        self._input_index: Optional[int] = None
        self._output_index: Optional[int] = None
        self._input_dtype = None
        self._expected_input_shape: Optional[tuple] = None

    def load(self) -> None:
        if not self._model_path.exists():
            raise FileNotFoundError(
                f"TFLite model not found at {self._model_path}. "
                "Train and export a model, or update config.MODEL_PATH."
            )

        self._interpreter = Interpreter(model_path=str(self._model_path))
        self._interpreter.allocate_tensors()

        input_details = self._interpreter.get_input_details()
        output_details = self._interpreter.get_output_details()

        self._input_index = input_details[0]["index"]
        self._output_index = output_details[0]["index"]
        self._input_dtype = input_details[0]["dtype"]
        self._expected_input_shape = tuple(input_details[0]["shape"])

    def predict(self, image: np.ndarray) -> List[ClassPrediction]:
        if self._interpreter is None:
            raise RuntimeError("InferenceEngine.load() must be called before predict().")

        if image.shape != self._expected_input_shape:
            raise ValueError(
                f"Input shape {image.shape} does not match model's expected "
                f"shape {self._expected_input_shape}. Check config.INPUT_IMAGE_SIZE "
                "matches the training notebook."
            )

        if image.dtype != self._input_dtype:
            image = image.astype(self._input_dtype)

        self._interpreter.set_tensor(self._input_index, image)
        self._interpreter.invoke()
        raw_output = self._interpreter.get_tensor(self._output_index)[0]

        predictions = [
            ClassPrediction(label=label, confidence=float(score))
            for label, score in zip(config.DISEASE_CLASSES, raw_output)
        ]
        predictions.sort(key=lambda p: p.confidence, reverse=True)
        return predictions