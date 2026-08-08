"""
services/cup_segmentation_engine.py

Runs the trained cup segmentation TFLite model.
Same load/predict pattern as TFLiteInferenceEngine (services/inference_service.py)
for consistency — separate class because output shape is a full mask
[1,224,224,1], not per-class scores.
"""

from pathlib import Path
from typing import Optional
import numpy as np

try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    from tensorflow.lite.python.interpreter import Interpreter

import config


class CupSegmentationEngine:
    """Runs cup_model.tflite → returns a 224x224 binary cup mask."""

    def __init__(self, model_path: Path = config.CDR_MODEL_PATH) -> None:
        self._model_path = model_path
        self._interpreter: Optional[Interpreter] = None
        self._input_index: Optional[int] = None
        self._output_index: Optional[int] = None
        self._input_dtype = None

    def load(self) -> None:
        if not self._model_path.exists():
            raise FileNotFoundError(f"CDR model not found at {self._model_path}")

        self._interpreter = Interpreter(model_path=str(self._model_path))
        self._interpreter.allocate_tensors()

        input_details = self._interpreter.get_input_details()
        output_details = self._interpreter.get_output_details()

        self._input_index = input_details[0]["index"]
        self._output_index = output_details[0]["index"]
        self._input_dtype = input_details[0]["dtype"]

    def predict(self, crop_rgb_224: np.ndarray) -> np.ndarray:
        """
        Args:
            crop_rgb_224: [1, 224, 224, 3] float32, RGB, normalized to [-1,1]
                (same contract as the disease classifier).

        Returns:
            [224, 224] binary mask (0/1), cup pixels = 1.
        """
        if self._interpreter is None:
            raise RuntimeError("CupSegmentationEngine.load() must be called before predict().")

        if crop_rgb_224.dtype != self._input_dtype:
            crop_rgb_224 = crop_rgb_224.astype(self._input_dtype)

        self._interpreter.set_tensor(self._input_index, crop_rgb_224)
        self._interpreter.invoke()
        raw_output = self._interpreter.get_tensor(self._output_index)[0, :, :, 0]

        return (raw_output > 0.5).astype(np.uint8)