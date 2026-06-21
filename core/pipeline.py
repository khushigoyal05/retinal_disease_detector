"""
Orchestrates one full cycle: acquire image -> preprocess -> infer -> result.
Knows the *sequence* of steps, not *how* any individual step works.
"""

import time

from core.interfaces import ImageSource, InferenceEngine
from core.models import PredictionResult
from utils.image_utils import load_and_preprocess
import config


class InferencePipeline:
    """
    Both dependencies are injected through the constructor — never created
    internally. This is what lets us substitute fakes in tests, and swap
    real implementations (camera vs file, TFLite vs ONNX) with zero changes
    to this class.
    """

    def __init__(self, image_source: ImageSource, inference_engine: InferenceEngine):
        self._image_source = image_source
        self._inference_engine = inference_engine

    def run(self) -> PredictionResult:
        image_path = self._image_source.get_image()
        preprocessed = load_and_preprocess(image_path, config.INPUT_IMAGE_SIZE)

        start = time.perf_counter()
        predictions = self._inference_engine.predict(preprocessed)
        elapsed_ms = (time.perf_counter() - start) * 1000

        return PredictionResult(
            image_path=image_path,
            predictions=predictions,
            inference_time_ms=elapsed_ms,
        )