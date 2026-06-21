import numpy as np
import pytest
from services.inference_service import TFLiteInferenceEngine
from core.models import ClassPrediction
import config


# ── Test 1 ──────────────────────────────────────────────────────────────────
def test_load_succeeds_with_valid_model(tflite_model_path):
    """load() should complete without errors given a real .tflite file."""
    engine = TFLiteInferenceEngine(model_path=tflite_model_path)
    engine.load()  # should not raise


# ── Test 2 ──────────────────────────────────────────────────────────────────
def test_load_raises_if_model_missing(tmp_path):
    """load() should raise FileNotFoundError if the path doesn't exist."""
    engine = TFLiteInferenceEngine(model_path=tmp_path / "ghost.tflite")
    with pytest.raises(FileNotFoundError):
        engine.load()


# ── Test 3 ──────────────────────────────────────────────────────────────────
def test_predict_raises_if_not_loaded(tflite_model_path):
    """predict() before load() should raise RuntimeError."""
    engine = TFLiteInferenceEngine(model_path=tflite_model_path)
    dummy_image = np.zeros((1, 224, 224, 3), dtype=np.float32)
    with pytest.raises(RuntimeError):
        engine.predict(dummy_image)


# ── Test 4 ──────────────────────────────────────────────────────────────────
def test_predict_raises_on_wrong_shape(tflite_model_path):
    """predict() should raise ValueError if image shape doesn't match model."""
    engine = TFLiteInferenceEngine(model_path=tflite_model_path)
    engine.load()
    wrong_shape = np.zeros((1, 128, 128, 3), dtype=np.float32)  # wrong size
    with pytest.raises(ValueError):
        engine.predict(wrong_shape)


# ── Test 5 ──────────────────────────────────────────────────────────────────
def test_predict_returns_correct_labels(tflite_model_path):
    """predict() should return one ClassPrediction per disease class, correctly labeled."""
    engine = TFLiteInferenceEngine(model_path=tflite_model_path)
    engine.load()
    image = np.zeros((1, 224, 224, 3), dtype=np.float32)
    predictions = engine.predict(image)

    # Correct count
    assert len(predictions) == len(config.DISEASE_CLASSES)

    # All labels match config (order doesn't matter here — that's test 6)
    returned_labels = [p.label for p in predictions]
    assert returned_labels == sorted(returned_labels,
                                     key=lambda l: config.DISEASE_CLASSES.index(l))


# ── Test 6 ──────────────────────────────────────────────────────────────────
def test_predict_returns_sorted_by_confidence(tflite_model_path, monkeypatch):
    """
    predict() must return predictions sorted highest confidence first.
    We monkeypatch get_tensor() to return a known UNSORTED array,
    so this test would FAIL if the .sort() call were ever removed.
    """
    engine = TFLiteInferenceEngine(model_path=tflite_model_path)
    engine.load()

    # Force a known unsorted output: class 4 has highest confidence
    fake_output = np.array([[0.1, 0.05, 0.3, 0.15, 0.4]], dtype=np.float32)
    monkeypatch.setattr(engine._interpreter, "get_tensor",
                        lambda idx: fake_output)

    image = np.zeros((1, 224, 224, 3), dtype=np.float32)
    predictions = engine.predict(image)

    confidences = [p.confidence for p in predictions]
    assert confidences == sorted(confidences, reverse=True), \
        "Predictions must be sorted highest confidence first"