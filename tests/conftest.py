import pytest
import numpy as np
import tensorflow as tf


@pytest.fixture(scope="session")
def tflite_model_path(tmp_path_factory):
    """
    Builds a minimal Keras model, converts it to .tflite,
    saves it to a temp directory, and returns the path.
    'session' scope = built ONCE for the entire test run, reused by all tests.
    """
    # 1. Build the simplest possible model with correct shape
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(224, 224, 3)),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(5, activation="softmax")  # 5 disease classes
    ])

    # 2. Convert to TFLite (no training needed — random weights are fine)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    # 3. Save to a temp path pytest manages for us
    tmp_dir = tmp_path_factory.mktemp("models")
    model_path = tmp_dir / "test_model.tflite"
    model_path.write_bytes(tflite_model)

    return model_path