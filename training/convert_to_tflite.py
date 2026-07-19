# training/convert_to_tflite.py
from pathlib import Path
import tensorflow as tf

MODEL_PATH = Path(r"F:\capstone\retinal_disease_detector\training\checkpoints\phase2v2_best.keras")
OUT_PATH = Path(r"F:\capstone\retinal_disease_detector\models\retinal_model.tflite")
OUT_PATH.parent.mkdir(exist_ok=True)

model = tf.keras.models.load_model(MODEL_PATH)
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
OUT_PATH.write_bytes(tflite_model)
print(f"Done. Size: {OUT_PATH.stat().st_size / 1024 / 1024:.2f} MB")