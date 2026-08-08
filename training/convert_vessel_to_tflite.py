"""
Converts vessel_unet_best.keras -> vessel_model.tflite
No quantization, float32 - matches project-wide model format.
"""

from pathlib import Path
import tensorflow as tf

from losses import dice_bce_loss

CHECKPOINT_PATH = Path(r"F:\capstone\retinal_disease_detector\training\checkpoints\vessel_unet_best.keras")
OUT_PATH = Path(r"F:\capstone\retinal_disease_detector\models\vessel_model.tflite")


def main():
    model = tf.keras.models.load_model(CHECKPOINT_PATH, custom_objects={"dice_bce_loss": dice_bce_loss})

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    OUT_PATH.write_bytes(tflite_model)
    size_mb = OUT_PATH.stat().st_size / (1024 * 1024)
    print(f"Saved {OUT_PATH} ({size_mb:.2f} MB)")


if __name__ == "__main__":
    main()