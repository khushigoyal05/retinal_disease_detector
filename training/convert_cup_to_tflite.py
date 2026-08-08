import sys
from pathlib import Path
import tensorflow as tf

sys.path.append(str(Path(__file__).resolve().parents[1]))
from training.losses import dice_bce_loss

CHECKPOINT = Path(__file__).parent / "checkpoints" / "cup_unet_best.keras"
OUTPUT = Path(__file__).resolve().parents[1] / "models" / "cdr_model.tflite"

model = tf.keras.models.load_model(CHECKPOINT, custom_objects={"dice_bce_loss": dice_bce_loss})

converter = tf.lite.TFLiteConverter.from_keras_model(model)
# No quantization — matches locked contract (float32 in/out, no quantization)
tflite_model = converter.convert()

OUTPUT.parent.mkdir(exist_ok=True)
OUTPUT.write_bytes(tflite_model)
print(f"Saved: {OUTPUT}  ({len(tflite_model) / 1e6:.2f} MB)")