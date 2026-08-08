"""
Loads vessel_model.tflite and runs vessel segmentation on a retinal image.
Same pattern as CupSegmentationEngine.
"""

from pathlib import Path
import numpy as np
from PIL import Image

try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    import tensorflow.lite as tflite  # fallback for Windows dev


class VesselSegmentationEngine:
    def __init__(self, model_path: Path):
        self.interpreter = tflite.Interpreter(model_path=str(model_path))
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def predict_mask(self, image: Image.Image) -> np.ndarray:
        """
        Takes a PIL image, returns a (224,224) binary numpy array.
        255 = vessel pixel, 0 = background.
        """
        img_resized = image.convert("RGB").resize((224, 224))
        img_arr = np.array(img_resized, dtype="float32") / 127.5 - 1.0
        img_arr = img_arr[None, ...]  # add batch dim

        self.interpreter.set_tensor(self.input_details[0]["index"], img_arr)
        self.interpreter.invoke()
        pred = self.interpreter.get_tensor(self.output_details[0]["index"])[0, ..., 0]

        return (pred > 0.5).astype("uint8") * 255