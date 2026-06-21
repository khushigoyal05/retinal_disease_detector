"""
Image preprocessing shared by every InferenceEngine implementation.

CRITICAL: this transform must exactly match whatever preprocessing you use
in your training notebook on the laptop. A mismatch here is the single most
common reason a model scores 95% accuracy in training and produces garbage
on deployed hardware. When you finalize your training pipeline, come back
to this function and make sure resize method, channel order, and the
normalization formula all match exactly.
"""

from pathlib import Path
import numpy as np
from PIL import Image


def load_and_preprocess(image_path: Path, target_size: tuple[int, int]) -> np.ndarray:
    """
    Load an image from disk and convert it to the array shape/range a
    Keras MobileNet/EfficientNet-style model expects.

    Args:
        image_path: path to a JPEG/PNG retinal image.
        target_size: (width, height) the model was trained on.

    Returns:
        float32 array of shape (1, height, width, 3), values in [-1, 1].
    """
    with Image.open(image_path) as img:
        img = img.convert("RGB")               # normalize away grayscale/RGBA inputs
        img = img.resize(target_size, Image.BILINEAR)
        array = np.asarray(img, dtype=np.float32)

    array = (array / 127.5) - 1.0               # placeholder — match your training script exactly
    array = np.expand_dims(array, axis=0)        # (H, W, 3) -> (1, H, W, 3): models expect a batch dim
    return array

import cv2


def compute_sharpness_score(image: np.ndarray) -> float:
    """Estimates image sharpness using variance of the Laplacian.

    Higher scores indicate a sharper image; values near zero indicate
    heavy blur. There is no universal "good" threshold -- it depends on
    sensor, lens, and lighting, so this must be calibrated empirically
    once real hardware exists (capture known-sharp and known-blurry
    reference images on the actual rig and compare scores).
    """
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image

    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    return float(laplacian.var())