"""
services/cdr_service.py

Computes CDR (cup-to-disc ratio) for a fundus image using the hybrid
pipeline: classical disc localization + trained cup segmentation model.

CDR = sqrt(cup_area / disc_area)  — diameter-ratio approximation.
(area scales with diameter^2, so sqrt(area ratio) ≈ diameter ratio —
this avoids needing to fit a second circle to the irregular cup shape.)
"""

from dataclasses import dataclass
import numpy as np
import cv2

from utils.disc_localization import locate_disc, crop_to_disc, DiscLocation
from services.cup_segmentation_engine import CupSegmentationEngine


@dataclass
class CDRResult:
    cdr: float
    disc: DiscLocation           # original-image coordinates, for drawing
    cup_mask_in_crop: np.ndarray # 224x224 binary mask, crop-local coordinates
    crop_bbox: tuple             # (x1, y1, x2, y2) in original-image coordinates


def compute_cdr(bgr_image: np.ndarray, cup_engine: CupSegmentationEngine) -> CDRResult:
    """
    Raises:
        ValueError: if disc detection fails (image quality too low).
    """
    disc = locate_disc(bgr_image)  # raises ValueError if no disc found

    # Need the crop's bounding box in original coordinates too, for drawing
    # later — crop_to_disc() doesn't return this, so recompute the same way.
    side = int(disc.radius * 2 * 1.8)  # matches crop_to_disc's padding_factor default
    half = side // 2
    img_h, img_w = bgr_image.shape[:2]
    x1 = max(0, disc.center_x - half)
    y1 = max(0, disc.center_y - half)
    x2 = min(img_w, disc.center_x + half)
    y2 = min(img_h, disc.center_y + half)

    crop = crop_to_disc(bgr_image, disc)
    crop_resized = cv2.resize(crop, (224, 224), interpolation=cv2.INTER_LINEAR)
    crop_rgb = cv2.cvtColor(crop_resized, cv2.COLOR_BGR2RGB)
    crop_norm = (crop_rgb.astype(np.float32) / 127.5) - 1.0
    crop_input = np.expand_dims(crop_norm, axis=0)  # [1,224,224,3]

    cup_mask = cup_engine.predict(crop_input)  # [224,224] binary

    # Disc area within the SAME 224x224 crop space, for a fair ratio —
    # the disc circle, scaled from original-image radius down to crop-local
    # radius (crop was resized from its true side length to 224).
    scale = 224 / (x2 - x1) if (x2 - x1) > 0 else 1.0
    disc_radius_in_crop = disc.radius * scale
    disc_area = np.pi * (disc_radius_in_crop ** 2)

    cup_area = float(np.sum(cup_mask > 0))
    cdr = min(float(np.sqrt(cup_area / disc_area)), 0.99) if disc_area > 0 else 0.0

    return CDRResult(
        cdr=round(cdr, 2),
        disc=disc,
        cup_mask_in_crop=cup_mask,
        crop_bbox=(x1, y1, x2, y2),
    )