"""
utils/disc_localization.py

Classical (non-ML) optic disc localization for fundus images.

WHY THIS EXISTS AS ITS OWN MODULE:
This exact function is called from two places that MUST stay in sync:
  1. dataset_prep/cdr/build_cup_dataset.py  (builds training crops later)
  2. services/inference_service.py           (live inference, added later)
If those two call sites ever drift apart, the cup model ends up trained on
crops that don't match what it sees in production, and CDR accuracy
degrades silently with no error thrown. One shared function makes that bug
class impossible.

ALGORITHM (brightest-region + morphological closing + circle fit):
  1. Use the red channel — the optic disc has the highest contrast against
     surrounding retina on red (standard result in fundus analysis
     literature, e.g. Walter et al. 2002).
  2. Gaussian blur to suppress vessel texture and sensor noise.
  3. Threshold the brightest N% of pixels -> rough disc candidate mask.
  4. Morphological closing to bridge gaps where vessels cross the disc and
     locally darken it (raw thresholding alone leaves a mask full of holes).
  5. Keep only the largest connected component (disc = biggest bright blob;
     camera glare/reflections are usually smaller and scattered elsewhere).
  6. Fit a minimum enclosing circle -> clean (center, radius) instead of a
     ragged blob boundary.

This is a ROUGH localizer, not pixel-perfect segmentation — that's fine,
its only job is producing a crop window. The trained cup model inside that
crop is what needs to be pixel-accurate.
"""

from dataclasses import dataclass
import cv2
import numpy as np


@dataclass
class DiscLocation:
    center_x: int
    center_y: int
    radius: int
    confidence: float  # fraction of bright-pixel mask kept after largest-component filtering


def locate_disc(bgr_image: np.ndarray, brightness_percentile: float = 99.0) -> DiscLocation:
    """
    Locate the optic disc using classical image processing.

    Args:
        bgr_image: full-resolution fundus photo, OpenCV BGR order, uint8.
        brightness_percentile: pixels above this percentile (red channel)
            are disc candidates. 99.0 = brightest 1% of pixels. Tune this
            if your rig's exposure differs a lot from clinical cameras.

    Returns:
        DiscLocation in ORIGINAL image pixel coordinates.

    Raises:
        ValueError: if no plausible disc region is found — caller should
            surface this as a "capture quality too low" error, not guess.
    """
    if bgr_image.ndim != 3 or bgr_image.shape[2] != 3:
        raise ValueError(f"Expected a 3-channel BGR image, got shape {bgr_image.shape}")

    red_channel = bgr_image[:, :, 2]  # OpenCV loads BGR, so index 2 = red

    blurred = cv2.GaussianBlur(red_channel, ksize=(9, 9), sigmaX=0)

    threshold_value = np.percentile(blurred, brightness_percentile) - 1
    _, bright_mask = cv2.threshold(blurred, threshold_value, 255, cv2.THRESH_BINARY)
    bright_mask = bright_mask.astype(np.uint8)

    # Kernel scales with image size so this works on any capture resolution
    kernel_size = max(15, int(min(bgr_image.shape[:2]) * 0.02))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    closed_mask = cv2.morphologyEx(bright_mask, cv2.MORPH_CLOSE, kernel)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(closed_mask, connectivity=8)
    if num_labels <= 1:
        raise ValueError("No bright candidate region found — image may be underexposed or not a fundus photo")

    total_bright_pixels = int(np.sum(closed_mask > 0))

    # Check candidates largest-first, but SKIP any that aren't disc-shaped.
    # Why: closing can fuse the real disc blob with an unrelated bright
    # patch elsewhere into one big, irregular blob. That fused blob has
    # the biggest AREA, but its min-enclosing-circle center can land in
    # the empty gap between the two original blobs — nowhere near the
    # disc. "Largest" alone isn't enough; it also has to look like a disc.
    candidate_labels = np.argsort(stats[1:, cv2.CC_STAT_AREA])[::-1] + 1  # biggest area first

    best_contour = None
    best_label = None
    for label in candidate_labels:
        component_mask = (labels == label).astype(np.uint8) * 255
        contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        contour = max(contours, key=cv2.contourArea)

        area = cv2.contourArea(contour)
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        if hull_area == 0:
            continue

        # solidity: 1.0 = perfectly compact/convex blob (real disc).
        # A fused/branching blob has extra "arms" outside its own convex
        # hull filling ratio, so this drops well below 1.0.
        solidity = area / hull_area
        if solidity < 0.85:
            continue  # not disc-shaped, likely a fused blob — try the next candidate

        best_contour = contour
        best_label = label
        break

    if best_contour is None:
        raise ValueError("No sufficiently disc-shaped bright region found — image may need recapture")

    largest_component_pixels = int(stats[best_label, cv2.CC_STAT_AREA])
    confidence = largest_component_pixels / total_bright_pixels if total_bright_pixels > 0 else 0.0

    (center_x, center_y), radius = cv2.minEnclosingCircle(best_contour)

    return DiscLocation(
        center_x=int(round(center_x)),
        center_y=int(round(center_y)),
        radius=int(round(radius)),
        confidence=round(confidence, 3),
    )


def crop_to_disc(bgr_image: np.ndarray, disc: DiscLocation, padding_factor: float = 1.8) -> np.ndarray:
    """
    Crop a square region centered on the detected disc.

    Args:
        bgr_image: original full-resolution image.
        disc: output of locate_disc() on this same image.
        padding_factor: crop side = disc diameter * padding_factor. 1.8 is
            a starting point from disc-crop literature (e.g. M-Net) — wide
            enough to never clip the cup, tight enough to not waste model
            capacity on irrelevant retina.

    Returns:
        Square BGR crop. Resizing to 224x224 is a separate explicit step
        (kept out of this function so "where to crop" and "how to resample"
        stay independently testable).
    """
    side = int(disc.radius * 2 * padding_factor)
    half = side // 2

    img_h, img_w = bgr_image.shape[:2]
    x1 = max(0, disc.center_x - half)
    y1 = max(0, disc.center_y - half)
    x2 = min(img_w, disc.center_x + half)
    y2 = min(img_h, disc.center_y + half)

    return bgr_image[y1:y2, x1:x2]

