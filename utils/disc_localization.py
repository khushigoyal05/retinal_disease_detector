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



# Real optic discs occupy a tight, predictable size range relative to
# image width (measured empirically on G1020 ground-truth masks:
# mean radius/width = 0.0772, std = 0.0076 — very tight clustering).
EXPECTED_DISC_RADIUS_FRACTION = 0.0772
MIN_DISC_RADIUS_FRACTION = 0.03
MAX_DISC_RADIUS_FRACTION = 0.15


def _find_disc_candidate(bgr_image: np.ndarray, brightness_percentile: float):
    """
    Try to find a disc-shaped, disc-sized blob at ONE brightness threshold.
    Returns (contour, confidence) or (None, None) if nothing plausible was
    found at this threshold — caller tries a different threshold next.
    """
    red_channel = bgr_image[:, :, 2]  # OpenCV loads BGR, so index 2 = red
    blurred = cv2.GaussianBlur(red_channel, ksize=(9, 9), sigmaX=0)

    threshold_value = np.percentile(blurred, brightness_percentile) - 1
    _, bright_mask = cv2.threshold(blurred, threshold_value, 255, cv2.THRESH_BINARY)
    bright_mask = bright_mask.astype(np.uint8)

    kernel_size = max(15, int(min(bgr_image.shape[:2]) * 0.02))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    closed_mask = cv2.morphologyEx(bright_mask, cv2.MORPH_CLOSE, kernel)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(closed_mask, connectivity=8)
    if num_labels <= 1:
        return None, None

    total_bright_pixels = int(np.sum(closed_mask > 0))
    img_width = bgr_image.shape[1]
    min_radius_px = MIN_DISC_RADIUS_FRACTION * img_width
    max_radius_px = MAX_DISC_RADIUS_FRACTION * img_width
    expected_radius_px = EXPECTED_DISC_RADIUS_FRACTION * img_width

    # Collect EVERY candidate that passes the shape/size checks, then pick
    # the one closest to the expected disc size — not just the first one
    # found. Why: solidity alone doesn't catch elongated-but-smooth blobs
    # (e.g. a glare streak) since a smooth ellipse is already convex, so
    # solidity stays near 1.0 even for non-disc shapes. Circularity below
    # catches those. But even after both checks, more than one candidate
    # can pass — picking the one whose SIZE best matches the tight known
    # disc-size distribution is a much stronger tiebreaker than "biggest".
    best_contour = None
    best_confidence = None
    best_size_error = None

    for label in range(1, num_labels):  # label 0 = background
        component_mask = (labels == label).astype(np.uint8) * 255
        contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        contour = max(contours, key=cv2.contourArea)

        area = cv2.contourArea(contour)
        if area <= 0:
            continue

        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        if hull_area == 0:
            continue

        # solidity: catches branching/concave blobs (fused-together regions)
        solidity = area / hull_area
        if solidity < 0.85:
            continue

        # circularity: catches smooth-but-elongated blobs solidity misses
        # (perfect circle = 1.0; a thin streak is much lower even if convex)
        perimeter = cv2.arcLength(contour, closed=True)
        if perimeter == 0:
            continue

        _, candidate_radius = cv2.minEnclosingCircle(contour)

        circularity = (4 * np.pi * area) / (perimeter ** 2)
        if circularity < 0.6:
            # print(f"  REJECTED (circularity): {circularity:.3f}  radius_frac={candidate_radius/img_width:.4f}")
            continue

        if not (min_radius_px <= candidate_radius <= max_radius_px):
            continue

        size_error = abs(candidate_radius - expected_radius_px)
        if best_size_error is None or size_error < best_size_error:
            best_contour = contour
            best_size_error = size_error
            best_confidence = int(stats[label, cv2.CC_STAT_AREA]) / total_bright_pixels if total_bright_pixels > 0 else 0.0

    return best_contour, best_confidence


# Try strictest threshold first (fewest false positives), loosen if that
# finds nothing. Covers images where the disc isn't quite bright enough
# to make the top 1% cut due to normal exposure variation.
PERCENTILE_SEARCH_ORDER = [99.5, 99.0, 98.0, 97.0, 95.0, 92.0, 90.0]


def locate_disc(bgr_image: np.ndarray) -> DiscLocation:
    """
    Locate the optic disc using classical image processing.

    Tries several brightness thresholds (strictest first) since fundus
    photos vary in overall exposure — a fixed threshold makes the disc
    too dim to detect on some images and too easily confused with glare
    on others.

    Args:
        bgr_image: full-resolution fundus photo, OpenCV BGR order, uint8.

    Returns:
        DiscLocation in ORIGINAL image pixel coordinates.

    Raises:
        ValueError: if no plausible disc region is found at ANY threshold
            — caller should surface this as a "capture quality too low"
            error, not guess.
    """
    if bgr_image.ndim != 3 or bgr_image.shape[2] != 3:
        raise ValueError(f"Expected a 3-channel BGR image, got shape {bgr_image.shape}")

    for percentile in PERCENTILE_SEARCH_ORDER:
        contour, confidence = _find_disc_candidate(bgr_image, percentile)
        if contour is not None:
            (center_x, center_y), radius = cv2.minEnclosingCircle(contour)
            return DiscLocation(
                center_x=int(round(center_x)),
                center_y=int(round(center_y)),
                radius=int(round(radius)),
                confidence=round(confidence, 3),
            )

    raise ValueError("No candidate region matched the expected disc shape/size at any threshold — image may need recapture")


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