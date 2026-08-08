"""
Compares locate_disc() output against ground-truth disc location (from mask
value=1) for every image. Flags detections that are too far off or too
different in size -- these should be EXCLUDED from training, not silently
kept as bad data.
"""
import sys
from pathlib import Path
import cv2
import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[2]))
from utils.disc_localization import locate_disc

RAW_DIR = Path(__file__).parent / "raw" / "archive" / "G1020"
DISC_VALUE = 1


def true_disc_from_mask(mask: np.ndarray):
    """Get true disc center + radius from ground-truth mask."""
    disc_pixels = (mask == DISC_VALUE).astype(np.uint8)
    contours, _ = cv2.findContours(disc_pixels, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    (cx, cy), radius = cv2.minEnclosingCircle(largest)
    return cx, cy, radius


def main():
    image_files = sorted((RAW_DIR / "Images").glob("*.jpg"))
    results = []

    # NEW: for the shape/position diagnostic -- only filled in for
    # successful detections (dist_ratio is not None)
    diag_det_radius_frac = []
    diag_true_radius_frac = []
    diag_det_cx_frac = []
    diag_true_cx_frac = []

    for i, img_path in enumerate(image_files, start=1):
        if i % 100 == 0:
            print(f"Progress: {i}/{len(image_files)}")

        mask_path = RAW_DIR / "Masks" / f"{img_path.stem}.png"
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        bgr = cv2.imread(str(img_path))
        if mask is None or bgr is None:
            continue

        true = true_disc_from_mask(mask)
        if true is None:
            continue
        true_cx, true_cy, true_r = true

        try:
            detected = locate_disc(bgr)
        except ValueError:
            results.append((img_path.stem, None, true_r, None))
            continue

        dist = np.hypot(detected.center_x - true_cx, detected.center_y - true_cy)
        # normalize distance by true disc radius -> "how many disc-widths off"
        dist_ratio = dist / true_r

        results.append((img_path.stem, dist_ratio, true_r, detected.radius))

        # NEW: record fractions (of image width) for the diagnostic
        img_w = bgr.shape[1]
        diag_det_radius_frac.append(detected.radius / img_w)
        diag_true_radius_frac.append(true_r / img_w)
        diag_det_cx_frac.append(detected.center_x / img_w)
        diag_true_cx_frac.append(true_cx / img_w)

    # summarize
    valid = [r for r in results if r[1] is not None]
    failed = [r for r in results if r[1] is None]

    print(f"\nTotal: {len(results)}  Detection failed: {len(failed)}")
    dist_ratios = np.array([r[1] for r in valid])
    print(f"dist_ratio (0=perfect, 1=off by one disc-width):")
    print(f"  median: {np.median(dist_ratios):.2f}")
    print(f"  mean:   {np.mean(dist_ratios):.2f}")
    for thresh in [0.3, 0.5, 1.0, 2.0]:
        pct_bad = 100 * np.mean(dist_ratios > thresh)
        print(f"  fraction with dist_ratio > {thresh}: {pct_bad:.1f}%")

    # NEW: shape/position diagnostic
    print("\n--- shape/position diagnostic ---")
    print(f"detected radius/width  mean: {np.mean(diag_det_radius_frac):.4f}  std: {np.std(diag_det_radius_frac):.4f}")
    print(f"true radius/width      mean: {np.mean(diag_true_radius_frac):.4f}  std: {np.std(diag_true_radius_frac):.4f}")
    print(f"detected center_x/width std: {np.std(diag_det_cx_frac):.4f}")
    print(f"true center_x/width     std: {np.std(diag_true_cx_frac):.4f}")


if __name__ == "__main__":
    main()