"""Draw TRUE disc (blue) vs DETECTED disc (green) on worst-mismatch images."""
import sys
from pathlib import Path
import cv2
import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[2]))
from utils.disc_localization import locate_disc

RAW_DIR = Path(__file__).parent / "raw" / "archive" / "G1020"
OUT_DIR = Path(__file__).parent / "_check_output"
OUT_DIR.mkdir(exist_ok=True)
DISC_VALUE = 1


def true_disc_from_mask(mask):
    disc_pixels = (mask == DISC_VALUE).astype(np.uint8)
    contours, _ = cv2.findContours(disc_pixels, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    (cx, cy), radius = cv2.minEnclosingCircle(largest)
    return int(cx), int(cy), int(radius)


def main():
    # pick 5 fixed images to inspect closely
    names = ["image_0", "image_5", "image_100", "image_300", "image_600"]

    for name in names:
        img_path = RAW_DIR / "Images" / f"{name}.jpg"
        mask_path = RAW_DIR / "Masks" / f"{name}.png"
        bgr = cv2.imread(str(img_path))
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if bgr is None or mask is None:
            continue

        true = true_disc_from_mask(mask)
        if true is None:
            continue
        tx, ty, tr = true

        try:
            detected = locate_disc(bgr)
            dx, dy, dr = detected.center_x, detected.center_y, detected.radius
        except ValueError:
            dx = dy = dr = None

        annotated = bgr.copy()
        cv2.circle(annotated, (tx, ty), tr, (255, 0, 0), 10)      # TRUE = blue
        if dx is not None:
            cv2.circle(annotated, (dx, dy), dr, (0, 255, 0), 10)  # DETECTED = green

        h, w = annotated.shape[:2]
        small = cv2.resize(annotated, (900, 900 * h // w))
        out_path = OUT_DIR / f"{name}_truevsdetected.png"
        cv2.imwrite(str(out_path), small)
        print(f"saved {out_path}  true=({tx},{ty},r={tr})  detected=({dx},{dy},r={dr})")


if __name__ == "__main__":
    main()