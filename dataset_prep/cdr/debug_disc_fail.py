"""Debug why locate_disc() fails on a specific image."""
import sys
from pathlib import Path
import cv2
import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[2]))

RAW_DIR = Path(__file__).parent / "raw" / "archive" / "G1020"

def main():
    name = "image_1"
    bgr = cv2.imread(str(RAW_DIR / "Images" / f"{name}.jpg"))

    if bgr is None:
        print("Image failed to load entirely — path/format issue")
        return

    print(f"Shape: {bgr.shape}, dtype: {bgr.dtype}")

    red_channel = bgr[:, :, 2]
    print(f"Red channel min: {red_channel.min()}, max: {red_channel.max()}, mean: {red_channel.mean():.1f}")

    blurred = cv2.GaussianBlur(red_channel, (9, 9), 0)
    for p in [90, 95, 99, 99.5]:
        val = np.percentile(blurred, p)
        print(f"  {p}th percentile brightness: {val:.1f}")

    cv2.imwrite(str(Path(__file__).parent / "_check_output" / f"{name}_debug.png"), bgr)

if __name__ == "__main__":
    main()