"""Check disc detection confidence + location for suspect images."""
import sys
from pathlib import Path
import cv2

sys.path.append(str(Path(__file__).resolve().parents[2]))
from utils.disc_localization import locate_disc

RAW_DIR = Path(__file__).parent / "raw" / "archive" / "G1020"

# EDIT THIS LIST with the 2 failing image names you found
SUSPECT_NAMES = ["image_10", "image_50"]

def main():
    for name in SUSPECT_NAMES:
        img_path = RAW_DIR / "Images" / f"{name}.jpg"
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            print(f"{name}: failed to load")
            continue

        disc = locate_disc(bgr)
        h, w = bgr.shape[:2]
        print(f"{name}: image size {w}x{h}")
        print(f"  center=({disc.center_x}, {disc.center_y}) radius={disc.radius} confidence={disc.confidence}")

if __name__ == "__main__":
    main()