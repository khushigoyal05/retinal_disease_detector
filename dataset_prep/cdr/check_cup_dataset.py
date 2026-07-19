"""Overlay cup mask on cropped image for a few samples — visual sanity check."""
import numpy as np
import cv2
from pathlib import Path

CUP_DIR = Path(__file__).parent / "cup_dataset"
OUT_DIR = Path(__file__).parent / "_check_output"
OUT_DIR.mkdir(exist_ok=True)

import random
random.seed(42)
all_built = sorted((CUP_DIR / "images").glob("*.png"))
SAMPLES = [f.stem for f in random.sample(all_built, 30)]

def main():
    for name in SAMPLES:
        img_path = CUP_DIR / "images" / f"{name}.png"
        mask_path = CUP_DIR / "masks" / f"{name}.png"

        if not img_path.exists():
            print(f"skip {name} (not built)")
            continue

        img = cv2.imread(str(img_path))
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        overlay = img.copy()
        overlay[mask == 255] = [0, 255, 0]
        blended = cv2.addWeighted(img, 0.5, overlay, 0.5, 0)

        out_path = OUT_DIR / f"{name}_cupcheck.png"
        cv2.imwrite(str(out_path), blended)
        print(f"saved {out_path}")

if __name__ == "__main__":
    main()