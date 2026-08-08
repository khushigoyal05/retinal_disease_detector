import sys
from pathlib import Path
import cv2
import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[2]))

BASE = Path(__file__).parent / "cup_dataset"
MASKS_DIR = BASE / "masks"
IMAGES_DIR = BASE / "images"

mask_files = sorted(MASKS_DIR.glob("*.png"))
removed = 0

for mask_path in mask_files:
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    cup_pixel_count = int(np.sum(mask > 0))

    if cup_pixel_count == 0:
        img_path = IMAGES_DIR / mask_path.name
        mask_path.unlink()          # delete empty mask
        if img_path.exists():
            img_path.unlink()       # delete matching image
        removed += 1

print(f"Removed {removed} empty pairs.")
print(f"Remaining: {len(list(MASKS_DIR.glob('*.png')))}")