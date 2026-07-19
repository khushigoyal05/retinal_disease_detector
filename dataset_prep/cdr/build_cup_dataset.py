"""
Builds cup-segmentation training crops from G1020.

For each image:
  1. Run locate_disc() (same function used at inference time).
  2. Crop the image AND the mask using that same disc location.
  3. Extract cup-only binary mask from the cropped mask (value 2 -> 255).
  4. Resize both to 224x224 and save.

Images where disc detection fails are skipped and logged — we do NOT want
bad crops silently polluting training data.
"""

import sys
from pathlib import Path
import cv2
import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[2]))  # so we can import utils/
from utils.disc_localization import locate_disc, crop_to_disc

RAW_DIR = Path(__file__).parent / "raw" / "archive" / "G1020"
OUT_DIR = Path(__file__).parent / "cup_dataset"
OUT_IMAGES = OUT_DIR / "images"
OUT_MASKS = OUT_DIR / "masks"

TARGET_SIZE = (224, 224)
CUP_VALUE = 2


def process_one(image_path: Path, mask_path: Path) -> bool:
    """Returns True on success, False if skipped."""
    bgr = cv2.imread(str(image_path))
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

    if bgr is None or mask is None:
        print(f"  SKIP (failed to load): {image_path.name}")
        return False

    try:
        disc = locate_disc(bgr)
    except ValueError as e:
        print(f"  SKIP (disc not found): {image_path.name} — {e}")
        return False

    img_crop = crop_to_disc(bgr, disc)
    mask_crop = crop_to_disc(mask, disc)  # same disc -> same crop window

    if img_crop.size == 0 or mask_crop.size == 0:
        print(f"  SKIP (empty crop): {image_path.name}")
        return False

    # keep only cup pixels, make binary 0/255
    cup_binary = np.where(mask_crop == CUP_VALUE, 255, 0).astype(np.uint8)

    img_resized = cv2.resize(img_crop, TARGET_SIZE, interpolation=cv2.INTER_LINEAR)
    mask_resized = cv2.resize(cup_binary, TARGET_SIZE, interpolation=cv2.INTER_NEAREST)
    # INTER_NEAREST for mask: keeps values exactly 0 or 255, no blurry in-between

    cv2.imwrite(str(OUT_IMAGES / f"{image_path.stem}.png"), img_resized)
    cv2.imwrite(str(OUT_MASKS / f"{image_path.stem}.png"), mask_resized)
    return True


def main():
    OUT_IMAGES.mkdir(parents=True, exist_ok=True)
    OUT_MASKS.mkdir(parents=True, exist_ok=True)

    image_files = sorted((RAW_DIR / "Images").glob("*.jpg")) + \
                  sorted((RAW_DIR / "Images").glob("*.png"))
    
    ok, skipped = 0, 0
    total = len(image_files)
    for i, img_path in enumerate(image_files, start=1):
        if i % 25 == 0:
            print(f"Progress: {i}/{total}")
        mask_path = RAW_DIR / "Masks" / f"{img_path.stem}.png"
        if not mask_path.exists():
            print(f"  SKIP (no mask): {img_path.name}")
            skipped += 1
            continue

        if process_one(img_path, mask_path):
            ok += 1
        else:
            skipped += 1

    print(f"\nDone. Built: {ok}  Skipped: {skipped}")


if __name__ == "__main__":
    main()