import sys
from pathlib import Path
import cv2
import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[2]))

MASKS_DIR = Path(__file__).parent / "cup_dataset" / "masks"

mask_files = sorted(MASKS_DIR.glob("*.png"))
empty_count = 0
tiny_count = 0  # has SOME cup pixels but suspiciously few

for mask_path in mask_files:
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    cup_pixel_count = int(np.sum(mask > 0))

    if cup_pixel_count == 0:
        empty_count += 1
    elif cup_pixel_count < 50:  # arbitrary "suspiciously small" cutoff
        tiny_count += 1

total = len(mask_files)
print(f"Total masks: {total}")
print(f"Empty (no cup pixels): {empty_count} ({100*empty_count/total:.1f}%)")
print(f"Tiny (<50 cup pixels): {tiny_count} ({100*tiny_count/total:.1f}%)")
print(f"Usable: {total - empty_count - tiny_count} ({100*(total-empty_count-tiny_count)/total:.1f}%)")