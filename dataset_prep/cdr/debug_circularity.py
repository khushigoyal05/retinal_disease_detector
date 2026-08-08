import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import cv2
from utils.disc_localization import locate_disc

RAW_DIR = "dataset_prep/cdr/raw/archive/G1020/Images"
all_files = sorted(os.listdir(RAW_DIR))
image_files = [f for f in all_files if f.lower().endswith((".jpg", ".jpeg", ".png"))][:40]

fail_count = 0
for fname in image_files:
    path = os.path.join(RAW_DIR, fname)
    img = cv2.imread(path)
    if img is None:
        continue
    print(f"--- {fname} ---")
    try:
        result = locate_disc(img)
        print("  SUCCESS")
    except ValueError:
        print("  FAILED (no candidate passed)")
        fail_count += 1

print(f"\nFailed: {fail_count}/{len(image_files)}")