"""Look at actual pixel values in one mask to understand encoding."""
import numpy as np
from PIL import Image
from pathlib import Path

RAW_DIR = Path(__file__).parent / "raw" / "archive" / "G1020"

def main():
    mask_files = sorted((RAW_DIR / "Masks").glob("*.png"))
    if not mask_files:
        mask_files = sorted((RAW_DIR / "Masks").glob("*.jpg"))

    sample = mask_files[0]
    print(f"Inspecting: {sample.name}")

    mask = np.array(Image.open(sample))
    print(f"Shape: {mask.shape}")
    print(f"Dtype: {mask.dtype}")

    unique_vals = np.unique(mask)
    print(f"Unique pixel values: {unique_vals}")

    # count how many pixels have each value
    for v in unique_vals:
        count = np.sum(mask == v)
        print(f"  value {v}: {count} pixels")

if __name__ == "__main__":
    main()