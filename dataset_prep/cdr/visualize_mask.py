"""Overlay mask colors on the image to visually confirm disc vs cup."""
import numpy as np
from PIL import Image
from pathlib import Path

RAW_DIR = Path(__file__).parent / "raw" / "archive" / "G1020"
OUT_DIR = Path(__file__).parent / "_check_output"
OUT_DIR.mkdir(exist_ok=True)

def main():
    name = "image_0"
    img = np.array(Image.open(RAW_DIR / "Images" / f"{name}.jpg").convert("RGB"))
    mask = np.array(Image.open(RAW_DIR / "Masks" / f"{name}.png"))

    overlay = img.copy()
    overlay[mask == 1] = [255, 0, 0]   # value 1 -> RED
    overlay[mask == 2] = [0, 255, 0]   # value 2 -> GREEN

    # blend with original so we can still see the eye underneath
    blended = (0.5 * img + 0.5 * overlay).astype(np.uint8)

    out_path = OUT_DIR / f"{name}_maskcheck.png"
    Image.fromarray(blended).save(out_path)
    print(f"Saved: {out_path}")

if __name__ == "__main__":
    main()