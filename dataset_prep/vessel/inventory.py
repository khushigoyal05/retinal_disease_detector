"""
Checks the DRIVE training set:
- image <-> mask pairing (correct filenames match up)
- image sizes match mask sizes
- mask pixel values are basically binary (vessel vs background)
"""

from pathlib import Path
import numpy as np
from PIL import Image

# EDIT this if your path is different
DRIVE_ROOT = Path(r"F:\capstone\retinal_disease_detector\dataset_prep\vessel\raw\DRIVE")

IMAGES_DIR = DRIVE_ROOT / "training" / "images"
MASKS_DIR = DRIVE_ROOT / "training" / "1st_manual"


def get_id(filename: str) -> str:
    # "21_training.tif" -> "21"   |   "21_manual1.gif" -> "21"
    return filename.split("_")[0]


def main():
    images = sorted(IMAGES_DIR.glob("*.tif"))
    masks = sorted(MASKS_DIR.glob("*.gif"))

    print(f"Found {len(images)} images, {len(masks)} masks")

    image_ids = {get_id(p.name) for p in images}
    mask_ids = {get_id(p.name) for p in masks}

    missing_masks = image_ids - mask_ids
    missing_images = mask_ids - image_ids
    if missing_masks:
        print(f"WARNING: images with no mask: {missing_masks}")
    if missing_images:
        print(f"WARNING: masks with no image: {missing_images}")
    if not missing_masks and not missing_images:
        print("All images and masks pair up correctly.")

    # check a few pairs for size match + mask binariness
    for img_path, mask_path in zip(images, masks):
        img = Image.open(img_path)
        mask = Image.open(mask_path).convert("L")  # force grayscale

        if img.size != mask.size:
            print(f"SIZE MISMATCH: {img_path.name} {img.size} vs {mask_path.name} {mask.size}")

        mask_arr = np.array(mask)
        unique_vals = np.unique(mask_arr)
        if len(unique_vals) > 2:
            print(f"{mask_path.name}: not binary, unique values sample = {unique_vals[:10]}")

    # print stats for just the first pair, as a sanity check
    img0 = Image.open(images[0])
    mask0 = np.array(Image.open(masks[0]).convert("L"))
    print("\nSample pair:", images[0].name, "<->", masks[0].name)
    print("Image size:", img0.size, "mode:", img0.mode)
    print("Mask unique values:", np.unique(mask0))
    print("Vessel pixel % (mask==255):", round((mask0 == 255).mean() * 100, 2), "%")


if __name__ == "__main__":
    main()