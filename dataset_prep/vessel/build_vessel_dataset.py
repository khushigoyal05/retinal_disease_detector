"""
Resizes DRIVE training images + masks to 224x224 and saves as .png pairs.
Run once. Output goes to dataset_prep/vessel/vessel_dataset/
"""

from pathlib import Path
from PIL import Image

DRIVE_ROOT = Path(r"F:\capstone\retinal_disease_detector\dataset_prep\vessel\raw\DRIVE")
IMAGES_DIR = DRIVE_ROOT / "training" / "images"
MASKS_DIR = DRIVE_ROOT / "training" / "1st_manual"

OUT_ROOT = Path(r"F:\capstone\retinal_disease_detector\dataset_prep\vessel\vessel_dataset")
OUT_IMAGES = OUT_ROOT / "images"
OUT_MASKS = OUT_ROOT / "masks"

SIZE = (224, 224)  # matches MobileNetV2 input, same as cup model


def get_id(filename: str) -> str:
    return filename.split("_")[0]


def main():
    OUT_IMAGES.mkdir(parents=True, exist_ok=True)
    OUT_MASKS.mkdir(parents=True, exist_ok=True)

    images = sorted(IMAGES_DIR.glob("*.tif"))
    masks_by_id = {get_id(p.name): p for p in MASKS_DIR.glob("*.gif")}

    built = 0
    for img_path in images:
        img_id = get_id(img_path.name)
        mask_path = masks_by_id[img_id]

        # BILINEAR for image = smooth resize, normal for photos
        img = Image.open(img_path).convert("RGB").resize(SIZE, Image.BILINEAR)

        # NEAREST for mask = keeps pixels strictly 0 or 255, no blurry mix
        mask = Image.open(mask_path).convert("L").resize(SIZE, Image.NEAREST)

        img.save(OUT_IMAGES / f"{img_id}.png")
        mask.save(OUT_MASKS / f"{img_id}.png")
        built += 1

    print(f"Built {built} image/mask pairs into {OUT_ROOT}")


if __name__ == "__main__":
    main()