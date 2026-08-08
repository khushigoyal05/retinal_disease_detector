import shutil
import random
from pathlib import Path

random.seed(42)  # reproducible split

BASE = Path(__file__).parent / "cup_dataset"
IMAGES_DIR = BASE / "images"
MASKS_DIR = BASE / "masks"

OUT_BASE = Path(__file__).parent / "cup_split"
VAL_FRACTION = 0.2

def main():
    image_files = sorted(IMAGES_DIR.glob("*.png"))
    random.shuffle(image_files)

    val_count = int(len(image_files) * VAL_FRACTION)
    val_files = image_files[:val_count]
    train_files = image_files[val_count:]

    for split_name, files in [("train", train_files), ("val", val_files)]:
        (OUT_BASE / split_name / "images").mkdir(parents=True, exist_ok=True)
        (OUT_BASE / split_name / "masks").mkdir(parents=True, exist_ok=True)
        for img_path in files:
            mask_path = MASKS_DIR / img_path.name
            shutil.copy(img_path, OUT_BASE / split_name / "images" / img_path.name)
            shutil.copy(mask_path, OUT_BASE / split_name / "masks" / img_path.name)

    print(f"Train: {len(train_files)}  Val: {len(val_files)}")

if __name__ == "__main__":
    main()