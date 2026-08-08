"""
Splits the 20 vessel image/mask pairs into train/val folders.
16 train, 4 val. Same seed=42 used for cup dataset split, for consistency.
"""

import random
import shutil
from pathlib import Path

SRC_ROOT = Path(r"F:\capstone\retinal_disease_detector\dataset_prep\vessel\vessel_dataset")
SRC_IMAGES = SRC_ROOT / "images"
SRC_MASKS = SRC_ROOT / "masks"

OUT_ROOT = Path(r"F:\capstone\retinal_disease_detector\dataset_prep\vessel\vessel_split")

VAL_COUNT = 4
SEED = 42


def make_dirs():
    for split in ["train", "val"]:
        (OUT_ROOT / split / "images").mkdir(parents=True, exist_ok=True)
        (OUT_ROOT / split / "masks").mkdir(parents=True, exist_ok=True)


def main():
    make_dirs()

    ids = sorted(p.stem for p in SRC_IMAGES.glob("*.png"))
    random.seed(SEED)
    random.shuffle(ids)

    val_ids = set(ids[:VAL_COUNT])
    train_ids = set(ids[VAL_COUNT:])

    for img_id in ids:
        split = "val" if img_id in val_ids else "train"
        shutil.copy(SRC_IMAGES / f"{img_id}.png", OUT_ROOT / split / "images" / f"{img_id}.png")
        shutil.copy(SRC_MASKS / f"{img_id}.png", OUT_ROOT / split / "masks" / f"{img_id}.png")

    print(f"Train: {len(train_ids)} pairs")
    print(f"Val:   {len(val_ids)} pairs")
    print(f"Val IDs: {sorted(val_ids)}")


if __name__ == "__main__":
    main()