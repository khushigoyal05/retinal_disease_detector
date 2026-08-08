"""
Trains vessel segmentation U-Net on DRIVE dataset.
Reuses build_cup_unet() architecture and dice_bce_loss() - both are generic
binary segmentation tools, not actually cup-specific despite the name.
"""

from pathlib import Path
import numpy as np
import tensorflow as tf
from PIL import Image

from unet_model import build_cup_unet
from losses import dice_bce_loss

SPLIT_ROOT = Path(r"F:\capstone\retinal_disease_detector\dataset_prep\vessel\vessel_split")
CHECKPOINT_DIR = Path(r"F:\capstone\retinal_disease_detector\training\checkpoints")
CHECKPOINT_DIR.mkdir(exist_ok=True)

IMG_SIZE = (224, 224)
BATCH_SIZE = 4  # small dataset, small batch
EPOCHS = 20     # same starting point as cup model


def load_pairs(split: str):
    img_dir = SPLIT_ROOT / split / "images"
    mask_dir = SPLIT_ROOT / split / "masks"

    images, masks = [], []
    for img_path in sorted(img_dir.glob("*.png")):
        img = np.array(Image.open(img_path)) / 127.5 - 1.0  # same normalization as all models
        mask = np.array(Image.open(mask_dir / img_path.name)) / 255.0  # 0 or 1
        images.append(img.astype("float32"))
        masks.append(mask.astype("float32")[..., None])  # add channel dim

    return np.array(images), np.array(masks)


def main():
    print("Loading data...")
    train_x, train_y = load_pairs("train")
    val_x, val_y = load_pairs("val")
    print(f"Train: {train_x.shape}, Val: {val_x.shape}")

    model = build_cup_unet()
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss=dice_bce_loss)

    checkpoint_path = CHECKPOINT_DIR / "vessel_unet_best.keras"
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            str(checkpoint_path), save_best_only=True, monitor="val_loss"
        )
    ]

    model.fit(
        train_x, train_y,
        validation_data=(val_x, val_y),
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=callbacks,
    )

    print(f"Best model saved to {checkpoint_path}")


if __name__ == "__main__":
    main()