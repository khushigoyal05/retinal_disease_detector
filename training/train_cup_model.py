"""
training/train_cup_model.py

Trains the cup segmentation U-Net on the cup_split dataset.
First pass: get it training end-to-end, not accuracy-tuned yet.
"""

import sys
from pathlib import Path
import numpy as np
import cv2
import tensorflow as tf

sys.path.append(str(Path(__file__).resolve().parents[1]))
from training.unet_model import build_cup_unet
from training.losses import dice_bce_loss

DATA_DIR = Path(__file__).resolve().parents[1] / "dataset_prep" / "cdr" / "cup_split"
CHECKPOINT_DIR = Path(__file__).parent / "checkpoints"
IMG_SIZE = (224, 224)
BATCH_SIZE = 8
EPOCHS = 20


def load_pair(img_path, mask_path):
    img = cv2.imread(str(img_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = (img.astype(np.float32) / 127.5) - 1.0  # matches locked [-1,1] normalization

    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    mask = (mask.astype(np.float32) / 255.0)  # 0 or 1
    mask = np.expand_dims(mask, axis=-1)  # add channel dim -> (224,224,1)

    return img, mask


def make_dataset(split_name):
    img_dir = DATA_DIR / split_name / "images"
    mask_dir = DATA_DIR / split_name / "masks"
    filenames = sorted(p.name for p in img_dir.glob("*.png"))

    images, masks = [], []
    for fname in filenames:
        img, mask = load_pair(img_dir / fname, mask_dir / fname)
        images.append(img)
        masks.append(mask)

    images = np.stack(images)
    masks = np.stack(masks)
    return images, masks


def main():
    print("Loading data...")
    train_x, train_y = make_dataset("train")
    val_x, val_y = make_dataset("val")
    print(f"Train: {train_x.shape}  Val: {val_x.shape}")

    model = build_cup_unet()
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss=dice_bce_loss)

    CHECKPOINT_DIR.mkdir(exist_ok=True)
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        str(CHECKPOINT_DIR / "cup_unet_best.keras"),
        save_best_only=True,
        monitor="val_loss",
    )

    model.fit(
        train_x, train_y,
        validation_data=(val_x, val_y),
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=[checkpoint],
    )


if __name__ == "__main__":
    main()