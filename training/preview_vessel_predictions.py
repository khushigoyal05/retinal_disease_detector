"""
Runs the trained vessel model on val images and saves prediction images
next to the originals, so we can visually check if it's learning vessels.
"""

from pathlib import Path
import numpy as np
import tensorflow as tf
from PIL import Image

from losses import dice_bce_loss

SPLIT_ROOT = Path(r"F:\capstone\retinal_disease_detector\dataset_prep\vessel\vessel_split")
CHECKPOINT_PATH = Path(r"F:\capstone\retinal_disease_detector\training\checkpoints\vessel_unet_best.keras")
OUT_DIR = Path(r"F:\capstone\retinal_disease_detector\dataset_prep\vessel\_check_output")


def main():
    OUT_DIR.mkdir(exist_ok=True)

    model = tf.keras.models.load_model(CHECKPOINT_PATH, custom_objects={"dice_bce_loss": dice_bce_loss})

    val_images_dir = SPLIT_ROOT / "val" / "images"

    for img_path in sorted(val_images_dir.glob("*.png")):
        img = Image.open(img_path)
        img_arr = np.array(img) / 127.5 - 1.0
        pred = model.predict(img_arr[None, ...], verbose=0)[0, ..., 0]  # (224,224)

        pred_mask = (pred > 0.5).astype("uint8") * 255  # threshold to binary
        pred_img = Image.fromarray(pred_mask)

        img.save(OUT_DIR / f"{img_path.stem}_original.png")
        pred_img.save(OUT_DIR / f"{img_path.stem}_predicted.png")

    print(f"Saved predictions to {OUT_DIR}")


if __name__ == "__main__":
    main()