import sys
from pathlib import Path
import numpy as np
import cv2
import tensorflow as tf

sys.path.append(str(Path(__file__).resolve().parents[1]))
from training.losses import dice_bce_loss

DATA_DIR = Path(__file__).resolve().parents[1] / "dataset_prep" / "cdr" / "cup_split" / "val"
CHECKPOINT = Path(__file__).parent / "checkpoints" / "cup_unet_best.keras"
OUT_DIR = Path(__file__).parent / "cup_preview_output"

model = tf.keras.models.load_model(CHECKPOINT, custom_objects={"dice_bce_loss": dice_bce_loss})

OUT_DIR.mkdir(exist_ok=True)
img_files = sorted((DATA_DIR / "images").glob("*.png"))[:8]

for img_path in img_files:
    img = cv2.imread(str(img_path))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_norm = (img_rgb.astype(np.float32) / 127.5) - 1.0
    pred = model.predict(np.expand_dims(img_norm, axis=0), verbose=0)[0, :, :, 0]

    pred_mask = (pred > 0.5).astype(np.uint8) * 255
    overlay = img.copy()
    overlay[pred_mask > 0] = [0, 255, 0]  # green where predicted cup
    blended = cv2.addWeighted(img, 0.6, overlay, 0.4, 0)

    cv2.imwrite(str(OUT_DIR / f"{img_path.stem}_pred.png"), blended)
    print(f"saved {img_path.stem}_pred.png")