# save as training/evaluate_model.py and run it
from pathlib import Path
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

SPLIT_DIR = Path(r"F:\capstone\retinal_disease_detector\dataset_prep\split")
MODEL_PATH = Path(r"F:\capstone\retinal_disease_detector\training\checkpoints\phase2v2_best.keras")
CLASS_ORDER = ["Normal", "Diabetic Retinopathy", "Glaucoma", "Cataract", "AMD"]

model = tf.keras.models.load_model(MODEL_PATH)
val_ds = tf.keras.utils.image_dataset_from_directory(
    SPLIT_DIR / "val", class_names=CLASS_ORDER, image_size=(224, 224),
    batch_size=16, label_mode="categorical", shuffle=False,
)

y_true, y_pred = [], []
for images, labels in val_ds:
    preds = model.predict(images, verbose=0)
    y_true.extend(np.argmax(labels.numpy(), axis=1))
    y_pred.extend(np.argmax(preds, axis=1))

print(classification_report(y_true, y_pred, target_names=CLASS_ORDER))
print("Confusion matrix (rows=true, cols=predicted):")
print(confusion_matrix(y_true, y_pred))