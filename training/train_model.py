"""
training/train_model.py

Phase 1: trains a MobileNetV2-based classifier with the base FROZEN
(feature extraction only, no fine-tuning yet). Deliberately lightweight
for CPU training. Fine-tuning the base layers is a separate, later step
once this baseline is evaluated.
"""

from pathlib import Path
import tensorflow as tf

SPLIT_DIR = Path(r"F:\capstone\retinal_disease_detector\dataset_prep\split")
MODEL_OUT = Path(r"F:\capstone\retinal_disease_detector\training\checkpoints")
MODEL_OUT.mkdir(parents=True, exist_ok=True)

IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 15

# CRITICAL: must exactly match config.DISEASE_CLASSES. The model's output
# index N gets read later as DISEASE_CLASSES[N] -- a mismatch here causes
# silent mislabeling with no error, since image_dataset_from_directory
# would otherwise default to alphabetical folder order instead.
CLASS_ORDER = ["Normal", "Diabetic Retinopathy", "Glaucoma", "Cataract", "AMD"]


def build_datasets():
    train_ds = tf.keras.utils.image_dataset_from_directory(
        SPLIT_DIR / "train", class_names=CLASS_ORDER, image_size=IMG_SIZE,
        batch_size=BATCH_SIZE, label_mode="categorical", shuffle=True, seed=42,
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        SPLIT_DIR / "val", class_names=CLASS_ORDER, image_size=IMG_SIZE,
        batch_size=BATCH_SIZE, label_mode="categorical", shuffle=False,
    )
    # overlaps disk reads with training steps -- free speedup, no GPU needed
    return train_ds.prefetch(tf.data.AUTOTUNE), val_ds.prefetch(tf.data.AUTOTUNE)


def compute_class_weights(train_dir: Path) -> dict:
    """Inverse-frequency weights -- the loss penalizes mistakes on thin
    classes (Glaucoma, AMD) harder than mistakes on Normal."""
    counts = {cls: len(list((train_dir / cls).glob("*.jpg"))) for cls in CLASS_ORDER}
    total = sum(counts.values())
    weights = {i: total / (len(CLASS_ORDER) * counts[cls]) for i, cls in enumerate(CLASS_ORDER)}
    print("Class weights:", {CLASS_ORDER[i]: round(w, 2) for i, w in weights.items()})
    return weights


def build_model() -> tf.keras.Model:
    base = tf.keras.applications.MobileNetV2(input_shape=IMG_SIZE + (3,), include_top=False, weights="imagenet")
    base.trainable = False  # Phase 1: frozen base, train head only

    inputs = tf.keras.Input(shape=IMG_SIZE + (3,))
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
    x = tf.keras.layers.RandomFlip("horizontal")(x)   # augmentation, active only during training
    x = tf.keras.layers.RandomRotation(0.05)(x)
    x = tf.keras.layers.RandomZoom(0.1)(x)
    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(len(CLASS_ORDER), activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def main():
    train_ds, val_ds = build_datasets()
    class_weights = compute_class_weights(SPLIT_DIR / "train")
    model = build_model()
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=4, restore_best_weights=True, monitor="val_accuracy"),
        tf.keras.callbacks.ModelCheckpoint(MODEL_OUT / "phase1_best.keras", save_best_only=True, monitor="val_accuracy"),
    ]

    model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, class_weight=class_weights, callbacks=callbacks)

    model.save(MODEL_OUT / "phase1_final.keras")
    print(f"\nDone. Best model saved to {MODEL_OUT / 'phase1_best.keras'}")


if __name__ == "__main__":
    main()