"""
training/finetune_v2.py

Phase 2 v3: conservative fine-tune starting from phase1_best.keras.
Key fixes vs previous attempt:
- Only last 10 layers unfrozen (was 30 -- too aggressive)
- LR = 1e-5 (was 5e-5 -- too high)
- Label smoothing = 0.1 (prevents overconfident collapse)
- Dropout rebuilt correctly via a wrapper model
- Augmentation kept the same (was fine)
"""

from pathlib import Path
import tensorflow as tf

SPLIT_DIR = Path(r"F:\capstone\retinal_disease_detector\dataset_prep\split")
PHASE1_MODEL = Path(r"F:\capstone\retinal_disease_detector\training\checkpoints\phase1_best.keras")
MODEL_OUT = Path(r"F:\capstone\retinal_disease_detector\training\checkpoints")

IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 20
UNFREEZE_LAST_N = 10  # only last 10 of 154 base layers

CLASS_ORDER = ["Normal", "Diabetic Retinopathy", "Glaucoma", "Cataract", "AMD"]


def build_datasets():
    def augment(image, label):
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_flip_up_down(image)
        image = tf.image.random_brightness(image, max_delta=0.2)
        image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
        image = tf.image.random_saturation(image, lower=0.9, upper=1.1)
        image = tf.clip_by_value(image, 0, 255)
        return image, label

    train_ds = (
        tf.keras.utils.image_dataset_from_directory(
            SPLIT_DIR / "train", class_names=CLASS_ORDER,
            image_size=IMG_SIZE, batch_size=BATCH_SIZE,
            label_mode="categorical", shuffle=True, seed=42,
        )
        .map(augment, num_parallel_calls=tf.data.AUTOTUNE)
        .prefetch(tf.data.AUTOTUNE)
    )
    val_ds = (
        tf.keras.utils.image_dataset_from_directory(
            SPLIT_DIR / "val", class_names=CLASS_ORDER,
            image_size=IMG_SIZE, batch_size=BATCH_SIZE,
            label_mode="categorical", shuffle=False,
        )
        .prefetch(tf.data.AUTOTUNE)
    )
    return train_ds, val_ds


def compute_class_weights(train_dir: Path) -> dict:
    counts = {cls: len(list((train_dir / cls).glob("*.jpg")))
              for cls in CLASS_ORDER}
    total = sum(counts.values())
    weights = {
        i: total / (len(CLASS_ORDER) * counts[cls])
        for i, cls in enumerate(CLASS_ORDER)
    }
    print("Class weights:",
          {CLASS_ORDER[i]: round(w, 2) for i, w in weights.items()})
    return weights


def main():
    # ── Load phase 1 model ────────────────────────────────────────
    model = tf.keras.models.load_model(PHASE1_MODEL)

    # ── Find the MobileNetV2 base inside the model ─────────────────
    base = next(l for l in model.layers
                if isinstance(l, tf.keras.Model))
    print(f"Base: {base.name}, total layers: {len(base.layers)}")

    # ── Unfreeze only last 10 layers ──────────────────────────────
    base.trainable = True
    for layer in base.layers[:-UNFREEZE_LAST_N]:
        layer.trainable = False

    unfrozen = sum(1 for l in base.layers if l.trainable)
    print(f"Unfrozen base layers: {unfrozen} of {len(base.layers)}")

    # ── Rebuild model with higher dropout ─────────────────────────
    # We rebuild the top so dropout rate change actually takes effect.
    # The base (with its internal preprocess_input) stays identical.
    inputs = tf.keras.Input(shape=IMG_SIZE + (3,))

    # Replay the same graph as phase 1 but with dropout=0.5
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
    x = base(x, training=True)  # training=True = BatchNorm uses batch stats
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.5)(x)  # was 0.3 in phase 1
    outputs = tf.keras.layers.Dense(
        len(CLASS_ORDER), activation="softmax"
    )(x)

    new_model = tf.keras.Model(inputs, outputs)

    # Copy weights from old dense head into new model
    # Layer order: preprocess → base → GAP → dropout → dense
    old_dense = model.layers[-1]   # Dense layer from phase 1
    new_dense = new_model.layers[-1]
    new_dense.set_weights(old_dense.get_weights())

    # ── Compile with label smoothing ──────────────────────────────
    # Label smoothing 0.1: instead of [0,0,1,0,0] target becomes
    # [0.02, 0.02, 0.92, 0.02, 0.02] — stops the model from becoming
    # overconfident on any single class, which caused the collapse.
    new_model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-5),  # very low LR
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics=["accuracy"],
    )

    train_ds, val_ds = build_datasets()
    class_weights = compute_class_weights(SPLIT_DIR / "train")

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            patience=5, restore_best_weights=True, monitor="val_accuracy"
        ),
        tf.keras.callbacks.ModelCheckpoint(
            MODEL_OUT / "phase2v3_best.keras",
            save_best_only=True, monitor="val_accuracy"
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_accuracy", factor=0.5,
            patience=3, min_lr=1e-7, verbose=1
        ),
    ]

    new_model.fit(
        train_ds, validation_data=val_ds,
        epochs=EPOCHS, class_weight=class_weights,
        callbacks=callbacks
    )
    new_model.save(MODEL_OUT / "phase2v3_final.keras")
    print(f"\nDone. Best saved to {MODEL_OUT / 'phase2v3_best.keras'}")


if __name__ == "__main__":
    main()