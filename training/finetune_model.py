"""
training/finetune_model.py

Phase 2: loads the Phase 1 best model, unfreezes the top layers of the
MobileNetV2 base, and fine-tunes at a very low learning rate. This adapts
the feature extractor to retinal images without destroying the pretrained
ImageNet weights (catastrophic forgetting).
"""

from pathlib import Path
import tensorflow as tf

SPLIT_DIR = Path(r"F:\capstone\retinal_disease_detector\dataset_prep\split")
PHASE1_MODEL = Path(r"F:\capstone\retinal_disease_detector\training\checkpoints\phase1_best.keras")
MODEL_OUT = Path(r"F:\capstone\retinal_disease_detector\training\checkpoints")

IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 20
UNFREEZE_FROM_LAYER = 100  # MobileNetV2 has 154 layers total; unfreeze top ~54

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
    return train_ds.prefetch(tf.data.AUTOTUNE), val_ds.prefetch(tf.data.AUTOTUNE)


def compute_class_weights(train_dir: Path) -> dict:
    counts = {cls: len(list((train_dir / cls).glob("*.jpg"))) for cls in CLASS_ORDER}
    total = sum(counts.values())
    weights = {i: total / (len(CLASS_ORDER) * counts[cls]) for i, cls in enumerate(CLASS_ORDER)}
    print("Class weights:", {CLASS_ORDER[i]: round(w, 2) for i, w in weights.items()})
    return weights


def main():
    model = tf.keras.models.load_model(PHASE1_MODEL)

    # find the MobileNetV2 base layer inside our functional model
    base = next(l for l in model.layers if isinstance(l, tf.keras.Model))
    print(f"Base model: {base.name}, total layers: {len(base.layers)}")

    # unfreeze everything from layer 100 onwards; keep earlier layers frozen
    # (earlier layers = basic edges/textures, already generic enough to reuse)
    base.trainable = True
    for layer in base.layers[:UNFREEZE_FROM_LAYER]:
        layer.trainable = False

    trainable_count = sum(1 for l in base.layers if l.trainable)
    print(f"Unfrozen base layers: {trainable_count} of {len(base.layers)}")

    # 10x lower LR than Phase 1 -- nudge weights, not overwrite them
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    train_ds, val_ds = build_datasets()
    class_weights = compute_class_weights(SPLIT_DIR / "train")

    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True, monitor="val_accuracy"),
        tf.keras.callbacks.ModelCheckpoint(MODEL_OUT / "phase2_best.keras", save_best_only=True, monitor="val_accuracy"),
        # reduce LR if val_accuracy plateaus for 3 epochs
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_accuracy", factor=0.5, patience=3, min_lr=1e-6, verbose=1),
    ]

    model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, class_weight=class_weights, callbacks=callbacks)

    model.save(MODEL_OUT / "phase2_final.keras")
    print(f"\nDone. Best model saved to {MODEL_OUT / 'phase2_best.keras'}")


if __name__ == "__main__":
    main()