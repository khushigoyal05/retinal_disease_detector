"""
training/unet_model.py

U-Net with MobileNetV2 encoder for cup segmentation.
Encoder: pretrained MobileNetV2 (feature extraction, Pi-friendly, matches
locked architecture decision for all models in this project).
Decoder: standard U-Net upsampling path with skip connections from encoder.
"""

import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import MobileNetV2


def build_cup_unet(input_shape=(224, 224, 3)):
    inputs = layers.Input(shape=input_shape)

    # Encoder: pretrained MobileNetV2, frozen at first (transfer learning)
    base_model = MobileNetV2(input_tensor=inputs, include_top=False, weights="imagenet")

    # Pull out skip-connection layers at different resolutions.
    # These layer names are MobileNetV2's internal names at each downsample stage.
    skip_names = [
        "block_1_expand_relu",   # 112x112
        "block_3_expand_relu",   # 56x56
        "block_6_expand_relu",   # 28x28
        "block_13_expand_relu",  # 14x14
    ]
    skips = [base_model.get_layer(name).output for name in skip_names]
    bottleneck = base_model.get_layer("block_16_project").output  # 7x7, deepest features

    # Decoder: upsample step by step, concatenating matching skip connection each time
    x = bottleneck
    decoder_filters = [512, 256, 128, 64]
    for filters, skip in zip(decoder_filters, reversed(skips)):
        x = layers.Conv2DTranspose(filters, kernel_size=3, strides=2, padding="same")(x)
        x = layers.Concatenate()([x, skip])
        x = layers.Conv2D(filters, kernel_size=3, padding="same", activation="relu")(x)
        x = layers.Conv2D(filters, kernel_size=3, padding="same", activation="relu")(x)

    # Final upsample to full 224x224, then 1-channel sigmoid output (cup probability per pixel)
    x = layers.Conv2DTranspose(32, kernel_size=3, strides=2, padding="same")(x)
    x = layers.Conv2D(32, kernel_size=3, padding="same", activation="relu")(x)
    outputs = layers.Conv2D(1, kernel_size=1, activation="sigmoid")(x)

    model = Model(inputs, outputs)

    # Freeze encoder initially — train decoder first (same phase 1/2 strategy
    # as your disease classifier: frozen base, then fine-tune later if needed)
    base_model.trainable = False

    return model