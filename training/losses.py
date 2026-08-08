"""
training/losses.py

Dice + BCE loss for cup segmentation.
WHY combined: BCE alone tends to under-segment small regions (background
dominates pixel count). Dice directly rewards correct overlap shape.
Together they give both pixel-level and shape-level accuracy.
"""

import tensorflow as tf


def dice_loss(y_true, y_pred, smooth=1e-6):
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    dice_coef = (2.0 * intersection + smooth) / (
        tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth
    )
    return 1.0 - dice_coef


def dice_bce_loss(y_true, y_pred):
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    bce = tf.reduce_mean(bce)
    dice = dice_loss(y_true, y_pred)
    return bce + dice