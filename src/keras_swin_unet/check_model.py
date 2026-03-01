import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

import sys
sys.path.append("/home/user_one/projects/GEOINT/keras-swin-unet/src")

#from swin_transformer.AUC_LOSS import auc_focal_loss
import keras_swin_unet.transformer_layers as transformer_layers
import keras_swin_unet.swin_layers as swin_layers

def auc_focal_loss(alpha=0.25, gamma=2.0):
    auc_metric = tf.keras.metrics.AUC()

    def auc_focal_loss_fixed(y_true, y_pred):
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
        y_true = tf.cast(y_true, tf.float32)

        alpha_t = y_true * alpha + (tf.keras.backend.ones_like(y_true) - y_true) * (
            1 - alpha
        )
        p_t = y_true * y_pred + (tf.keras.backend.ones_like(y_true) - y_true) * (
            1 - y_pred
        )
        fl = (
            -alpha_t
            * tf.keras.backend.pow((tf.keras.backend.ones_like(y_true) - p_t), gamma)
            * tf.keras.backend.log(p_t)
        )

        y_true_f = tf.reshape(y_true, [-1])
        y_pred_f = tf.reshape(y_pred, [-1])
        auc_metric.update_state(y_true_f, y_pred_f)
        auc = auc_metric.result()

        return fl + (1 - auc)

    return auc_focal_loss_fixed


# Manually define needed parameters
alpha = 0.25
gamma = 2.0

custom_objects = {
    "auc_focal_loss_fixed": auc_focal_loss(alpha=alpha, gamma=gamma),
    **transformer_layers.__dict__,
    **swin_layers.__dict__,
}

model = load_model(
    "/home/user_one/projects/GEOINT/Model_Checkpoints/best_model.keras",
    custom_objects=custom_objects,
    compile=False   # Important: avoids optimizer issues
)

model.summary()

print(model.layers[-1].activation)