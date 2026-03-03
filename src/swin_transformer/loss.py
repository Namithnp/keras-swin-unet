import tensorflow as tf
from tensorflow.keras import backend as K

'''
alpha = [
    0.034,   # Background
    0.083,   # Building
    0.280,   # Road
    0.269,   # Utilities (capped)
    0.334    # WaterBodies
]

dice_class_weights = [
    0.088,   # Background
    0.139,   # Building
    0.254,   # Road
    0.242,   # Utilities
    0.277    # WaterBodies
]
'''

# -------------------------------------------------------
# 1️⃣ CATEGORICAL FOCAL LOSS (Multiclass, Softmax)
# -------------------------------------------------------
def categorical_focal_loss(alpha, gamma=1.5):

    alpha = tf.constant(alpha, dtype=tf.float32)

    def loss(y_true, y_pred):
        epsilon = K.epsilon()

        # Clip predictions to prevent log(0)
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)

        # Cross entropy
        cross_entropy = -y_true * tf.math.log(y_pred)

        # Focal factor
        alpha_factor = y_true * alpha
        focal_factor = alpha_factor * tf.pow(1 - y_pred, gamma)

        # Apply focal weighting
        loss = focal_factor * cross_entropy

        # Sum over classes
        loss = tf.reduce_sum(loss, axis=-1)

        return tf.reduce_mean(loss)

    return loss


# -------------------------------------------------------
# 2️⃣ MULTICLASS SOFT DICE LOSS
# -------------------------------------------------------
def dice_loss(dice_class_weights, smooth=1e-6):

    class_weights = tf.constant(dice_class_weights, dtype=tf.float32)

    def loss(y_true, y_pred):

        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        # Create mask: ignore pixels where all classes are zero
        valid_mask = tf.reduce_sum(y_true, axis=-1, keepdims=True) > 0
        valid_mask = tf.cast(valid_mask, tf.float32)

        y_true = y_true * valid_mask
        y_pred = y_pred * valid_mask

        axes = (1, 2)

        intersection = tf.reduce_sum(y_true * y_pred, axis=axes)
        denominator = tf.reduce_sum(y_true + y_pred, axis=axes)

        dice = (2. * intersection + smooth) / (denominator + smooth)

        dice = dice * class_weights
        dice = tf.reduce_sum(dice, axis=-1)

        return 1 - tf.reduce_mean(dice)

    return loss


# -------------------------------------------------------
# 3️⃣ COMBINED LOSS
# -------------------------------------------------------
def focal_dice_loss(alpha, gamma=1.5, dice_weight=0.6, dice_class_weights=None):

    focal = categorical_focal_loss(alpha=alpha, gamma=gamma)
    dice = dice_loss(dice_class_weights=dice_class_weights)

    def loss(y_true, y_pred):
        return (1 - dice_weight) * focal(y_true, y_pred) + \
               dice_weight * dice(y_true, y_pred)

    return loss