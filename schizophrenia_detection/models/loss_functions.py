"""
Custom loss functions for the schizophrenia detection model
"""

import tensorflow as tf


def focal_loss(gamma=2.0, alpha=0.25):
    """
    Focal loss for addressing class imbalance

    Args:
        gamma (float): Focusing parameter
        alpha (float): Weighting factor

    Returns:
        Loss function
    """

    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        # Clip predictions to prevent log(0)
        y_pred = tf.clip_by_value(y_pred, 1e-8, 1 - 1e-8)

        # Calculate cross entropy
        cross_entropy = -y_true * tf.math.log(y_pred)

        # Calculate focal weight
        weight = alpha * tf.pow(1 - y_pred, gamma)

        # Calculate focal loss
        focal_loss = weight * cross_entropy

        return tf.reduce_sum(focal_loss, axis=-1)

    return loss


def weighted_binary_crossentropy(class_weights):
    """
    Weighted binary cross-entropy loss

    Args:
        class_weights (dict): Dictionary of class weights

    Returns:
        Loss function
    """

    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        # Clip predictions to prevent log(0)
        y_pred = tf.clip_by_value(y_pred, 1e-8, 1 - 1e-8)

        # Calculate binary cross entropy
        bce = -(y_true * tf.math.log(y_pred) + (1 - y_true) * tf.math.log(1 - y_pred))

        # Apply class weights
        weights = y_true * class_weights[1] + (1 - y_true) * class_weights[0]
        weighted_bce = weights * bce

        return tf.reduce_mean(weighted_bce)

    return loss


def dice_loss(smooth=1e-6):
    """
    Dice loss for segmentation tasks

    Args:
        smooth (float): Smoothing factor

    Returns:
        Loss function
    """

    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        # Calculate intersection and union
        intersection = tf.reduce_sum(y_true * y_pred)
        union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred)

        # Calculate dice coefficient
        dice = (2.0 * intersection + smooth) / (union + smooth)

        return 1.0 - dice

    return loss


def combined_loss(losses, weights=None):
    """
    Combine multiple loss functions

    Args:
        losses (list): List of loss functions
        weights (list): List of weights for each loss function

    Returns:
        Combined loss function
    """
    if weights is None:
        weights = [1.0] * len(losses)

    def loss(y_true, y_pred):
        total_loss = 0.0
        for loss_fn, weight in zip(losses, weights):
            total_loss += weight * loss_fn(y_true, y_pred)
        return total_loss

    return loss


def binary_focal_loss(gamma=2.0, alpha=0.25):
    """
    Binary focal loss for addressing class imbalance in binary classification

    Args:
        gamma (float): Focusing parameter
        alpha (float): Weighting factor for positive class

    Returns:
        Loss function
    """

    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        # Clip predictions to prevent log(0)
        y_pred = tf.clip_by_value(y_pred, 1e-8, 1 - 1e-8)

        # Calculate binary cross entropy
        bce = -(y_true * tf.math.log(y_pred) + (1 - y_true) * tf.math.log(1 - y_pred))

        # Calculate focal weight
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        alpha_t = y_true * alpha + (1 - y_true) * (1 - alpha)
        weight = alpha_t * tf.pow(1 - p_t, gamma)

        # Calculate focal loss
        focal_loss = weight * bce

        return tf.reduce_mean(focal_loss)

    return loss


def tversky_loss(alpha=0.7, beta=0.3, smooth=1e-6):
    """
    Tversky loss for imbalanced classification problems

    Args:
        alpha (float): Weight for false positives
        beta (float): Weight for false negatives
        smooth (float): Smoothing factor

    Returns:
        Loss function
    """

    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        # Flatten the tensors
        y_true = tf.reshape(y_true, [-1])
        y_pred = tf.reshape(y_pred, [-1])

        # Calculate true positives, false positives, and false negatives
        true_positives = tf.reduce_sum(y_true * y_pred)
        false_positives = tf.reduce_sum((1 - y_true) * y_pred)
        false_negatives = tf.reduce_sum(y_true * (1 - y_pred))

        # Calculate Tversky index
        tversky = (true_positives + smooth) / (
            true_positives + alpha * false_positives + beta * false_negatives + smooth
        )

        return 1.0 - tversky

    return loss


def focal_tversky_loss(gamma=2.0, alpha=0.7, beta=0.3, smooth=1e-6):
    """
    Focal Tversky loss for highly imbalanced classification problems

    Args:
        gamma (float): Focusing parameter
        alpha (float): Weight for false positives
        beta (float): Weight for false negatives
        smooth (float): Smoothing factor

    Returns:
        Loss function
    """

    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        # Flatten the tensors
        y_true = tf.reshape(y_true, [-1])
        y_pred = tf.reshape(y_pred, [-1])

        # Calculate true positives, false positives, and false negatives
        true_positives = tf.reduce_sum(y_true * y_pred)
        false_positives = tf.reduce_sum((1 - y_true) * y_pred)
        false_negatives = tf.reduce_sum(y_true * (1 - y_pred))

        # Calculate Tversky index
        tversky = (true_positives + smooth) / (
            true_positives + alpha * false_positives + beta * false_negatives + smooth
        )

        # Apply focal term
        focal_tversky = tf.pow(1.0 - tversky, gamma)

        return focal_tversky

    return loss


def log_cosh_dice_loss(smooth=1e-6):
    """
    Log-cosh Dice loss for more stable training

    Args:
        smooth (float): Smoothing factor

    Returns:
        Loss function
    """

    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        # Calculate intersection and union
        intersection = tf.reduce_sum(y_true * y_pred)
        union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred)

        # Calculate dice coefficient
        dice = (2.0 * intersection + smooth) / (union + smooth)

        # Apply log-cosh
        return tf.math.log(tf.cosh(1.0 - dice))

    return loss


def asymmetric_loss(gamma_neg=4, gamma_pos=1, clip=0.05):
    """
    Asymmetric loss for highly imbalanced classification

    Args:
        gamma_neg (float): Focusing parameter for negative class
        gamma_pos (float): Focusing parameter for positive class
        clip (float): Probability clipping value

    Returns:
        Loss function
    """

    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        # Clip predictions
        y_pred = tf.clip_by_value(y_pred, clip, 1 - clip)

        # Calculate cross entropy
        cross_entropy = -(
            y_true * tf.math.log(y_pred) + (1 - y_true) * tf.math.log(1 - y_pred)
        )

        # Calculate asymmetric focusing weights
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        focusing_weight = tf.pow(1 - p_t, y_true * gamma_pos + (1 - y_true) * gamma_neg)

        # Calculate asymmetric loss
        asymmetric_loss = focusing_weight * cross_entropy

        return tf.reduce_mean(asymmetric_loss)

    return loss


def get_loss_function(loss_name):
    """
    Get a loss function by name

    Args:
        loss_name (str): Name of the loss function

    Returns:
        Loss function
    """
    loss_name = loss_name.lower()

    if loss_name == "binary_crossentropy":
        return tf.keras.losses.binary_crossentropy
    elif loss_name == "categorical_crossentropy":
        return tf.keras.losses.categorical_crossentropy
    elif loss_name == "focal_loss":
        return focal_loss()
    elif loss_name == "binary_focal_loss":
        return binary_focal_loss()
    elif loss_name == "weighted_binary_crossentropy":
        # Default class weights for schizophrenia detection
        class_weights = {0: 1.0, 1: 1.5}  # Higher weight for positive class
        return weighted_binary_crossentropy(class_weights)
    elif loss_name == "dice_loss":
        return dice_loss()
    elif loss_name == "tversky_loss":
        return tversky_loss()
    elif loss_name == "focal_tversky_loss":
        return focal_tversky_loss()
    elif loss_name == "log_cosh_dice_loss":
        return log_cosh_dice_loss()
    elif loss_name == "asymmetric_loss":
        return asymmetric_loss()
    else:
        raise ValueError(f"Unknown loss function: {loss_name}")
