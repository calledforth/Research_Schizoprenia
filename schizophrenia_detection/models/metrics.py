"""
Custom metrics for model evaluation
"""

import tensorflow as tf
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc


class Sensitivity(tf.keras.metrics.Metric):
    """
    Sensitivity (True Positive Rate) metric
    """

    def __init__(self, name="sensitivity", **kwargs):
        super(Sensitivity, self).__init__(name=name, **kwargs)
        self.true_positives = self.add_weight(name="tp", initializer="zeros")
        self.false_negatives = self.add_weight(name="fn", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.bool)
        y_pred = tf.cast(tf.round(y_pred), tf.bool)

        true_positives = tf.logical_and(tf.equal(y_true, True), tf.equal(y_pred, True))
        false_negatives = tf.logical_and(
            tf.equal(y_true, True), tf.equal(y_pred, False)
        )

        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, tf.float32)
            true_positives = tf.cast(true_positives, tf.float32) * sample_weight
            false_negatives = tf.cast(false_negatives, tf.float32) * sample_weight

        self.true_positives.assign_add(tf.reduce_sum(true_positives))
        self.false_negatives.assign_add(tf.reduce_sum(false_negatives))

    def result(self):
        return self.true_positives / (self.true_positives + self.false_negatives + 1e-7)

    def reset_states(self):
        self.true_positives.assign(0)
        self.false_negatives.assign(0)


class Specificity(tf.keras.metrics.Metric):
    """
    Specificity (True Negative Rate) metric
    """

    def __init__(self, name="specificity", **kwargs):
        super(Specificity, self).__init__(name=name, **kwargs)
        self.true_negatives = self.add_weight(name="tn", initializer="zeros")
        self.false_positives = self.add_weight(name="fp", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.bool)
        y_pred = tf.cast(tf.round(y_pred), tf.bool)

        true_negatives = tf.logical_and(
            tf.equal(y_true, False), tf.equal(y_pred, False)
        )
        false_positives = tf.logical_and(
            tf.equal(y_true, False), tf.equal(y_pred, True)
        )

        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, tf.float32)
            true_negatives = tf.cast(true_negatives, tf.float32) * sample_weight
            false_positives = tf.cast(false_positives, tf.float32) * sample_weight

        self.true_negatives.assign_add(tf.reduce_sum(true_negatives))
        self.false_positives.assign_add(tf.reduce_sum(false_positives))

    def result(self):
        return self.true_negatives / (self.true_negatives + self.false_positives + 1e-7)

    def reset_states(self):
        self.true_negatives.assign(0)
        self.false_positives.assign(0)


class AUC(tf.keras.metrics.Metric):
    """
    Area Under the ROC Curve metric
    """

    def __init__(self, name="auc", **kwargs):
        super(AUC, self).__init__(name=name, **kwargs)
        self.y_true_list = []
        self.y_pred_list = []

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.y_true_list.append(tf.reshape(y_true, [-1]))
        self.y_pred_list.append(tf.reshape(y_pred, [-1]))

    def result(self):
        y_true = tf.concat(self.y_true_list, axis=0)
        y_pred = tf.concat(self.y_pred_list, axis=0)

        # Use sklearn's roc_auc_score
        return tf.py_function(
            func=lambda y_true, y_pred: roc_auc_score(y_true.numpy(), y_pred.numpy()),
            inp=[y_true, y_pred],
            Tout=tf.float32,
        )

    def reset_states(self):
        self.y_true_list = []
        self.y_pred_list = []


class AUPRC(tf.keras.metrics.Metric):
    """
    Area Under the Precision-Recall Curve metric
    """

    def __init__(self, name="auprc", **kwargs):
        super(AUPRC, self).__init__(name=name, **kwargs)
        self.y_true_list = []
        self.y_pred_list = []

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.y_true_list.append(tf.reshape(y_true, [-1]))
        self.y_pred_list.append(tf.reshape(y_pred, [-1]))

    def result(self):
        y_true = tf.concat(self.y_true_list, axis=0)
        y_pred = tf.concat(self.y_pred_list, axis=0)

        # Use sklearn's precision_recall_curve and auc
        def calculate_auprc(y_true, y_pred):
            precision, recall, _ = precision_recall_curve(
                y_true.numpy(), y_pred.numpy()
            )
            return auc(recall, precision)

        return tf.py_function(
            func=calculate_auprc, inp=[y_true, y_pred], Tout=tf.float32
        )

    def reset_states(self):
        self.y_true_list = []
        self.y_pred_list = []


def calculate_f1_score(precision, recall):
    """
    Calculate F1 score from precision and recall

    Args:
        precision: Precision value
        recall: Recall value

    Returns:
        F1 score
    """
    return 2 * (precision * recall) / (precision + recall + 1e-7)


def calculate_accuracy(y_true, y_pred, threshold=0.5):
    """
    Calculate accuracy with a given threshold

    Args:
        y_true: True labels
        y_pred: Predicted probabilities
        threshold (float): Classification threshold

    Returns:
        Accuracy value
    """
    y_pred_classes = (y_pred > threshold).astype(int)
    return (y_pred_classes == y_true).mean()


class F1Score(tf.keras.metrics.Metric):
    """
    F1 Score metric
    """

    def __init__(self, name="f1_score", **kwargs):
        super(F1Score, self).__init__(name=name, **kwargs)
        self.true_positives = self.add_weight(name="tp", initializer="zeros")
        self.false_positives = self.add_weight(name="fp", initializer="zeros")
        self.false_negatives = self.add_weight(name="fn", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.bool)
        y_pred = tf.cast(tf.round(y_pred), tf.bool)

        true_positives = tf.logical_and(tf.equal(y_true, True), tf.equal(y_pred, True))
        false_positives = tf.logical_and(
            tf.equal(y_true, False), tf.equal(y_pred, True)
        )
        false_negatives = tf.logical_and(
            tf.equal(y_true, True), tf.equal(y_pred, False)
        )

        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, tf.float32)
            true_positives = tf.cast(true_positives, tf.float32) * sample_weight
            false_positives = tf.cast(false_positives, tf.float32) * sample_weight
            false_negatives = tf.cast(false_negatives, tf.float32) * sample_weight

        self.true_positives.assign_add(tf.reduce_sum(true_positives))
        self.false_positives.assign_add(tf.reduce_sum(false_positives))
        self.false_negatives.assign_add(tf.reduce_sum(false_negatives))

    def result(self):
        precision = self.true_positives / (
            self.true_positives + self.false_positives + 1e-7
        )
        recall = self.true_positives / (
            self.true_positives + self.false_negatives + 1e-7
        )
        return 2 * (precision * recall) / (precision + recall + 1e-7)

    def reset_states(self):
        self.true_positives.assign(0)
        self.false_positives.assign(0)
        self.false_negatives.assign(0)


class Precision(tf.keras.metrics.Metric):
    """
    Precision metric
    """

    def __init__(self, name="precision", **kwargs):
        super(Precision, self).__init__(name=name, **kwargs)
        self.true_positives = self.add_weight(name="tp", initializer="zeros")
        self.false_positives = self.add_weight(name="fp", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.bool)
        y_pred = tf.cast(tf.round(y_pred), tf.bool)

        true_positives = tf.logical_and(tf.equal(y_true, True), tf.equal(y_pred, True))
        false_positives = tf.logical_and(
            tf.equal(y_true, False), tf.equal(y_pred, True)
        )

        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, tf.float32)
            true_positives = tf.cast(true_positives, tf.float32) * sample_weight
            false_positives = tf.cast(false_positives, tf.float32) * sample_weight

        self.true_positives.assign_add(tf.reduce_sum(true_positives))
        self.false_positives.assign_add(tf.reduce_sum(false_positives))

    def result(self):
        return self.true_positives / (self.true_positives + self.false_positives + 1e-7)

    def reset_states(self):
        self.true_positives.assign(0)
        self.false_positives.assign(0)


class MatthewsCorrelationCoefficient(tf.keras.metrics.Metric):
    """
    Matthews Correlation Coefficient (MCC) metric
    """

    def __init__(self, name="mcc", **kwargs):
        super(MatthewsCorrelationCoefficient, self).__init__(name=name, **kwargs)
        self.true_positives = self.add_weight(name="tp", initializer="zeros")
        self.true_negatives = self.add_weight(name="tn", initializer="zeros")
        self.false_positives = self.add_weight(name="fp", initializer="zeros")
        self.false_negatives = self.add_weight(name="fn", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.bool)
        y_pred = tf.cast(tf.round(y_pred), tf.bool)

        true_positives = tf.logical_and(tf.equal(y_true, True), tf.equal(y_pred, True))
        true_negatives = tf.logical_and(
            tf.equal(y_true, False), tf.equal(y_pred, False)
        )
        false_positives = tf.logical_and(
            tf.equal(y_true, False), tf.equal(y_pred, True)
        )
        false_negatives = tf.logical_and(
            tf.equal(y_true, True), tf.equal(y_pred, False)
        )

        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, tf.float32)
            true_positives = tf.cast(true_positives, tf.float32) * sample_weight
            true_negatives = tf.cast(true_negatives, tf.float32) * sample_weight
            false_positives = tf.cast(false_positives, tf.float32) * sample_weight
            false_negatives = tf.cast(false_negatives, tf.float32) * sample_weight

        self.true_positives.assign_add(tf.reduce_sum(true_positives))
        self.true_negatives.assign_add(tf.reduce_sum(true_negatives))
        self.false_positives.assign_add(tf.reduce_sum(false_positives))
        self.false_negatives.assign_add(tf.reduce_sum(false_negatives))

    def result(self):
        tp = self.true_positives
        tn = self.true_negatives
        fp = self.false_positives
        fn = self.false_negatives

        numerator = tp * tn - fp * fn
        denominator = tf.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

        return numerator / (denominator + 1e-7)

    def reset_states(self):
        self.true_positives.assign(0)
        self.true_negatives.assign(0)
        self.false_positives.assign(0)
        self.false_negatives.assign(0)


class BalancedAccuracy(tf.keras.metrics.Metric):
    """
    Balanced Accuracy metric (average of sensitivity and specificity)
    """

    def __init__(self, name="balanced_accuracy", **kwargs):
        super(BalancedAccuracy, self).__init__(name=name, **kwargs)
        self.true_positives = self.add_weight(name="tp", initializer="zeros")
        self.true_negatives = self.add_weight(name="tn", initializer="zeros")
        self.false_positives = self.add_weight(name="fp", initializer="zeros")
        self.false_negatives = self.add_weight(name="fn", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.bool)
        y_pred = tf.cast(tf.round(y_pred), tf.bool)

        true_positives = tf.logical_and(tf.equal(y_true, True), tf.equal(y_pred, True))
        true_negatives = tf.logical_and(
            tf.equal(y_true, False), tf.equal(y_pred, False)
        )
        false_positives = tf.logical_and(
            tf.equal(y_true, False), tf.equal(y_pred, True)
        )
        false_negatives = tf.logical_and(
            tf.equal(y_true, True), tf.equal(y_pred, False)
        )

        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, tf.float32)
            true_positives = tf.cast(true_positives, tf.float32) * sample_weight
            true_negatives = tf.cast(true_negatives, tf.float32) * sample_weight
            false_positives = tf.cast(false_positives, tf.float32) * sample_weight
            false_negatives = tf.cast(false_negatives, tf.float32) * sample_weight

        self.true_positives.assign_add(tf.reduce_sum(true_positives))
        self.true_negatives.assign_add(tf.reduce_sum(true_negatives))
        self.false_positives.assign_add(tf.reduce_sum(false_positives))
        self.false_negatives.assign_add(tf.reduce_sum(false_negatives))

    def result(self):
        sensitivity = self.true_positives / (
            self.true_positives + self.false_negatives + 1e-7
        )
        specificity = self.true_negatives / (
            self.true_negatives + self.false_positives + 1e-7
        )
        return (sensitivity + specificity) / 2

    def reset_states(self):
        self.true_positives.assign(0)
        self.true_negatives.assign(0)
        self.false_positives.assign(0)
        self.false_negatives.assign(0)


def calculate_sensitivity_specificity(y_true, y_pred, threshold=0.5):
    """
    Calculate sensitivity and specificity

    Args:
        y_true: True labels
        y_pred: Predicted probabilities
        threshold (float): Classification threshold

    Returns:
        Tuple of (sensitivity, specificity)
    """
    y_pred_classes = (y_pred > threshold).astype(int)

    # Calculate confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_classes).ravel()

    # Calculate sensitivity and specificity
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    return sensitivity, specificity


def calculate_optimal_threshold(y_true, y_pred_proba):
    """
    Calculate the optimal threshold using Youden's J statistic

    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities

    Returns:
        Optimal threshold value
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    youden_j = tpr - fpr
    optimal_idx = np.argmax(youden_j)
    return thresholds[optimal_idx]


def calculate_metrics_at_threshold(y_true, y_pred_proba, threshold):
    """
    Calculate various metrics at a specific threshold

    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        threshold: Classification threshold

    Returns:
        Dictionary of metrics
    """
    y_pred_classes = (y_pred_proba > threshold).astype(int)

    # Calculate confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_classes).ravel()

    # Calculate metrics
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1 = (
        2 * (precision * sensitivity) / (precision + sensitivity)
        if (precision + sensitivity) > 0
        else 0
    )

    # Calculate Matthews correlation coefficient
    mcc = (
        (tp * tn - fp * fn) / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        if (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn) > 0
        else 0
    )

    return {
        "accuracy": accuracy,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "f1_score": f1,
        "mcc": mcc,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def get_metric(metric_name):
    """
    Get a metric by name

    Args:
        metric_name (str): Name of the metric

    Returns:
        Metric object or function
    """
    metric_name = metric_name.lower()

    if metric_name == "accuracy":
        return "accuracy"
    elif metric_name == "precision":
        return Precision()
    elif metric_name == "recall" or metric_name == "sensitivity":
        return Sensitivity()
    elif metric_name == "specificity":
        return Specificity()
    elif metric_name == "f1_score":
        return F1Score()
    elif metric_name == "auc":
        return AUC()
    elif metric_name == "auprc":
        return AUPRC()
    elif metric_name == "mcc":
        return MatthewsCorrelationCoefficient()
    elif metric_name == "balanced_accuracy":
        return BalancedAccuracy()
    else:
        raise ValueError(f"Unknown metric: {metric_name}")
