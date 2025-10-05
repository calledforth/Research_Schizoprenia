"""
Comprehensive model evaluation script for the SSPNet 3D CNN model
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
)
from ..config import default_config
from ..utils.logging_utils import setup_logger


class ModelEvaluator:
    """
    Comprehensive class for evaluating the schizophrenia detection model
    """

    def __init__(self, model, config=None):
        """
        Initialize the model evaluator

        Args:
            model: Trained model to evaluate
            config: Configuration object
        """
        self.model = model
        self.config = config if config is not None else default_config
        self.results = {}
        self.logger = setup_logger(
            "evaluator", self.config.logging.log_file, self.config.logging.level
        )

        # Create results directory if it doesn't exist
        os.makedirs(self.config.evaluation.results_dir, exist_ok=True)

    def evaluate(self, test_data, batch_size=None):
        """
        Evaluate the model on test data with comprehensive metrics

        Args:
            test_data: Test data generator or dataset
            batch_size: Batch size for evaluation (if None, uses config batch size)

        Returns:
            Dictionary of evaluation metrics
        """
        self.logger.info("Starting model evaluation...")

        # Set batch size
        if batch_size is None:
            batch_size = self.config.data.batch_size

        # Get model predictions
        y_true = []
        y_pred = []
        y_pred_proba = []

        # Process data in batches
        for batch_x, batch_y in test_data:
            batch_pred = self.model.predict(batch_x, verbose=0)

            # Handle different label formats
            if len(batch_y.shape) > 1 and batch_y.shape[1] > 1:
                # One-hot encoded labels
                y_true.extend(np.argmax(batch_y, axis=1))
            else:
                # Binary labels
                y_true.extend(batch_y.numpy().flatten())

            # Get predicted classes and probabilities
            y_pred.extend(np.argmax(batch_pred, axis=1))
            y_pred_proba.extend(
                batch_pred[:, 1].numpy()
            )  # Probability of positive class

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        y_pred_proba = np.array(y_pred_proba)

        self.logger.info(f"Evaluated {len(y_true)} samples")

        # Calculate basic metrics
        self.results["accuracy"] = accuracy_score(y_true, y_pred)
        self.results["precision"] = precision_score(y_true, y_pred, zero_division=0)
        self.results["recall"] = recall_score(y_true, y_pred, zero_division=0)
        self.results["f1_score"] = f1_score(y_true, y_pred, zero_division=0)

        # Calculate sensitivity (same as recall for binary classification)
        self.results["sensitivity"] = recall_score(y_true, y_pred, zero_division=0)

        # Calculate specificity (true negative rate)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        self.results["specificity"] = tn / (tn + fp) if (tn + fp) > 0 else 0

        # Calculate confusion matrix
        self.results["confusion_matrix"] = confusion_matrix(y_true, y_pred)

        # Calculate classification report
        self.results["classification_report"] = classification_report(
            y_true, y_pred, target_names=["Control", "Schizophrenia"], output_dict=True
        )

        # ROC curve and AUC
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
        self.results["roc_curve"] = {"fpr": fpr, "tpr": tpr, "thresholds": thresholds}
        self.results["auc"] = auc(fpr, tpr)

        # Precision-Recall curve
        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
        self.results["pr_curve"] = {"precision": precision, "recall": recall}
        self.results["average_precision"] = average_precision_score(
            y_true, y_pred_proba
        )

        # Calculate optimal threshold using Youden's J statistic
        youden_j = tpr - fpr
        optimal_idx = np.argmax(youden_j)
        self.results["optimal_threshold"] = thresholds[optimal_idx]
        self.results["optimal_sensitivity"] = tpr[optimal_idx]
        self.results["optimal_specificity"] = 1 - fpr[optimal_idx]

        # Store raw predictions for further analysis
        self.results["y_true"] = y_true
        self.results["y_pred"] = y_pred
        self.results["y_pred_proba"] = y_pred_proba

        # Log results
        self.logger.info(f"Evaluation Results:")
        self.logger.info(f"  Accuracy (ACC): {self.results['accuracy']:.4f}")
        self.logger.info(f"  Sensitivity (SEN): {self.results['sensitivity']:.4f}")
        self.logger.info(f"  Specificity (SPEC): {self.results['specificity']:.4f}")
        self.logger.info(f"  Precision: {self.results['precision']:.4f}")
        self.logger.info(f"  F1 Score: {self.results['f1_score']:.4f}")
        self.logger.info(f"  AUC: {self.results['auc']:.4f}")
        self.logger.info(
            f"  Average Precision: {self.results['average_precision']:.4f}"
        )
        self.logger.info(
            f"  Optimal Threshold: {self.results['optimal_threshold']:.4f}"
        )

        return self.results

    def plot_confusion_matrix(self, save_path=None, normalize=False):
        """
        Plot the confusion matrix

        Args:
            save_path (str): Path to save the plot
            normalize (bool): Whether to normalize the confusion matrix
        """
        if "confusion_matrix" not in self.results:
            self.logger.warning(
                "No evaluation results available. Please run evaluate() first."
            )
            return

        cm = self.results["confusion_matrix"]

        if normalize:
            cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
            title = "Normalized Confusion Matrix"
            fmt = ".2f"
        else:
            title = "Confusion Matrix"
            fmt = "d"

        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt=fmt,
            cmap="Blues",
            xticklabels=["Control", "Schizophrenia"],
            yticklabels=["Control", "Schizophrenia"],
        )
        plt.title(title)
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")

        if save_path:
            plt.savefig(
                save_path, dpi=self.config.visualization.dpi, bbox_inches="tight"
            )
            self.logger.info(f"Confusion matrix plot saved to {save_path}")
        else:
            save_path = os.path.join(
                self.config.evaluation.results_dir, "confusion_matrix.png"
            )
            plt.savefig(
                save_path, dpi=self.config.visualization.dpi, bbox_inches="tight"
            )
            self.logger.info(f"Confusion matrix plot saved to {save_path}")

        plt.show()

    def plot_roc_curve(self, save_path=None):
        """
        Plot the ROC curve

        Args:
            save_path (str): Path to save the plot
        """
        if "roc_curve" not in self.results:
            self.logger.warning(
                "No evaluation results available. Please run evaluate() first."
            )
            return

        fpr = self.results["roc_curve"]["fpr"]
        tpr = self.results["roc_curve"]["tpr"]
        roc_auc = self.results["auc"]

        # Plot ROC curve
        plt.figure(figsize=self.config.visualization.figure_size)
        plt.plot(
            fpr,
            tpr,
            color="darkorange",
            lw=2,
            label=f"ROC curve (area = {roc_auc:.4f})",
        )
        plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")

        # Mark optimal threshold
        optimal_idx = np.argmax(
            self.results["roc_curve"]["tpr"] - self.results["roc_curve"]["fpr"]
        )
        plt.plot(
            fpr[optimal_idx],
            tpr[optimal_idx],
            marker="o",
            markersize=10,
            label=f"Optimal threshold = {self.results['optimal_threshold']:.4f}",
        )

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Receiver Operating Characteristic (ROC) Curve")
        plt.legend(loc="lower right")

        if save_path:
            plt.savefig(
                save_path, dpi=self.config.visualization.dpi, bbox_inches="tight"
            )
            self.logger.info(f"ROC curve plot saved to {save_path}")
        else:
            save_path = os.path.join(
                self.config.evaluation.results_dir, "roc_curve.png"
            )
            plt.savefig(
                save_path, dpi=self.config.visualization.dpi, bbox_inches="tight"
            )
            self.logger.info(f"ROC curve plot saved to {save_path}")

        plt.show()

    def plot_precision_recall_curve(self, save_path=None):
        """
        Plot the Precision-Recall curve

        Args:
            save_path (str): Path to save the plot
        """
        if "pr_curve" not in self.results:
            self.logger.warning(
                "No evaluation results available. Please run evaluate() first."
            )
            return

        precision = self.results["pr_curve"]["precision"]
        recall = self.results["pr_curve"]["recall"]
        avg_precision = self.results["average_precision"]

        plt.figure(figsize=self.config.visualization.figure_size)
        plt.plot(
            recall,
            precision,
            color="blue",
            lw=2,
            label=f"Precision-Recall curve (AP = {avg_precision:.4f})",
        )
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curve")
        plt.legend(loc="lower left")

        if save_path:
            plt.savefig(
                save_path, dpi=self.config.visualization.dpi, bbox_inches="tight"
            )
            self.logger.info(f"Precision-Recall curve plot saved to {save_path}")
        else:
            save_path = os.path.join(self.config.evaluation.results_dir, "pr_curve.png")
            plt.savefig(
                save_path, dpi=self.config.visualization.dpi, bbox_inches="tight"
            )
            self.logger.info(f"Precision-Recall curve plot saved to {save_path}")

        plt.show()

    def print_classification_report(self):
        """
        Print the classification report
        """
        if "classification_report" not in self.results:
            self.logger.warning(
                "No evaluation results available. Please run evaluate() first."
            )
            return

        print("\nClassification Report:")
        print(
            classification_report(
                self.results["y_true"],
                self.results["y_pred"],
                target_names=["Control", "Schizophrenia"],
            )
        )

        # Print key metrics
        print("\nKey Metrics:")
        print(f"  Accuracy (ACC): {self.results['accuracy']:.4f}")
        print(f"  Sensitivity (SEN): {self.results['sensitivity']:.4f}")
        print(f"  Specificity (SPEC): {self.results['specificity']:.4f}")
        print(f"  Precision: {self.results['precision']:.4f}")
        print(f"  F1 Score: {self.results['f1_score']:.4f}")
        print(f"  AUC: {self.results['auc']:.4f}")
        print(f"  Average Precision: {self.results['average_precision']:.4f}")
        print(f"  Optimal Threshold: {self.results['optimal_threshold']:.4f}")

    def save_results(self, path=None):
        """
        Save evaluation results to a file

        Args:
            path (str): Path to save the results
        """
        import json

        if path is None:
            path = os.path.join(
                self.config.evaluation.results_dir, "evaluation_results.json"
            )

        # Create a copy of results for JSON serialization
        results_copy = {}
        for key, value in self.results.items():
            if key in ["y_true", "y_pred", "y_pred_proba"]:
                # Convert numpy arrays to lists
                results_copy[key] = value.tolist()
            elif key in ["confusion_matrix", "roc_curve", "pr_curve"]:
                # Convert nested numpy arrays to lists
                if isinstance(value, dict):
                    results_copy[key] = {
                        k: v.tolist() if hasattr(v, "tolist") else v
                        for k, v in value.items()
                    }
                else:
                    results_copy[key] = (
                        value.tolist() if hasattr(value, "tolist") else value
                    )
            elif isinstance(value, np.ndarray):
                results_copy[key] = value.tolist()
            elif isinstance(value, dict):
                results_copy[key] = value
            else:
                results_copy[key] = (
                    float(value)
                    if isinstance(value, (np.float32, np.float64))
                    else value
                )

        with open(path, "w") as f:
            json.dump(results_copy, f, indent=2)

        self.logger.info(f"Evaluation results saved to {path}")

    def evaluate_per_class(self):
        """
        Calculate per-class performance metrics

        Returns:
            Dictionary of per-class metrics
        """
        if "classification_report" not in self.results:
            self.logger.warning(
                "No evaluation results available. Please run evaluate() first."
            )
            return None

        # Extract per-class metrics from classification report
        report = self.results["classification_report"]
        per_class_metrics = {}

        for class_name in ["Control", "Schizophrenia"]:
            if class_name in report:
                per_class_metrics[class_name] = {
                    "precision": report[class_name]["precision"],
                    "recall": report[class_name]["recall"],
                    "f1-score": report[class_name]["f1-score"],
                    "support": report[class_name]["support"],
                }

        return per_class_metrics

    def batch_evaluate(self, test_data, batch_size=None):
        """
        Evaluate the model on test data in batches for memory efficiency

        Args:
            test_data: Test data generator or dataset
            batch_size: Batch size for evaluation

        Returns:
            Dictionary of evaluation metrics
        """
        self.logger.info("Starting batch evaluation...")

        # Set batch size
        if batch_size is None:
            batch_size = self.config.data.batch_size

        # Initialize lists to store predictions
        y_true = []
        y_pred = []
        y_pred_proba = []

        # Process data in batches
        batch_count = 0
        for batch_x, batch_y in test_data:
            batch_count += 1
            self.logger.debug(f"Processing batch {batch_count}")

            batch_pred = self.model.predict(batch_x, verbose=0)

            # Handle different label formats
            if len(batch_y.shape) > 1 and batch_y.shape[1] > 1:
                # One-hot encoded labels
                y_true.extend(np.argmax(batch_y, axis=1))
            else:
                # Binary labels
                y_true.extend(batch_y.numpy().flatten())

            # Get predicted classes and probabilities
            y_pred.extend(np.argmax(batch_pred, axis=1))
            y_pred_proba.extend(
                batch_pred[:, 1].numpy()
            )  # Probability of positive class

        # Convert to numpy arrays
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        y_pred_proba = np.array(y_pred_proba)

        self.logger.info(f"Evaluated {len(y_true)} samples in {batch_count} batches")

        # Calculate metrics (same as in evaluate method)
        self.results["accuracy"] = accuracy_score(y_true, y_pred)
        self.results["precision"] = precision_score(y_true, y_pred, zero_division=0)
        self.results["recall"] = recall_score(y_true, y_pred, zero_division=0)
        self.results["f1_score"] = f1_score(y_true, y_pred, zero_division=0)
        self.results["sensitivity"] = recall_score(y_true, y_pred, zero_division=0)

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        self.results["specificity"] = tn / (tn + fp) if (tn + fp) > 0 else 0

        self.results["confusion_matrix"] = confusion_matrix(y_true, y_pred)
        self.results["classification_report"] = classification_report(
            y_true, y_pred, target_names=["Control", "Schizophrenia"], output_dict=True
        )

        fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
        self.results["roc_curve"] = {"fpr": fpr, "tpr": tpr, "thresholds": thresholds}
        self.results["auc"] = auc(fpr, tpr)

        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
        self.results["pr_curve"] = {"precision": precision, "recall": recall}
        self.results["average_precision"] = average_precision_score(
            y_true, y_pred_proba
        )

        youden_j = tpr - fpr
        optimal_idx = np.argmax(youden_j)
        self.results["optimal_threshold"] = thresholds[optimal_idx]
        self.results["optimal_sensitivity"] = tpr[optimal_idx]
        self.results["optimal_specificity"] = 1 - fpr[optimal_idx]

        self.results["y_true"] = y_true
        self.results["y_pred"] = y_pred
        self.results["y_pred_proba"] = y_pred_proba

        # Log results
        self.logger.info(f"Batch Evaluation Results:")
        self.logger.info(f"  Accuracy (ACC): {self.results['accuracy']:.4f}")
        self.logger.info(f"  Sensitivity (SEN): {self.results['sensitivity']:.4f}")
        self.logger.info(f"  Specificity (SPEC): {self.results['specificity']:.4f}")
        self.logger.info(f"  AUC: {self.results['auc']:.4f}")

        return self.results


def evaluate_model(model, test_data, config=None, batch_size=None):
    """
    Convenience function to evaluate a model

    Args:
        model: Trained model to evaluate
        test_data: Test data
        config: Configuration object
        batch_size: Batch size for evaluation

    Returns:
        Dictionary of evaluation metrics
    """
    evaluator = ModelEvaluator(model, config)
    results = evaluator.evaluate(test_data, batch_size)

    return results, evaluator
