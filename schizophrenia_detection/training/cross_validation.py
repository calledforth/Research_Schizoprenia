"""
Cross-validation utilities for model evaluation
"""

import numpy as np
from sklearn.model_selection import StratifiedKFold
from ..models.sspnet_3d_cnn import create_sspnet_model
from ..training.trainer import ModelTrainer
from ..training.evaluator import ModelEvaluator
from ..config import default_config


class CrossValidator:
    """
    Class for performing cross-validation on the model
    """

    def __init__(self, config=None):
        """
        Initialize the cross-validator

        Args:
            config: Configuration object
        """
        self.config = config if config is not None else default_config
        self.cv_results = {}

    def stratified_k_fold_cv(self, data_paths, labels, n_folds=None):
        """
        Perform stratified k-fold cross-validation

        Args:
            data_paths (list): List of data file paths
            labels (list): List of corresponding labels
            n_folds (int): Number of folds for cross-validation

        Returns:
            Dictionary of cross-validation results
        """
        if n_folds is None:
            n_folds = self.config.evaluation.cv_folds

        # Initialize stratified k-fold
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

        # Initialize results storage
        fold_results = []
        all_predictions = []
        all_labels = []

        # Perform cross-validation
        for fold, (train_idx, val_idx) in enumerate(skf.split(data_paths, labels)):
            print(f"Training fold {fold + 1}/{n_folds}")

            # Split data
            train_paths = [data_paths[i] for i in train_idx]
            train_labels = [labels[i] for i in train_idx]
            val_paths = [data_paths[i] for i in val_idx]
            val_labels = [labels[i] for i in val_idx]

            # Create data generators
            train_data = self._create_data_generator(
                train_paths, train_labels, training=True
            )
            val_data = self._create_data_generator(
                val_paths, val_labels, training=False
            )

            # Train model
            trainer = ModelTrainer(self.config)
            model = trainer.build_model()
            trainer.train(train_data, val_data)

            # Evaluate model
            evaluator = ModelEvaluator(model, self.config)
            fold_result = evaluator.evaluate(val_data)
            fold_result["fold"] = fold + 1
            fold_results.append(fold_result)

            # Collect predictions for overall metrics
            fold_predictions = self._get_predictions(model, val_data)
            all_predictions.extend(fold_predictions)
            all_labels.extend(val_labels)

        # Calculate overall metrics
        overall_metrics = self._calculate_overall_metrics(all_labels, all_predictions)

        # Store results
        self.cv_results = {
            "fold_results": fold_results,
            "overall_metrics": overall_metrics,
            "mean_metrics": self._calculate_mean_metrics(fold_results),
            "std_metrics": self._calculate_std_metrics(fold_results),
        }

        return self.cv_results

    def _create_data_generator(self, data_paths, labels, training=False):
        """
        Create a data generator for training or validation

        Args:
            data_paths (list): List of data file paths
            labels (list): List of corresponding labels
            training (bool): Whether this is for training

        Returns:
            Data generator
        """
        # This would be implemented based on the actual data loading requirements
        # For now, return a placeholder
        pass

    def _get_predictions(self, model, data):
        """
        Get predictions from a model on given data

        Args:
            model: Trained model
            data: Data to predict on

        Returns:
            List of predictions
        """
        predictions = []
        for batch_x, _ in data:
            batch_pred = model.predict(batch_x)
            predictions.extend(batch_pred)
        return predictions

    def _calculate_overall_metrics(self, labels, predictions):
        """
        Calculate overall metrics from all predictions

        Args:
            labels (list): True labels
            predictions (list): Model predictions

        Returns:
            Dictionary of overall metrics
        """
        from sklearn.metrics import (
            accuracy_score,
            roc_auc_score,
            precision_recall_fscore_support,
        )

        # Convert to numpy arrays
        labels = np.array(labels)
        predictions = np.array(predictions)

        # Get predicted classes
        pred_classes = (
            np.argmax(predictions, axis=1)
            if len(predictions.shape) > 1
            else (predictions > 0.5).astype(int)
        )

        # Calculate metrics
        accuracy = accuracy_score(labels, pred_classes)

        # Calculate AUC if we have probability predictions
        if len(predictions.shape) > 1:
            auc = roc_auc_score(labels, predictions[:, 1])
        else:
            auc = roc_auc_score(labels, predictions)

        # Calculate precision, recall, F1
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, pred_classes, average="binary"
        )

        return {
            "accuracy": accuracy,
            "auc": auc,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    def _calculate_mean_metrics(self, fold_results):
        """
        Calculate mean metrics across all folds

        Args:
            fold_results (list): List of results from each fold

        Returns:
            Dictionary of mean metrics
        """
        metrics = ["accuracy", "auc", "precision", "recall", "f1"]
        mean_metrics = {}

        for metric in metrics:
            values = [fold.get(metric, 0) for fold in fold_results]
            mean_metrics[metric] = np.mean(values)

        return mean_metrics

    def _calculate_std_metrics(self, fold_results):
        """
        Calculate standard deviation of metrics across all folds

        Args:
            fold_results (list): List of results from each fold

        Returns:
            Dictionary of standard deviation of metrics
        """
        metrics = ["accuracy", "auc", "precision", "recall", "f1"]
        std_metrics = {}

        for metric in metrics:
            values = [fold.get(metric, 0) for fold in fold_results]
            std_metrics[metric] = np.std(values)

        return std_metrics

    def print_results(self):
        """
        Print cross-validation results
        """
        if not self.cv_results:
            print(
                "No cross-validation results available. Please run cross-validation first."
            )
            return

        print("Cross-Validation Results:")
        print("=" * 50)

        # Print per-fold results
        for fold_result in self.cv_results["fold_results"]:
            fold_num = fold_result["fold"]
            print(f"Fold {fold_num}:")
            for metric, value in fold_result.items():
                if metric != "fold" and isinstance(value, (int, float)):
                    print(f"  {metric}: {value:.4f}")
            print()

        # Print mean and standard deviation
        print("Mean Metrics:")
        for metric, value in self.cv_results["mean_metrics"].items():
            std = self.cv_results["std_metrics"][metric]
            print(f"  {metric}: {value:.4f} ± {std:.4f}")

        # Print overall metrics
        print("\nOverall Metrics:")
        for metric, value in self.cv_results["overall_metrics"].items():
            print(f"  {metric}: {value:.4f}")

    def save_results(self, path):
        """
        Save cross-validation results to a file

        Args:
            path (str): Path to save the results
        """
        import json

        # Convert numpy arrays to lists for JSON serialization
        results_copy = {}
        for key, value in self.cv_results.items():
            if isinstance(value, dict):
                results_copy[key] = {}
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, (np.ndarray, np.float32, np.float64)):
                        results_copy[key][sub_key] = float(sub_value)
                    else:
                        results_copy[key][sub_key] = sub_value
            else:
                results_copy[key] = value

        with open(path, "w") as f:
            json.dump(results_copy, f, indent=2)

        print(f"Cross-validation results saved to {path}")


def perform_cross_validation(data_paths, labels, n_folds=5, config=None):
    """
    Convenience function to perform cross-validation

    Args:
        data_paths (list): List of data file paths
        labels (list): List of corresponding labels
        n_folds (int): Number of folds for cross-validation
        config: Configuration object

    Returns:
        Dictionary of cross-validation results
    """
    cv = CrossValidator(config)
    results = cv.stratified_k_fold_cv(data_paths, labels, n_folds)
    return results
