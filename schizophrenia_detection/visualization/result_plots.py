"""
Result plotting utilities for model evaluation
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
from pathlib import Path
import os
from ..config import default_config
from ..utils.file_utils import ensure_dir


class ResultPlotter:
    """
    Class for plotting model evaluation results
    """

    def __init__(self, config=None):
        """
        Initialize the result plotter

        Args:
            config: Configuration object
        """
        self.config = config if config is not None else default_config

        # Set up matplotlib style
        plt.style.use("seaborn-v0_8")
        sns.set_palette("husl")

        # Set default font sizes
        self.title_fontsize = 14
        self.label_fontsize = 12
        self.tick_fontsize = 10
        self.legend_fontsize = 11

    def plot_training_curves(
        self,
        history_dict,
        metrics=None,
        smoothing=None,
        confidence_bands=None,
        save_path=None,
        ax=None,
    ):
        """
        Plot training vs. validation curves for loss, accuracy, sensitivity, specificity, and AUC

        Args:
            history_dict: Dictionary containing training history or list of dictionaries for multiple runs
            metrics: List of metrics to plot (default: ['loss', 'accuracy'])
            smoothing: Window size for smoothing (None for no smoothing)
            confidence_bands: List of confidence bands for cross-validation (optional)
            save_path: Path to save the plot
            ax: Matplotlib axis to plot on (creates new figure if None)

        Returns:
            Matplotlib axis object
        """
        if metrics is None:
            metrics = ["loss", "accuracy"]

        # Handle multiple runs
        if isinstance(history_dict, list):
            multiple_runs = True
            histories = history_dict
        else:
            multiple_runs = False
            histories = [history_dict]

        # Create figure if no axis provided
        if ax is None:
            fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 5))
            if len(metrics) == 1:
                axes = [axes]
        else:
            axes = [ax] if len(metrics) == 1 else ax

        # Colors for different runs
        colors = plt.cm.tab10(np.linspace(0, 1, len(histories)))

        for metric_idx, metric in enumerate(metrics):
            ax = axes[metric_idx]

            # Plot each run
            for run_idx, history in enumerate(histories):
                # Get training and validation values
                train_values = history.get(metric, [])
                val_values = history.get(f"val_{metric}", [])

                # Apply smoothing if requested
                if smoothing is not None and len(train_values) > smoothing:
                    train_values = self._smooth_curve(train_values, smoothing)
                if smoothing is not None and len(val_values) > smoothing:
                    val_values = self._smooth_curve(val_values, smoothing)

                # Plot training curve
                if train_values:
                    ax.plot(
                        range(len(train_values)),
                        train_values,
                        color=colors[run_idx],
                        linestyle="-",
                        linewidth=2,
                        label=(
                            f"Train {metric}"
                            if not multiple_runs
                            else f"Train {metric} (Run {run_idx+1})"
                        ),
                    )

                # Plot validation curve
                if val_values:
                    ax.plot(
                        range(len(val_values)),
                        val_values,
                        color=colors[run_idx],
                        linestyle="--",
                        linewidth=2,
                        label=(
                            f"Val {metric}"
                            if not multiple_runs
                            else f"Val {metric} (Run {run_idx+1})"
                        ),
                    )

            # Add confidence bands if provided
            if confidence_bands is not None and metric_idx < len(confidence_bands):
                if confidence_bands[metric_idx] is not None:
                    mean_values, std_values = confidence_bands[metric_idx]
                    epochs = range(len(mean_values))
                    ax.fill_between(
                        epochs,
                        mean_values - std_values,
                        mean_values + std_values,
                        alpha=0.2,
                        color="gray",
                        label="Confidence Band",
                    )

            # Set labels and title
            ax.set_title(
                f'{metric.replace("_", " ").title()} Curve',
                fontsize=self.title_fontsize,
            )
            ax.set_xlabel("Epoch", fontsize=self.label_fontsize)
            ax.set_ylabel(
                metric.replace("_", " ").title(), fontsize=self.label_fontsize
            )
            ax.tick_params(axis="both", which="major", labelsize=self.tick_fontsize)

            # Add legend
            ax.legend(fontsize=self.legend_fontsize)

            # Add grid
            ax.grid(True, alpha=0.3)

        # Adjust layout
        plt.tight_layout()

        # Save figure if path provided
        if save_path:
            ensure_dir(os.path.dirname(save_path))
            plt.savefig(
                save_path,
                dpi=self.config.visualization.dpi,
                bbox_inches="tight",
                format="png",
            )
            # Also save as PDF
            pdf_path = save_path.replace(".png", ".pdf")
            plt.savefig(
                pdf_path,
                dpi=self.config.visualization.dpi,
                bbox_inches="tight",
                format="pdf",
            )

        return axes

    def _smooth_curve(self, values, window_size):
        """
        Apply moving average smoothing to a curve

        Args:
            values: List of values to smooth
            window_size: Size of the moving window

        Returns:
            List of smoothed values
        """
        if len(values) < window_size:
            return values

        smoothed = []
        for i in range(len(values)):
            start = max(0, i - window_size // 2)
            end = min(len(values), i + window_size // 2 + 1)
            smoothed.append(np.mean(values[start:end]))

        return smoothed

    def plot_confusion_matrix(
        self, y_true, y_pred, labels=None, normalize=False, save_path=None, ax=None
    ):
        """
        Plot confusion matrix (absolute and normalized)

        Args:
            y_true: True labels
            y_pred: Predicted labels
            labels: Class labels
            normalize: Whether to normalize the confusion matrix
            save_path: Path to save the plot
            ax: Matplotlib axis to plot on

        Returns:
            Matplotlib axis object
        """
        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred)

        if normalize:
            cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
            title = "Normalized Confusion Matrix"
            fmt = ".2f"
        else:
            title = "Confusion Matrix"
            fmt = "d"

        # Create figure if no axis provided
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))

        # Plot heatmap
        sns.heatmap(
            cm,
            annot=True,
            fmt=fmt,
            cmap="Blues",
            xticklabels=labels if labels else ["Control", "Schizophrenia"],
            yticklabels=labels if labels else ["Control", "Schizophrenia"],
            ax=ax,
            annot_kws={"size": 12},
        )

        # Set title and labels
        ax.set_title(title, fontsize=self.title_fontsize)
        ax.set_ylabel("True Label", fontsize=self.label_fontsize)
        ax.set_xlabel("Predicted Label", fontsize=self.label_fontsize)
        ax.tick_params(axis="both", which="major", labelsize=self.tick_fontsize)

        # Save figure if path provided
        if save_path:
            ensure_dir(os.path.dirname(save_path))
            plt.savefig(
                save_path,
                dpi=self.config.visualization.dpi,
                bbox_inches="tight",
                format="png",
            )
            # Also save as PDF
            pdf_path = save_path.replace(".png", ".pdf")
            plt.savefig(
                pdf_path,
                dpi=self.config.visualization.dpi,
                bbox_inches="tight",
                format="pdf",
            )

        return ax

    def plot_roc_curve(self, y_true, y_score, save_path=None, ax=None):
        """
        Plot ROC curve with AUC

        Args:
            y_true: True labels
            y_score: Predicted probabilities or scores
            save_path: Path to save the plot
            ax: Matplotlib axis to plot on

        Returns:
            Matplotlib axis object
        """
        # Compute ROC curve and area under the curve
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)

        # Create figure if no axis provided
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))

        # Plot ROC curve
        ax.plot(
            fpr,
            tpr,
            color="darkorange",
            lw=2,
            label=f"ROC curve (AUC = {roc_auc:.4f})",
        )
        ax.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")

        # Set limits and labels
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("False Positive Rate", fontsize=self.label_fontsize)
        ax.set_ylabel("True Positive Rate", fontsize=self.label_fontsize)
        ax.set_title(
            "Receiver Operating Characteristic (ROC) Curve",
            fontsize=self.title_fontsize,
        )
        ax.tick_params(axis="both", which="major", labelsize=self.tick_fontsize)
        ax.legend(loc="lower right", fontsize=self.legend_fontsize)
        ax.grid(True, alpha=0.3)

        # Save figure if path provided
        if save_path:
            ensure_dir(os.path.dirname(save_path))
            plt.savefig(
                save_path,
                dpi=self.config.visualization.dpi,
                bbox_inches="tight",
                format="png",
            )
            # Also save as PDF
            pdf_path = save_path.replace(".png", ".pdf")
            plt.savefig(
                pdf_path,
                dpi=self.config.visualization.dpi,
                bbox_inches="tight",
                format="pdf",
            )

        return ax

    def plot_precision_recall_curve(self, y_true, y_score, save_path=None, ax=None):
        """
        Plot Precision-Recall curve with AP

        Args:
            y_true: True labels
            y_score: Predicted probabilities or scores
            save_path: Path to save the plot
            ax: Matplotlib axis to plot on

        Returns:
            Matplotlib axis object
        """
        # Compute Precision-Recall curve and area under the curve
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        pr_auc = auc(recall, precision)
        average_precision = np.mean(precision)

        # Create figure if no axis provided
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))

        # Plot PR curve
        ax.plot(
            recall,
            precision,
            color="blue",
            lw=2,
            label=f"PR curve (AUC = {pr_auc:.4f}, AP = {average_precision:.4f})",
        )

        # Set limits and labels
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("Recall", fontsize=self.label_fontsize)
        ax.set_ylabel("Precision", fontsize=self.label_fontsize)
        ax.set_title("Precision-Recall Curve", fontsize=self.title_fontsize)
        ax.tick_params(axis="both", which="major", labelsize=self.tick_fontsize)
        ax.legend(loc="lower left", fontsize=self.legend_fontsize)
        ax.grid(True, alpha=0.3)

        # Save figure if path provided
        if save_path:
            ensure_dir(os.path.dirname(save_path))
            plt.savefig(
                save_path,
                dpi=self.config.visualization.dpi,
                bbox_inches="tight",
                format="png",
            )
            # Also save as PDF
            pdf_path = save_path.replace(".png", ".pdf")
            plt.savefig(
                pdf_path,
                dpi=self.config.visualization.dpi,
                bbox_inches="tight",
                format="pdf",
            )

        return ax

    def plot_metrics_summary(self, metrics_dict, save_path=None, ax=None):
        """
        Plot metrics summary panel matching paper style (ACC, SEN, SPEC)

        Args:
            metrics_dict: Dictionary with metrics values
            save_path: Path to save the plot
            ax: Matplotlib axis to plot on

        Returns:
            Matplotlib axis object
        """
        # Create figure if no axis provided
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))

        # Extract key metrics
        key_metrics = ["accuracy", "sensitivity", "specificity"]
        metric_names = ["Accuracy (ACC)", "Sensitivity (SEN)", "Specificity (SPEC)"]
        values = []

        for metric in key_metrics:
            if metric in metrics_dict:
                values.append(metrics_dict[metric])
            elif metric.replace("accuracy", "acc") in metrics_dict:
                values.append(metrics_dict[metric.replace("accuracy", "acc")])
            elif metric.replace("sensitivity", "sen") in metrics_dict:
                values.append(metrics_dict[metric.replace("sensitivity", "sen")])
            elif metric.replace("specificity", "spec") in metrics_dict:
                values.append(metrics_dict[metric.replace("specificity", "spec")])
            else:
                values.append(0.0)

        # Create bar plot
        bars = ax.bar(metric_names, values, color=["#1f77b4", "#ff7f0e", "#2ca02c"])

        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.01,
                f"{value:.4f}",
                ha="center",
                va="bottom",
                fontsize=self.tick_fontsize,
            )

        # Set labels and title
        ax.set_title("Model Performance Metrics", fontsize=self.title_fontsize)
        ax.set_ylabel("Metric Value", fontsize=self.label_fontsize)
        ax.set_ylim([0, 1.1])
        ax.tick_params(axis="both", which="major", labelsize=self.tick_fontsize)
        ax.grid(True, alpha=0.3, axis="y")

        # Save figure if path provided
        if save_path:
            ensure_dir(os.path.dirname(save_path))
            plt.savefig(
                save_path,
                dpi=self.config.visualization.dpi,
                bbox_inches="tight",
                format="png",
            )
            # Also save as PDF
            pdf_path = save_path.replace(".png", ".pdf")
            plt.savefig(
                pdf_path,
                dpi=self.config.visualization.dpi,
                bbox_inches="tight",
                format="pdf",
            )

        return ax

    def generate_report_figure_set(
        self, evaluator_results, output_dir=None, prefix="model_evaluation"
    ):
        """
        Generate a "report" figure set from evaluator outputs

        Args:
            evaluator_results: Results from ModelEvaluator
            output_dir: Output directory for figures
            prefix: Prefix for figure filenames

        Returns:
            Dictionary with paths to generated figures
        """
        if output_dir is None:
            output_dir = os.path.join(
                self.config.visualization.output_dir, "figures", "metrics"
            )

        ensure_dir(output_dir)

        figure_paths = {}

        # 1. Confusion matrix (absolute)
        cm_path = os.path.join(output_dir, f"{prefix}_confusion_matrix.png")
        self.plot_confusion_matrix(
            evaluator_results["y_true"], evaluator_results["y_pred"], save_path=cm_path
        )
        figure_paths["confusion_matrix"] = cm_path

        # 2. Normalized confusion matrix
        cm_norm_path = os.path.join(
            output_dir, f"{prefix}_confusion_matrix_normalized.png"
        )
        self.plot_confusion_matrix(
            evaluator_results["y_true"],
            evaluator_results["y_pred"],
            normalize=True,
            save_path=cm_norm_path,
        )
        figure_paths["confusion_matrix_normalized"] = cm_norm_path

        # 3. ROC curve
        roc_path = os.path.join(output_dir, f"{prefix}_roc_curve.png")
        self.plot_roc_curve(
            evaluator_results["y_true"],
            evaluator_results["y_pred_proba"],
            save_path=roc_path,
        )
        figure_paths["roc_curve"] = roc_path

        # 4. Precision-Recall curve
        pr_path = os.path.join(output_dir, f"{prefix}_pr_curve.png")
        self.plot_precision_recall_curve(
            evaluator_results["y_true"],
            evaluator_results["y_pred_proba"],
            save_path=pr_path,
        )
        figure_paths["pr_curve"] = pr_path

        # 5. Metrics summary
        metrics_path = os.path.join(output_dir, f"{prefix}_metrics_summary.png")
        self.plot_metrics_summary(evaluator_results, save_path=metrics_path)
        figure_paths["metrics_summary"] = metrics_path

        return figure_paths

    def plot_multiple_runs_comparison(self, results_list, metrics=None, save_path=None):
        """
        Plot comparison of multiple runs with confidence bands

        Args:
            results_list: List of result dictionaries from multiple runs
            metrics: List of metrics to compare
            save_path: Path to save the plot

        Returns:
            Matplotlib figure object
        """
        if metrics is None:
            metrics = ["accuracy", "sensitivity", "specificity", "auc"]

        # Calculate mean and std for each metric
        metric_stats = {}
        for metric in metrics:
            values = []
            for result in results_list:
                if metric in result:
                    values.append(result[metric])
                elif metric.replace("accuracy", "acc") in result:
                    values.append(result[metric.replace("accuracy", "acc")])
                elif metric.replace("sensitivity", "sen") in result:
                    values.append(result[metric.replace("sensitivity", "sen")])
                elif metric.replace("specificity", "spec") in result:
                    values.append(result[metric.replace("specificity", "spec")])

            if values:
                metric_stats[metric] = {
                    "mean": np.mean(values),
                    "std": np.std(values),
                    "values": values,
                }

        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot bar chart with error bars
        metric_names = [m.replace("_", " ").title() for m in metric_stats.keys()]
        means = [metric_stats[m]["mean"] for m in metric_stats.keys()]
        stds = [metric_stats[m]["std"] for m in metric_stats.keys()]

        bars = ax.bar(metric_names, means, yerr=stds, capsize=5, alpha=0.7)

        # Add value labels on bars
        for bar, mean, std in zip(bars, means, stds):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + std + 0.01,
                f"{mean:.4f}±{std:.4f}",
                ha="center",
                va="bottom",
                fontsize=self.tick_fontsize,
            )

        # Set labels and title
        ax.set_title(
            "Performance Metrics Across Multiple Runs", fontsize=self.title_fontsize
        )
        ax.set_ylabel("Metric Value", fontsize=self.label_fontsize)
        ax.set_ylim([0, max(means) + max(stds) + 0.1])
        ax.tick_params(axis="both", which="major", labelsize=self.tick_fontsize)
        ax.grid(True, alpha=0.3, axis="y")

        # Rotate x-axis labels if needed
        if len(max(metric_names, key=len)) > 8:
            plt.xticks(rotation=45)

        plt.tight_layout()

        # Save figure if path provided
        if save_path:
            ensure_dir(os.path.dirname(save_path))
            plt.savefig(
                save_path,
                dpi=self.config.visualization.dpi,
                bbox_inches="tight",
                format="png",
            )
            # Also save as PDF
            pdf_path = save_path.replace(".png", ".pdf")
            plt.savefig(
                pdf_path,
                dpi=self.config.visualization.dpi,
                bbox_inches="tight",
                format="pdf",
            )

        return fig


# Convenience functions
def plot_confusion_matrix(y_true, y_pred, labels=None, normalize=False, save_path=None):
    """
    Convenience function to plot confusion matrix

    Args:
        y_true: True labels
        y_pred: Predicted labels
        labels: Class labels
        normalize: Whether to normalize the confusion matrix
        save_path: Path to save the plot
    """
    plotter = ResultPlotter()
    plotter.plot_confusion_matrix(y_true, y_pred, labels, normalize, save_path)


def plot_roc_curve(y_true, y_score, save_path=None):
    """
    Convenience function to plot ROC curve

    Args:
        y_true: True labels
        y_score: Predicted probabilities or scores
        save_path: Path to save the plot
    """
    plotter = ResultPlotter()
    plotter.plot_roc_curve(y_true, y_score, save_path)


def plot_training_curves(history_dict, metrics=None, smoothing=None, save_path=None):
    """
    Convenience function to plot training curves

    Args:
        history_dict: Dictionary containing training history
        metrics: List of metrics to plot
        smoothing: Window size for smoothing
        save_path: Path to save the plot
    """
    plotter = ResultPlotter()
    plotter.plot_training_curves(history_dict, metrics, smoothing, save_path=save_path)


def generate_evaluation_report(
    evaluator_results, output_dir=None, prefix="model_evaluation"
):
    """
    Convenience function to generate evaluation report

    Args:
        evaluator_results: Results from ModelEvaluator
        output_dir: Output directory for figures
        prefix: Prefix for figure filenames

    Returns:
        Dictionary with paths to generated figures
    """
    plotter = ResultPlotter()
    return plotter.generate_report_figure_set(evaluator_results, output_dir, prefix)
