"""
Interactive visualization utilities using Plotly
"""

import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from plotly.offline import plot
import plotly.figure_factory as ff
from pathlib import Path
import os
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
from ..config import default_config
from ..utils.file_utils import ensure_dir


class InteractivePlotter:
    """
    Class for creating interactive visualizations
    """

    def __init__(self, config=None):
        """
        Initialize the interactive plotter

        Args:
            config: Configuration object
        """
        self.config = config if config is not None else default_config

        # Set default color scheme
        self.color_scheme = px.colors.qualitative.Set1

    def plot_interactive_confusion_matrix(
        self, y_true, y_pred, labels=None, normalize=False, save_path=None
    ):
        """
        Create an interactive confusion matrix with hover tooltips

        Args:
            y_true: True labels
            y_pred: Predicted labels
            labels: Class labels
            normalize: Whether to normalize the confusion matrix
            save_path: Path to save the HTML file

        Returns:
            Plotly figure object
        """
        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred)

        if normalize:
            cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
            title = "Normalized Confusion Matrix"
            hovertemplate = (
                "True: %{y}<br>Predicted: %{x}<br>Value: %{z:.2f}<extra></extra>"
            )
        else:
            title = "Confusion Matrix"
            hovertemplate = (
                "True: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>"
            )

        # Set default labels if not provided
        if labels is None:
            labels = ["Control", "Schizophrenia"]

        # Create heatmap
        fig = go.Figure(
            data=go.Heatmap(
                z=cm,
                x=labels,
                y=labels,
                colorscale="Blues",
                showscale=True,
                text=cm,
                texttemplate="%{text}",
                textfont={"size": 14},
                hovertemplate=hovertemplate,
            )
        )

        # Update layout
        fig.update_layout(
            title=title,
            xaxis_title="Predicted Label",
            yaxis_title="True Label",
            width=600,
            height=500,
        )

        # Save as HTML if path provided
        if save_path:
            ensure_dir(os.path.dirname(save_path))
            plot(fig, filename=save_path, auto_open=False)

        return fig

    def plot_interactive_roc_curve(self, y_true, y_score, save_path=None):
        """
        Create an interactive ROC curve with hover tooltips

        Args:
            y_true: True labels
            y_score: Predicted probabilities or scores
            save_path: Path to save the HTML file

        Returns:
            Plotly figure object
        """
        # Compute ROC curve
        fpr, tpr, thresholds = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)

        # Create the plot
        fig = go.Figure()

        # Add ROC curve
        fig.add_trace(
            go.Scatter(
                x=fpr,
                y=tpr,
                mode="lines",
                name=f"ROC Curve (AUC = {roc_auc:.4f})",
                line=dict(color="darkorange", width=2),
                hovertemplate="False Positive Rate: %{x:.3f}<br>True Positive Rate: %{y:.3f}<br>Threshold: %{text:.3f}<extra></extra>",
                text=thresholds,
            )
        )

        # Add diagonal line
        fig.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode="lines",
                name="Random Classifier",
                line=dict(color="navy", width=2, dash="dash"),
                hovertemplate="False Positive Rate: %{x:.3f}<br>True Positive Rate: %{y:.3f}<extra></extra>",
            )
        )

        # Update layout
        fig.update_layout(
            title="Receiver Operating Characteristic (ROC) Curve",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            xaxis=dict(range=[0, 1]),
            yaxis=dict(range=[0, 1.05]),
            width=700,
            height=500,
        )

        # Save as HTML if path provided
        if save_path:
            ensure_dir(os.path.dirname(save_path))
            plot(fig, filename=save_path, auto_open=False)

        return fig

    def plot_interactive_pr_curve(self, y_true, y_score, save_path=None):
        """
        Create an interactive Precision-Recall curve with hover tooltips

        Args:
            y_true: True labels
            y_score: Predicted probabilities or scores
            save_path: Path to save the HTML file

        Returns:
            Plotly figure object
        """
        # Compute Precision-Recall curve
        precision, recall, thresholds = precision_recall_curve(y_true, y_score)
        pr_auc = auc(recall, precision)
        average_precision = np.mean(precision)

        # Create the plot
        fig = go.Figure()

        # Add PR curve
        fig.add_trace(
            go.Scatter(
                x=recall,
                y=precision,
                mode="lines",
                name=f"PR Curve (AUC = {pr_auc:.4f}, AP = {average_precision:.4f})",
                line=dict(color="blue", width=2),
                hovertemplate="Recall: %{x:.3f}<br>Precision: %{y:.3f}<extra></extra>",
            )
        )

        # Update layout
        fig.update_layout(
            title="Precision-Recall Curve",
            xaxis_title="Recall",
            yaxis_title="Precision",
            xaxis=dict(range=[0, 1]),
            yaxis=dict(range=[0, 1.05]),
            width=700,
            height=500,
        )

        # Save as HTML if path provided
        if save_path:
            ensure_dir(os.path.dirname(save_path))
            plot(fig, filename=save_path, auto_open=False)

        return fig

    def plot_interactive_volume_slicer(self, volume_data, title=None, save_path=None):
        """
        Create an interactive 3D volume slicer with x, y, z sliders

        Args:
            volume_data: 3D volume data
            title: Title for the plot
            save_path: Path to save the HTML file

        Returns:
            Plotly figure object
        """
        # Ensure 3D data
        if len(volume_data.shape) > 3:
            volume_data = np.squeeze(volume_data)

        # Get volume dimensions
        x_dim, y_dim, z_dim = volume_data.shape

        # Create initial slices (middle slices)
        x_idx = x_dim // 2
        y_idx = y_dim // 2
        z_idx = z_dim // 2

        # Create figure with subplots
        fig = make_subplots(
            rows=2,
            cols=2,
            specs=[
                [{"type": "xy"}, {"type": "xy"}],
                [{"type": "xy"}, {"type": "scene"}],
            ],
            subplot_titles=["Sagittal (X)", "Coronal (Y)", "Axial (Z)", "3D Volume"],
            horizontal_spacing=0.1,
            vertical_spacing=0.1,
        )

        # Add initial slices
        # Sagittal view
        fig.add_trace(
            go.Heatmap(
                z=volume_data[x_idx, :, :],
                colorscale="Viridis",
                showscale=False,
                name="Sagittal",
            ),
            row=1,
            col=1,
        )

        # Coronal view
        fig.add_trace(
            go.Heatmap(
                z=volume_data[:, y_idx, :],
                colorscale="Viridis",
                showscale=False,
                name="Coronal",
            ),
            row=1,
            col=2,
        )

        # Axial view
        fig.add_trace(
            go.Heatmap(
                z=volume_data[:, :, z_idx],
                colorscale="Viridis",
                showscale=True,
                name="Axial",
            ),
            row=2,
            col=1,
        )

        # Add 3D volume (isosurface)
        # Create mesh grid for 3D plot
        x, y, z = np.mgrid[0:x_dim, 0:y_dim, 0:z_dim]

        # Only show a subset of points for performance
        step = max(1, min(x_dim, y_dim, z_dim) // 20)
        x_sub = x[::step, ::step, ::step].flatten()
        y_sub = y[::step, ::step, ::step].flatten()
        z_sub = z[::step, ::step, ::step].flatten()
        values_sub = volume_data[::step, ::step, ::step].flatten()

        # Filter out very low values for better visualization
        threshold = np.percentile(volume_data, 80)
        mask = values_sub > threshold

        fig.add_trace(
            go.Scatter3d(
                x=x_sub[mask],
                y=y_sub[mask],
                z=z_sub[mask],
                mode="markers",
                marker=dict(
                    size=2,
                    color=values_sub[mask],
                    colorscale="Viridis",
                    showscale=False,
                    opacity=0.6,
                ),
                name="3D Volume",
            ),
            row=2,
            col=2,
        )

        # Update layout
        fig.update_layout(
            title=title if title else "Interactive 3D Volume Slicer",
            width=1000,
            height=800,
            sliders=[
                {
                    "active": x_idx,
                    "currentvalue": {"prefix": "X-Slice: "},
                    "pad": {"t": 50},
                    "steps": [
                        {
                            "args": [
                                {
                                    "x": [volume_data[i, :, :]],
                                    "xaxis": "x1",
                                    "yaxis": "y1",
                                }
                            ],
                            "label": str(i),
                            "method": "restyle",
                        }
                        for i in range(0, x_dim, max(1, x_dim // 50))
                    ],
                },
                {
                    "active": y_idx,
                    "currentvalue": {"prefix": "Y-Slice: "},
                    "pad": {"t": 50},
                    "steps": [
                        {
                            "args": [
                                {
                                    "x": [volume_data[:, i, :]],
                                    "xaxis": "x2",
                                    "yaxis": "y2",
                                }
                            ],
                            "label": str(i),
                            "method": "restyle",
                        }
                        for i in range(0, y_dim, max(1, y_dim // 50))
                    ],
                },
                {
                    "active": z_idx,
                    "currentvalue": {"prefix": "Z-Slice: "},
                    "pad": {"t": 50},
                    "steps": [
                        {
                            "args": [
                                {
                                    "x": [volume_data[:, :, i]],
                                    "xaxis": "x3",
                                    "yaxis": "y3",
                                }
                            ],
                            "label": str(i),
                            "method": "restyle",
                        }
                        for i in range(0, z_dim, max(1, z_dim // 50))
                    ],
                },
            ],
        )

        # Update axes labels
        fig.update_xaxes(title_text="Y", row=1, col=1)
        fig.update_yaxes(title_text="Z", row=1, col=1)
        fig.update_xaxes(title_text="X", row=1, col=2)
        fig.update_yaxes(title_text="Z", row=1, col=2)
        fig.update_xaxes(title_text="X", row=2, col=1)
        fig.update_yaxes(title_text="Y", row=2, col=1)

        # Update 3D scene
        fig.update_scenes(
            xaxis_title="X", yaxis_title="Y", zaxis_title="Z", row=2, col=2
        )

        # Save as HTML if path provided
        if save_path:
            ensure_dir(os.path.dirname(save_path))
            plot(fig, filename=save_path, auto_open=False)

        return fig

    def plot_interactive_fnc_matrix(
        self, fnc_matrix, labels=None, title=None, save_path=None
    ):
        """
        Create an interactive FNC matrix explorer with clustering toggle

        Args:
            fnc_matrix: Functional connectivity matrix
            labels: Region labels
            title: Title for the plot
            save_path: Path to save the HTML file

        Returns:
            Plotly figure object
        """
        # Create heatmap
        fig = go.Figure(
            data=go.Heatmap(
                z=fnc_matrix,
                x=labels if labels else list(range(len(fnc_matrix))),
                y=labels if labels else list(range(len(fnc_matrix))),
                colorscale="RdBu",
                zmid=0,
                zmin=-1,
                zmax=1,
                hovertemplate="Region 1: %{y}<br>Region 2: %{x}<br>Connectivity: %{z:.3f}<extra></extra>",
            )
        )

        # Update layout
        fig.update_layout(
            title=title if title else "Functional Network Connectivity Matrix",
            xaxis_title="Brain Regions",
            yaxis_title="Brain Regions",
            width=800,
            height=700,
        )

        # Add clustering toggle button
        fig.update_layout(
            updatemenus=[
                {
                    "buttons": [
                        {
                            "args": [
                                {
                                    "x": (
                                        labels
                                        if labels
                                        else list(range(len(fnc_matrix)))
                                    ),
                                    "y": (
                                        labels
                                        if labels
                                        else list(range(len(fnc_matrix)))
                                    ),
                                    "z": [fnc_matrix],
                                }
                            ],
                            "label": "Original",
                            "method": "restyle",
                        },
                        {
                            "args": [
                                {
                                    "x": (
                                        labels
                                        if labels
                                        else list(range(len(fnc_matrix)))
                                    ),
                                    "y": (
                                        labels
                                        if labels
                                        else list(range(len(fnc_matrix)))
                                    ),
                                    "z": [self._cluster_matrix(fnc_matrix)],
                                }
                            ],
                            "label": "Clustered",
                            "method": "restyle",
                        },
                    ],
                    "direction": "left",
                    "pad": {"r": 10, "t": 10},
                    "showactive": True,
                    "type": "buttons",
                    "x": 0.01,
                    "xanchor": "left",
                    "y": 1.02,
                    "yanchor": "top",
                }
            ]
        )

        # Save as HTML if path provided
        if save_path:
            ensure_dir(os.path.dirname(save_path))
            plot(fig, filename=save_path, auto_open=False)

        return fig

    def _cluster_matrix(self, matrix):
        """
        Apply hierarchical clustering to a matrix

        Args:
            matrix: Input matrix

        Returns:
            Clustered matrix
        """
        try:
            from scipy.cluster.hierarchy import linkage, leaves_list
            from scipy.spatial.distance import pdist

            # Compute distance matrix
            dist_matrix = pdist(matrix)

            # Perform hierarchical clustering
            linkage_matrix = linkage(dist_matrix, method="average")
            order = leaves_list(linkage_matrix)

            # Reorder matrix
            clustered_matrix = matrix[order, :][:, order]

            return clustered_matrix
        except ImportError:
            print("SciPy not available for clustering. Returning original matrix.")
            return matrix

    def plot_interactive_fnc_comparison(
        self,
        fnc_group1,
        fnc_group2,
        labels=None,
        group_names=None,
        title=None,
        save_path=None,
    ):
        """
        Create an interactive FNC matrix explorer with difference map slider

        Args:
            fnc_group1: FNC matrix for group 1
            fnc_group2: FNC matrix for group 2
            labels: Region labels
            group_names: Names of the groups
            title: Title for the plot
            save_path: Path to save the HTML file

        Returns:
            Plotly figure object
        """
        if group_names is None:
            group_names = ["Group 1", "Group 2"]

        # Calculate difference
        diff_matrix = fnc_group1 - fnc_group2

        # Create figure with subplots
        fig = make_subplots(
            rows=1,
            cols=3,
            subplot_titles=[f"{group_names[0]}", f"{group_names[1]}", "Difference"],
            specs=[[{"type": "xy"}, {"type": "xy"}, {"type": "xy"}]],
        )

        # Add group 1 heatmap
        fig.add_trace(
            go.Heatmap(
                z=fnc_group1,
                x=labels if labels else list(range(len(fnc_group1))),
                y=labels if labels else list(range(len(fnc_group1))),
                colorscale="RdBu",
                zmid=0,
                zmin=-1,
                zmax=1,
                showscale=False,
                name=group_names[0],
                hovertemplate="Region 1: %{y}<br>Region 2: %{x}<br>Connectivity: %{z:.3f}<extra></extra>",
            ),
            row=1,
            col=1,
        )

        # Add group 2 heatmap
        fig.add_trace(
            go.Heatmap(
                z=fnc_group2,
                x=labels if labels else list(range(len(fnc_group2))),
                y=labels if labels else list(range(len(fnc_group2))),
                colorscale="RdBu",
                zmid=0,
                zmin=-1,
                zmax=1,
                showscale=False,
                name=group_names[1],
                hovertemplate="Region 1: %{y}<br>Region 2: %{x}<br>Connectivity: %{z:.3f}<extra></extra>",
            ),
            row=1,
            col=2,
        )

        # Add difference heatmap
        max_diff = np.max(np.abs(diff_matrix))
        fig.add_trace(
            go.Heatmap(
                z=diff_matrix,
                x=labels if labels else list(range(len(diff_matrix))),
                y=labels if labels else list(range(len(diff_matrix))),
                colorscale="RdBu",
                zmid=0,
                zmin=-max_diff,
                zmax=max_diff,
                showscale=True,
                name="Difference",
                hovertemplate="Region 1: %{y}<br>Region 2: %{x}<br>Difference: %{z:.3f}<extra></extra>",
            ),
            row=1,
            col=3,
        )

        # Update layout
        fig.update_layout(
            title=title if title else "FNC Group Comparison", width=1500, height=500
        )

        # Save as HTML if path provided
        if save_path:
            ensure_dir(os.path.dirname(save_path))
            plot(fig, filename=save_path, auto_open=False)

        return fig

    def plot_interactive_training_history(
        self, history_dict, metrics=None, save_path=None
    ):
        """
        Create an interactive training history plot

        Args:
            history_dict: Dictionary containing training history
            metrics: List of metrics to plot
            save_path: Path to save the HTML file

        Returns:
            Plotly figure object
        """
        if metrics is None:
            metrics = ["loss", "accuracy"]

        # Create subplots
        fig = make_subplots(
            rows=1,
            cols=len(metrics),
            subplot_titles=[m.replace("_", " ").title() for m in metrics],
        )

        # Plot each metric
        for i, metric in enumerate(metrics):
            # Training data
            if metric in history_dict:
                fig.add_trace(
                    go.Scatter(
                        x=list(range(len(history_dict[metric]))),
                        y=history_dict[metric],
                        mode="lines",
                        name=f"Training {metric}",
                        line=dict(color="blue"),
                        hovertemplate=f"Epoch: %{{x}}<br>{metric}: %{{y:.4f}}<extra></extra>",
                    ),
                    row=1,
                    col=i + 1,
                )

            # Validation data if available
            if f"val_{metric}" in history_dict:
                fig.add_trace(
                    go.Scatter(
                        x=list(range(len(history_dict[f"val_{metric}"]))),
                        y=history_dict[f"val_{metric}"],
                        mode="lines",
                        name=f"Validation {metric}",
                        line=dict(color="red"),
                        hovertemplate=f"Epoch: %{{x}}<br>Val {metric}: %{{y:.4f}}<extra></extra>",
                    ),
                    row=1,
                    col=i + 1,
                )

            # Update axis labels
            fig.update_xaxes(title_text="Epoch", row=1, col=i + 1)
            fig.update_yaxes(
                title_text=metric.replace("_", " ").title(), row=1, col=i + 1
            )

        # Update layout
        fig.update_layout(
            title="Training History", width=300 * len(metrics), height=400
        )

        # Save as HTML if path provided
        if save_path:
            ensure_dir(os.path.dirname(save_path))
            plot(fig, filename=save_path, auto_open=False)

        return fig

    def generate_interactive_dashboard(
        self,
        evaluator_results=None,
        volume_data=None,
        fnc_matrices=None,
        training_history=None,
        output_dir=None,
        prefix="dashboard",
    ):
        """
        Generate a comprehensive interactive dashboard

        Args:
            evaluator_results: Results from ModelEvaluator
            volume_data: 3D volume data for visualization
            fnc_matrices: Dictionary with FNC matrices for different groups
            training_history: Training history dictionary
            output_dir: Output directory for HTML files
            prefix: Prefix for output files

        Returns:
            Dictionary with paths to generated HTML files
        """
        if output_dir is None:
            output_dir = os.path.join(
                self.config.visualization.output_dir, "figures", "interactive"
            )

        ensure_dir(output_dir)

        html_paths = {}

        # 1. Confusion matrix
        if evaluator_results is not None:
            cm_path = os.path.join(output_dir, f"{prefix}_confusion_matrix.html")
            self.plot_interactive_confusion_matrix(
                evaluator_results["y_true"],
                evaluator_results["y_pred"],
                save_path=cm_path,
            )
            html_paths["confusion_matrix"] = cm_path

            # ROC curve
            roc_path = os.path.join(output_dir, f"{prefix}_roc_curve.html")
            self.plot_interactive_roc_curve(
                evaluator_results["y_true"],
                evaluator_results["y_pred_proba"],
                save_path=roc_path,
            )
            html_paths["roc_curve"] = roc_path

            # PR curve
            pr_path = os.path.join(output_dir, f"{prefix}_pr_curve.html")
            self.plot_interactive_pr_curve(
                evaluator_results["y_true"],
                evaluator_results["y_pred_proba"],
                save_path=pr_path,
            )
            html_paths["pr_curve"] = pr_path

        # 2. Volume slicer
        if volume_data is not None:
            volume_path = os.path.join(output_dir, f"{prefix}_volume_slicer.html")
            self.plot_interactive_volume_slicer(
                volume_data, title="Interactive Volume Slicer", save_path=volume_path
            )
            html_paths["volume_slicer"] = volume_path

        # 3. FNC matrices
        if fnc_matrices is not None:
            for group_name, fnc_matrix in fnc_matrices.items():
                fnc_path = os.path.join(output_dir, f"{prefix}_fnc_{group_name}.html")
                self.plot_interactive_fnc_matrix(
                    fnc_matrix, title=f"FNC Matrix - {group_name}", save_path=fnc_path
                )
                html_paths[f"fnc_{group_name}"] = fnc_path

            # Group comparison if multiple groups
            if len(fnc_matrices) > 1:
                group_names = list(fnc_matrices.keys())
                if len(group_names) == 2:
                    fnc_comp_path = os.path.join(
                        output_dir, f"{prefix}_fnc_comparison.html"
                    )
                    self.plot_interactive_fnc_comparison(
                        fnc_matrices[group_names[0]],
                        fnc_matrices[group_names[1]],
                        group_names=group_names,
                        title="FNC Group Comparison",
                        save_path=fnc_comp_path,
                    )
                    html_paths["fnc_comparison"] = fnc_comp_path

        # 4. Training history
        if training_history is not None:
            training_path = os.path.join(output_dir, f"{prefix}_training_history.html")
            self.plot_interactive_training_history(
                training_history, save_path=training_path
            )
            html_paths["training_history"] = training_path

        return html_paths


# Convenience functions
def plot_interactive_confusion_matrix(
    y_true, y_pred, labels=None, normalize=False, save_path=None
):
    """
    Convenience function to create interactive confusion matrix

    Args:
        y_true: True labels
        y_pred: Predicted labels
        labels: Class labels
        normalize: Whether to normalize the confusion matrix
        save_path: Path to save the HTML file

    Returns:
        Plotly figure object
    """
    plotter = InteractivePlotter()
    return plotter.plot_interactive_confusion_matrix(
        y_true, y_pred, labels, normalize, save_path
    )


def plot_interactive_roc_curve(y_true, y_score, save_path=None):
    """
    Convenience function to create interactive ROC curve

    Args:
        y_true: True labels
        y_score: Predicted probabilities or scores
        save_path: Path to save the HTML file

    Returns:
        Plotly figure object
    """
    plotter = InteractivePlotter()
    return plotter.plot_interactive_roc_curve(y_true, y_score, save_path)


def plot_interactive_volume_slicer(volume_data, title=None, save_path=None):
    """
    Convenience function to create interactive volume slicer

    Args:
        volume_data: 3D volume data
        title: Title for the plot
        save_path: Path to save the HTML file

    Returns:
        Plotly figure object
    """
    plotter = InteractivePlotter()
    return plotter.plot_interactive_volume_slicer(volume_data, title, save_path)


def plot_interactive_fnc_matrix(fnc_matrix, labels=None, title=None, save_path=None):
    """
    Convenience function to create interactive FNC matrix

    Args:
        fnc_matrix: Functional connectivity matrix
        labels: Region labels
        title: Title for the plot
        save_path: Path to save the HTML file

    Returns:
        Plotly figure object
    """
    plotter = InteractivePlotter()
    return plotter.plot_interactive_fnc_matrix(fnc_matrix, labels, title, save_path)


def generate_interactive_dashboard(
    evaluator_results=None,
    volume_data=None,
    fnc_matrices=None,
    training_history=None,
    output_dir=None,
    prefix="dashboard",
):
    """
    Convenience function to generate interactive dashboard

    Args:
        evaluator_results: Results from ModelEvaluator
        volume_data: 3D volume data for visualization
        fnc_matrices: Dictionary with FNC matrices for different groups
        training_history: Training history dictionary
        output_dir: Output directory for HTML files
        prefix: Prefix for output files

    Returns:
        Dictionary with paths to generated HTML files
    """
    plotter = InteractivePlotter()
    return plotter.generate_interactive_dashboard(
        evaluator_results,
        volume_data,
        fnc_matrices,
        training_history,
        output_dir,
        prefix,
    )
