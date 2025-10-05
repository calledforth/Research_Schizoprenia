"""
Model visualization utilities
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.utils import plot_model
from pathlib import Path
import os
from typing import List, Dict, Optional, Union, Tuple
from ..config import default_config
from ..utils.file_utils import ensure_dir
from ..models.sspnet_3d_cnn import SSPNet3DCNN


class ModelVisualizer:
    """
    Class for visualizing neural network models and their interpretability
    """

    def __init__(self, config=None):
        """
        Initialize the model visualizer

        Args:
            config: Configuration object
        """
        self.config = config if config is not None else default_config

        # Set up matplotlib style
        plt.style.use("seaborn-v0_8")

        # Set default font sizes
        self.title_fontsize = 14
        self.label_fontsize = 12
        self.tick_fontsize = 10
        self.legend_fontsize = 11

    def plot_model_architecture(self, model, save_path=None):
        """
        Plot the model architecture

        Args:
            model: Neural network model
            save_path: Path to save the plot

        Returns:
            Path to the saved file
        """
        if save_path is None:
            save_path = f"{self.config.visualization.output_dir}/model_architecture.png"

        # Ensure output directory exists
        ensure_dir(os.path.dirname(save_path))

        # Plot the model
        plot_model(
            model,
            to_file=save_path,
            show_shapes=True,
            show_layer_names=True,
            rankdir="TB",
            expand_nested=True,
            dpi=self.config.visualization.dpi,
        )

        print(f"Model architecture saved to {save_path}")
        return save_path

    def generate_saliency_maps_batch(
        self,
        model: SSPNet3DCNN,
        input_data_list: List[np.ndarray],
        class_idx: int = 0,
        output_dir: Optional[str] = None,
        prefix: str = "saliency",
    ) -> Dict[str, List[str]]:
        """
        Generate saliency maps for a batch of subjects

        Args:
            model: SSPNet3DCNN model
            input_data_list: List of input data arrays
            class_idx: Class index for saliency generation
            output_dir: Output directory for saving results
            prefix: Prefix for output files

        Returns:
            Dictionary with paths to generated saliency maps for each class
        """
        if output_dir is None:
            output_dir = os.path.join(
                self.config.visualization.output_dir, "figures", "saliency"
            )

        ensure_dir(output_dir)

        # Determine class directories
        class_dirs = {}
        if hasattr(input_data_list[0], "shape") and len(input_data_list[0].shape) > 1:
            # If we have labels, create class-specific directories
            class_dirs = {
                "class_0": os.path.join(output_dir, "class_0"),
                "class_1": os.path.join(output_dir, "class_1"),
            }
            for class_dir in class_dirs.values():
                ensure_dir(class_dir)

        # Generate saliency maps for each subject
        saliency_paths = {"all": []}
        for class_name in class_dirs:
            saliency_paths[class_name] = []

        for i, input_data in enumerate(input_data_list):
            try:
                # Generate saliency map
                saliency_map = model.generate_saliency_map(input_data, class_idx)

                # Determine output path
                if class_dirs:
                    # If we have class information, save to class-specific directory
                    # For now, we'll just save to all class directories
                    for class_name, class_dir in class_dirs.items():
                        save_path = os.path.join(
                            class_dir, f"{prefix}_subject_{i+1}_class_{class_name}.png"
                        )
                        self._save_saliency_map(
                            saliency_map,
                            input_data,
                            save_path,
                            f"Saliency Map - Subject {i+1}",
                        )
                        saliency_paths[class_name].append(save_path)
                else:
                    # Save to general directory
                    save_path = os.path.join(output_dir, f"{prefix}_subject_{i+1}.png")
                    self._save_saliency_map(
                        saliency_map,
                        input_data,
                        save_path,
                        f"Saliency Map - Subject {i+1}",
                    )
                    saliency_paths["all"].append(save_path)

            except Exception as e:
                print(f"Error generating saliency map for subject {i+1}: {e}")
                continue

        return saliency_paths

    def generate_grad_cam_batch(
        self,
        model: SSPNet3DCNN,
        input_data_list: List[np.ndarray],
        class_idx: int = 0,
        layer_name: str = "conv2",
        output_dir: Optional[str] = None,
        prefix: str = "gradcam",
    ) -> Dict[str, List[str]]:
        """
        Generate Grad-CAM maps for a batch of subjects

        Args:
            model: SSPNet3DCNN model
            input_data_list: List of input data arrays
            class_idx: Class index for Grad-CAM generation
            layer_name: Layer name for Grad-CAM
            output_dir: Output directory for saving results
            prefix: Prefix for output files

        Returns:
            Dictionary with paths to generated Grad-CAM maps for each class
        """
        if output_dir is None:
            output_dir = os.path.join(
                self.config.visualization.output_dir, "figures", "gradcam"
            )

        ensure_dir(output_dir)

        # Determine class directories
        class_dirs = {}
        if hasattr(input_data_list[0], "shape") and len(input_data_list[0].shape) > 1:
            # If we have labels, create class-specific directories
            class_dirs = {
                "class_0": os.path.join(output_dir, "class_0"),
                "class_1": os.path.join(output_dir, "class_1"),
            }
            for class_dir in class_dirs.values():
                ensure_dir(class_dir)

        # Generate Grad-CAM maps for each subject
        gradcam_paths = {"all": []}
        for class_name in class_dirs:
            gradcam_paths[class_name] = []

        for i, input_data in enumerate(input_data_list):
            try:
                # Generate Grad-CAM
                grad_cam = model.generate_grad_cam(input_data, class_idx, layer_name)

                # Determine output path
                if class_dirs:
                    # If we have class information, save to class-specific directories
                    for class_name, class_dir in class_dirs.items():
                        save_path = os.path.join(
                            class_dir, f"{prefix}_subject_{i+1}_class_{class_name}.png"
                        )
                        self._save_grad_cam(
                            grad_cam, input_data, save_path, f"Grad-CAM - Subject {i+1}"
                        )
                        gradcam_paths[class_name].append(save_path)
                else:
                    # Save to general directory
                    save_path = os.path.join(output_dir, f"{prefix}_subject_{i+1}.png")
                    self._save_grad_cam(
                        grad_cam, input_data, save_path, f"Grad-CAM - Subject {i+1}"
                    )
                    gradcam_paths["all"].append(save_path)

            except Exception as e:
                print(f"Error generating Grad-CAM for subject {i+1}: {e}")
                continue

        return gradcam_paths

    def _save_saliency_map(
        self,
        saliency_map: np.ndarray,
        input_data: np.ndarray,
        save_path: str,
        title: str,
    ):
        """
        Save saliency map visualization

        Args:
            saliency_map: Saliency map array
            input_data: Input data array
            save_path: Path to save the visualization
            title: Title for the plot
        """
        # Remove channel dimension if present
        if len(saliency_map.shape) == 4:
            saliency_map = saliency_map[:, :, :, 0]
        if len(input_data.shape) == 4:
            input_data = input_data[:, :, :, 0]

        # Create middle slice visualization
        middle_slice = saliency_map.shape[2] // 2
        saliency_slice = saliency_map[:, :, middle_slice]
        input_slice = input_data[:, :, middle_slice]

        # Create figure
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Original input
        axes[0].imshow(input_slice, cmap="gray")
        axes[0].set_title("Original Input", fontsize=self.title_fontsize)
        axes[0].axis("off")

        # Saliency map
        im1 = axes[1].imshow(saliency_slice, cmap="hot")
        axes[1].set_title("Saliency Map", fontsize=self.title_fontsize)
        axes[1].axis("off")
        plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

        # Overlay
        axes[2].imshow(input_slice, cmap="gray")
        im2 = axes[2].imshow(saliency_slice, cmap="hot", alpha=0.5)
        axes[2].set_title("Overlay", fontsize=self.title_fontsize)
        axes[2].axis("off")
        plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

        plt.suptitle(title, fontsize=self.title_fontsize + 2)
        plt.tight_layout()

        # Save figure
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

        plt.close()

    def _save_grad_cam(
        self, grad_cam: np.ndarray, input_data: np.ndarray, save_path: str, title: str
    ):
        """
        Save Grad-CAM visualization

        Args:
            grad_cam: Grad-CAM array
            input_data: Input data array
            save_path: Path to save the visualization
            title: Title for the plot
        """
        # Remove channel dimension if present
        if len(input_data.shape) == 4:
            input_data = input_data[:, :, :, 0]

        # Create middle slice visualization
        middle_slice = grad_cam.shape[2] // 2
        grad_cam_slice = grad_cam[:, :, middle_slice]
        input_slice = input_data[:, :, middle_slice]

        # Create figure
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Original input
        axes[0].imshow(input_slice, cmap="gray")
        axes[0].set_title("Original Input", fontsize=self.title_fontsize)
        axes[0].axis("off")

        # Grad-CAM heatmap
        im1 = axes[1].imshow(grad_cam_slice, cmap="jet")
        axes[1].set_title("Grad-CAM Heatmap", fontsize=self.title_fontsize)
        axes[1].axis("off")
        plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

        # Overlay
        axes[2].imshow(input_slice, cmap="gray")
        im2 = axes[2].imshow(grad_cam_slice, cmap="jet", alpha=0.5)
        axes[2].set_title("Overlay", fontsize=self.title_fontsize)
        axes[2].axis("off")
        plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

        plt.suptitle(title, fontsize=self.title_fontsize + 2)
        plt.tight_layout()

        # Save figure
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

        plt.close()

    def plot_interpretability_comparison(
        self,
        input_data: np.ndarray,
        saliency_map: np.ndarray,
        grad_cam: np.ndarray,
        slice_idx: Optional[int] = None,
        save_path: Optional[str] = None,
    ):
        """
        Create a side-by-side comparison of input slices, saliency, and Grad-CAM

        Args:
            input_data: Input data array
            saliency_map: Saliency map array
            grad_cam: Grad-CAM array
            slice_idx: Index of the slice to visualize
            save_path: Path to save the plot

        Returns:
            Matplotlib figure object
        """
        # Remove channel dimension if present
        if len(input_data.shape) == 4:
            input_data = input_data[:, :, :, 0]
        if len(saliency_map.shape) == 4:
            saliency_map = saliency_map[:, :, :, 0]

        # Determine slice to visualize
        if slice_idx is None:
            slice_idx = input_data.shape[2] // 2

        # Extract slices
        input_slice = input_data[:, :, slice_idx]
        saliency_slice = saliency_map[:, :, slice_idx]
        grad_cam_slice = grad_cam[:, :, slice_idx]

        # Create figure with 2 rows and 3 columns
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        # First row: Individual visualizations
        # Original input
        axes[0, 0].imshow(input_slice, cmap="gray")
        axes[0, 0].set_title("Original Input", fontsize=self.title_fontsize)
        axes[0, 0].axis("off")

        # Saliency map
        im1 = axes[0, 1].imshow(saliency_slice, cmap="hot")
        axes[0, 1].set_title("Saliency Map", fontsize=self.title_fontsize)
        axes[0, 1].axis("off")
        plt.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)

        # Grad-CAM
        im2 = axes[0, 2].imshow(grad_cam_slice, cmap="jet")
        axes[0, 2].set_title("Grad-CAM", fontsize=self.title_fontsize)
        axes[0, 2].axis("off")
        plt.colorbar(im2, ax=axes[0, 2], fraction=0.046, pad=0.04)

        # Second row: Overlays
        # Saliency overlay
        axes[1, 0].imshow(input_slice, cmap="gray")
        im3 = axes[1, 0].imshow(saliency_slice, cmap="hot", alpha=0.5)
        axes[1, 0].set_title("Saliency Overlay", fontsize=self.title_fontsize)
        axes[1, 0].axis("off")
        plt.colorbar(im3, ax=axes[1, 0], fraction=0.046, pad=0.04)

        # Grad-CAM overlay
        axes[1, 1].imshow(input_slice, cmap="gray")
        im4 = axes[1, 1].imshow(grad_cam_slice, cmap="jet", alpha=0.5)
        axes[1, 1].set_title("Grad-CAM Overlay", fontsize=self.title_fontsize)
        axes[1, 1].axis("off")
        plt.colorbar(im4, ax=axes[1, 1], fraction=0.046, pad=0.04)

        # Combined overlay
        axes[1, 2].imshow(input_slice, cmap="gray")
        im5 = axes[1, 2].imshow(saliency_slice, cmap="hot", alpha=0.3)
        im6 = axes[1, 2].imshow(grad_cam_slice, cmap="jet", alpha=0.3)
        axes[1, 2].set_title("Combined Overlay", fontsize=self.title_fontsize)
        axes[1, 2].axis("off")
        plt.colorbar(im5, ax=axes[1, 2], fraction=0.046, pad=0.04)

        plt.suptitle(
            f"Model Interpretability Comparison (Slice {slice_idx})",
            fontsize=self.title_fontsize + 2,
        )
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

    def generate_interpretability_report(
        self,
        model: SSPNet3DCNN,
        input_data_list: List[np.ndarray],
        output_dir: Optional[str] = None,
        prefix: str = "interpretability",
    ) -> Dict[str, Union[List[str], str]]:
        """
        Generate a comprehensive interpretability report for a list of subjects

        Args:
            model: SSPNet3DCNN model
            input_data_list: List of input data arrays
            output_dir: Output directory for saving results
            prefix: Prefix for output files

        Returns:
            Dictionary with paths to generated files
        """
        if output_dir is None:
            output_dir = os.path.join(
                self.config.visualization.output_dir, "figures", "interpretability"
            )

        ensure_dir(output_dir)

        # Generate saliency maps
        saliency_paths = self.generate_saliency_maps_batch(
            model, input_data_list, output_dir=output_dir, prefix=f"{prefix}_saliency"
        )

        # Generate Grad-CAM maps
        gradcam_paths = self.generate_grad_cam_batch(
            model, input_data_list, output_dir=output_dir, prefix=f"{prefix}_gradcam"
        )

        # Generate comparison plots for a few examples
        comparison_paths = []
        num_examples = min(5, len(input_data_list))

        for i in range(num_examples):
            try:
                # Generate saliency and Grad-CAM for this example
                saliency_map = model.generate_saliency_map(input_data_list[i])
                grad_cam = model.generate_grad_cam(input_data_list[i])

                # Create comparison plot
                comparison_path = os.path.join(
                    output_dir, f"{prefix}_comparison_subject_{i+1}.png"
                )
                self.plot_interpretability_comparison(
                    input_data_list[i],
                    saliency_map,
                    grad_cam,
                    save_path=comparison_path,
                )
                comparison_paths.append(comparison_path)

            except Exception as e:
                print(f"Error generating comparison for subject {i+1}: {e}")
                continue

        # Plot model architecture
        architecture_path = os.path.join(output_dir, f"{prefix}_model_architecture.png")
        self.plot_model_architecture(model, architecture_path)

        return {
            "saliency_maps": saliency_paths,
            "gradcam_maps": gradcam_paths,
            "comparisons": comparison_paths,
            "architecture": architecture_path,
        }

    def plot_layer_weights(
        self, model: SSPNet3DCNN, layer_name: str, save_path: Optional[str] = None
    ):
        """
        Plot the weights of a specific layer

        Args:
            model: SSPNet3DCNN model
            layer_name: Name of the layer
            save_path: Path to save the plot

        Returns:
            Matplotlib figure object
        """
        # Get the layer
        layer = None
        for l in model.layers:
            if l.name == layer_name:
                layer = l
                break

        if layer is None:
            raise ValueError(f"Layer '{layer_name}' not found in the model")

        # Get weights
        weights = layer.get_weights()
        if not weights:
            raise ValueError(f"No weights found for layer '{layer_name}'")

        # Create figure
        fig, axes = plt.subplots(1, len(weights), figsize=(5 * len(weights), 5))
        if len(weights) == 1:
            axes = [axes]

        # Plot weights
        for i, weight_array in enumerate(weights):
            if len(weight_array.shape) == 4:  # Convolutional layer
                # Plot a few filters
                num_filters = min(16, weight_array.shape[-1])
                cols = 4
                rows = (num_filters + cols - 1) // cols

                axes[i].clear()
                for j in range(num_filters):
                    plt.subplot(rows, cols, j + 1)

                    # Take mean across spatial dimensions for visualization
                    if len(weight_array.shape) == 5:
                        filter_weights = np.mean(weight_array[..., j], axis=-1)
                    else:
                        filter_weights = weight_array[..., j]

                    plt.imshow(filter_weights, cmap="viridis")
                    plt.title(f"Filter {j}", fontsize=self.tick_fontsize)
                    plt.axis("off")

                plt.suptitle(
                    f"Layer: {layer_name} - Kernel Weights",
                    fontsize=self.title_fontsize,
                )

            elif len(weight_array.shape) == 2:  # Dense layer
                im = axes[i].imshow(weight_array, cmap="viridis")
                axes[i].set_title(
                    f"Layer: {layer_name} - Weights", fontsize=self.title_fontsize
                )
                axes[i].set_xlabel("Input Units", fontsize=self.label_fontsize)
                axes[i].set_ylabel("Output Units", fontsize=self.label_fontsize)
                plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)

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

    def plot_model_summary(self, model: SSPNet3DCNN, save_path: Optional[str] = None):
        """
        Create a summary plot of the model architecture and parameters

        Args:
            model: SSPNet3DCNN model
            save_path: Path to save the plot

        Returns:
            Matplotlib figure object
        """
        # Get parameter counts
        param_counts = model.count_parameters()

        # Create figure
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        # Plot parameter distribution
        labels = list(param_counts.keys())
        values = list(param_counts.values())

        # Remove 'total' for the pie chart
        pie_labels = [label for label in labels if label != "total"]
        pie_values = [values[i] for i, label in enumerate(labels) if label != "total"]

        # Pie chart
        axes[0].pie(pie_values, labels=pie_labels, autopct="%1.1f%%", startangle=90)
        axes[0].set_title("Parameter Distribution", fontsize=self.title_fontsize)

        # Bar chart
        bars = axes[1].bar(
            labels, values, color=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
        )
        axes[1].set_title("Parameter Counts", fontsize=self.title_fontsize)
        axes[1].set_ylabel("Number of Parameters", fontsize=self.label_fontsize)
        axes[1].tick_params(axis="both", which="major", labelsize=self.tick_fontsize)

        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            axes[1].text(
                bar.get_x() + bar.get_width() / 2.0,
                height + max(values) * 0.01,
                f"{value:,}",
                ha="center",
                va="bottom",
                fontsize=self.tick_fontsize,
            )

        # Use log scale if values vary widely
        if max(values) / min(values) > 100:
            axes[1].set_yscale("log")

        plt.suptitle("Model Architecture Summary", fontsize=self.title_fontsize + 2)
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
def plot_model_architecture(model, save_path=None):
    """
    Convenience function to plot model architecture

    Args:
        model: Neural network model
        save_path: Path to save the plot
    """
    visualizer = ModelVisualizer()
    return visualizer.plot_model_architecture(model, save_path)


def generate_saliency_maps_batch(
    model, input_data_list, class_idx=0, output_dir=None, prefix="saliency"
):
    """
    Convenience function to generate saliency maps for a batch of subjects

    Args:
        model: SSPNet3DCNN model
        input_data_list: List of input data arrays
        class_idx: Class index for saliency generation
        output_dir: Output directory for saving results
        prefix: Prefix for output files

    Returns:
        Dictionary with paths to generated saliency maps
    """
    visualizer = ModelVisualizer()
    return visualizer.generate_saliency_maps_batch(
        model, input_data_list, class_idx, output_dir, prefix
    )


def generate_grad_cam_batch(
    model,
    input_data_list,
    class_idx=0,
    layer_name="conv2",
    output_dir=None,
    prefix="gradcam",
):
    """
    Convenience function to generate Grad-CAM maps for a batch of subjects

    Args:
        model: SSPNet3DCNN model
        input_data_list: List of input data arrays
        class_idx: Class index for Grad-CAM generation
        layer_name: Layer name for Grad-CAM
        output_dir: Output directory for saving results
        prefix: Prefix for output files

    Returns:
        Dictionary with paths to generated Grad-CAM maps
    """
    visualizer = ModelVisualizer()
    return visualizer.generate_grad_cam_batch(
        model, input_data_list, class_idx, layer_name, output_dir, prefix
    )


def generate_interpretability_report(
    model, input_data_list, output_dir=None, prefix="interpretability"
):
    """
    Convenience function to generate a comprehensive interpretability report

    Args:
        model: SSPNet3DCNN model
        input_data_list: List of input data arrays
        output_dir: Output directory for saving results
        prefix: Prefix for output files

    Returns:
        Dictionary with paths to generated files
    """
    visualizer = ModelVisualizer()
    return visualizer.generate_interpretability_report(
        model, input_data_list, output_dir, prefix
    )
