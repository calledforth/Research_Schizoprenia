"""
Brain visualization utilities for neuroimaging data
"""

import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from nilearn import plotting, image, datasets
from nilearn.surface import SurfaceImage, vol_to_surf
from pathlib import Path
import os
from ..config import default_config
from ..utils.file_utils import ensure_dir


class BrainVisualizer:
    """
    Class for visualizing brain imaging data
    """

    def __init__(self, config=None):
        """
        Initialize the brain visualizer

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

        # Default slice coordinates for MNI space
        self.default_coords = {
            "axial": None,  # Will be set to middle slice
            "sagittal": None,
            "coronal": None,
        }

    def plot_grad_cam_slices(
        self,
        grad_cam_data,
        bg_img=None,
        slice_coords=None,
        title=None,
        save_path=None,
        axes=None,
    ):
        """
        Visualize Grad-CAM volumes as orthogonal slice grids with anatomical underlay

        Args:
            grad_cam_data: 3D Grad-CAM volume
            bg_img: Background image (path or NIfTI image)
            slice_coords: Dictionary with slice coordinates for each view
            title: Title for the plot
            save_path: Path to save the plot
            axes: Matplotlib axes array (creates new figure if None)

        Returns:
            Matplotlib axes array
        """
        # Ensure 3D data
        if len(grad_cam_data.shape) > 3:
            grad_cam_data = np.squeeze(grad_cam_data)

        # Set default slice coordinates if not provided
        if slice_coords is None:
            slice_coords = {
                "x": grad_cam_data.shape[0] // 2,
                "y": grad_cam_data.shape[1] // 2,
                "z": grad_cam_data.shape[2] // 2,
            }

        # Create figure if no axes provided
        if axes is None:
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Create NIfTI image from Grad-CAM data
        grad_cam_img = nib.Nifti1Image(grad_cam_data, affine=np.eye(4))

        # Plot sagittal view
        plotting.plot_stat_map(
            grad_cam_img,
            bg_img=bg_img,
            cut_coords=[slice_coords["x"]],
            display_mode="x",
            title=f"Sagittal (x={slice_coords['x']})",
            axes=axes[0],
            cmap="jet",
            threshold=0.1,
            black_bg=False,
        )

        # Plot coronal view
        plotting.plot_stat_map(
            grad_cam_img,
            bg_img=bg_img,
            cut_coords=[slice_coords["y"]],
            display_mode="y",
            title=f"Coronal (y={slice_coords['y']})",
            axes=axes[1],
            cmap="jet",
            threshold=0.1,
            black_bg=False,
        )

        # Plot axial view
        plotting.plot_stat_map(
            grad_cam_img,
            bg_img=bg_img,
            cut_coords=[slice_coords["z"]],
            display_mode="z",
            title=f"Axial (z={slice_coords['z']})",
            axes=axes[2],
            cmap="jet",
            threshold=0.1,
            black_bg=False,
        )

        # Set main title if provided
        if title:
            plt.suptitle(title, fontsize=self.title_fontsize)

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

    def plot_saliency_slices(
        self,
        saliency_data,
        bg_img=None,
        slice_coords=None,
        title=None,
        save_path=None,
        axes=None,
    ):
        """
        Visualize saliency volumes as orthogonal slice grids with anatomical underlay

        Args:
            saliency_data: 3D saliency volume
            bg_img: Background image (path or NIfTI image)
            slice_coords: Dictionary with slice coordinates for each view
            title: Title for the plot
            save_path: Path to save the plot
            axes: Matplotlib axes array (creates new figure if None)

        Returns:
            Matplotlib axes array
        """
        # Ensure 3D data
        if len(saliency_data.shape) > 3:
            saliency_data = np.squeeze(saliency_data)

        # Set default slice coordinates if not provided
        if slice_coords is None:
            slice_coords = {
                "x": saliency_data.shape[0] // 2,
                "y": saliency_data.shape[1] // 2,
                "z": saliency_data.shape[2] // 2,
            }

        # Create figure if no axes provided
        if axes is None:
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Create NIfTI image from saliency data
        saliency_img = nib.Nifti1Image(saliency_data, affine=np.eye(4))

        # Plot sagittal view
        plotting.plot_stat_map(
            saliency_img,
            bg_img=bg_img,
            cut_coords=[slice_coords["x"]],
            display_mode="x",
            title=f"Sagittal (x={slice_coords['x']})",
            axes=axes[0],
            cmap="hot",
            threshold=0.1,
            black_bg=False,
        )

        # Plot coronal view
        plotting.plot_stat_map(
            saliency_img,
            bg_img=bg_img,
            cut_coords=[slice_coords["y"]],
            display_mode="y",
            title=f"Coronal (y={slice_coords['y']})",
            axes=axes[1],
            cmap="hot",
            threshold=0.1,
            black_bg=False,
        )

        # Plot axial view
        plotting.plot_stat_map(
            saliency_img,
            bg_img=bg_img,
            cut_coords=[slice_coords["z"]],
            display_mode="z",
            title=f"Axial (z={slice_coords['z']})",
            axes=axes[2],
            cmap="hot",
            threshold=0.1,
            black_bg=False,
        )

        # Set main title if provided
        if title:
            plt.suptitle(title, fontsize=self.title_fontsize)

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

    def plot_slice_montage(
        self,
        volume_data,
        axis="z",
        slice_range=None,
        n_slices=8,
        title=None,
        save_path=None,
        ax=None,
    ):
        """
        Create a montage across selected z-slices

        Args:
            volume_data: 3D volume data
            axis: Axis along which to take slices ('x', 'y', or 'z')
            slice_range: Range of slices to include (start, end)
            n_slices: Number of slices to display
            title: Title for the plot
            save_path: Path to save the plot
            ax: Matplotlib axis to plot on

        Returns:
            Matplotlib axis object
        """
        # Ensure 3D data
        if len(volume_data.shape) > 3:
            volume_data = np.squeeze(volume_data)

        # Determine slice indices
        max_slice = volume_data.shape[{"x": 0, "y": 1, "z": 2}[axis]]

        if slice_range is None:
            start = max(0, max_slice // 2 - n_slices // 2)
            end = min(max_slice, start + n_slices)
        else:
            start, end = slice_range

        slice_indices = np.linspace(start, end, n_slices, dtype=int)

        # Create figure if no axis provided
        if ax is None:
            cols = 4
            rows = (n_slices + cols - 1) // cols
            fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
            if rows == 1:
                axes = axes.reshape(1, -1)
        else:
            axes = ax

        # Flatten axes array for easier indexing
        axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]

        # Plot each slice
        for i, slice_idx in enumerate(slice_indices):
            if i >= len(axes_flat):
                break

            # Extract slice
            if axis == "x":
                slice_data = volume_data[slice_idx, :, :]
            elif axis == "y":
                slice_data = volume_data[:, slice_idx, :]
            else:  # axis == 'z'
                slice_data = volume_data[:, :, slice_idx]

            # Plot slice
            im = axes_flat[i].imshow(slice_data.T, cmap="gray", origin="lower")
            axes_flat[i].set_title(f"Slice {slice_idx}", fontsize=self.tick_fontsize)
            axes_flat[i].axis("off")

            # Add colorbar
            plt.colorbar(im, ax=axes_flat[i], fraction=0.046, pad=0.04)

        # Hide unused subplots
        for i in range(len(slice_indices), len(axes_flat)):
            axes_flat[i].axis("off")

        # Set main title if provided
        if title:
            plt.suptitle(title, fontsize=self.title_fontsize)

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

    def plot_3d_volume_rendering(
        self, volume_data, threshold=None, title=None, save_path=None
    ):
        """
        Create 3D surface/volume rendering with fallback to slice-based visualization

        Args:
            volume_data: 3D volume data
            threshold: Threshold for volume rendering
            title: Title for the plot
            save_path: Path to save the plot

        Returns:
            Matplotlib figure object
        """
        # Ensure 3D data
        if len(volume_data.shape) > 3:
            volume_data = np.squeeze(volume_data)

        # Create NIfTI image
        img = nib.Nifti1Image(volume_data, affine=np.eye(4))

        # Try 3D rendering first
        try:
            fig = plt.figure(figsize=(10, 8))

            # Use nilearn's 3D plotting
            plotting.plot_glass_brain(
                img,
                title=title,
                threshold=threshold,
                display_mode="ortho",
                plot_abs=False,
                figure=fig,
            )

            plt.tight_layout()

        except Exception as e:
            print(f"3D rendering failed ({e}), falling back to slice visualization")

            # Fallback to slice visualization
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))

            # Plot orthogonal views
            plotting.plot_anat(
                img, cut_coords=8, display_mode="x", title="Sagittal", axes=axes[0]
            )

            plotting.plot_anat(
                img, cut_coords=8, display_mode="y", title="Coronal", axes=axes[1]
            )

            plotting.plot_anat(
                img, cut_coords=8, display_mode="z", title="Axial", axes=axes[2]
            )

            if title:
                plt.suptitle(title, fontsize=self.title_fontsize)

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

    def plot_fnc_heatmap(
        self, fnc_matrix, labels=None, title=None, save_path=None, ax=None
    ):
        """
        Plot FNC matrix as heatmap

        Args:
            fnc_matrix: Functional connectivity matrix
            labels: Region labels
            title: Title for the plot
            save_path: Path to save the plot
            ax: Matplotlib axis to plot on

        Returns:
            Matplotlib axis object
        """
        # Create figure if no axis provided
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 8))

        # Plot heatmap
        im = ax.imshow(fnc_matrix, cmap="coolwarm", vmin=-1, vmax=1)

        # Add colorbar
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # Add labels if provided
        if labels:
            ax.set_xticks(range(len(labels)))
            ax.set_yticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=90, fontsize=self.tick_fontsize)
            ax.set_yticklabels(labels, fontsize=self.tick_fontsize)

        # Set title
        if title:
            ax.set_title(title, fontsize=self.title_fontsize)
        else:
            ax.set_title(
                "Functional Network Connectivity", fontsize=self.title_fontsize
            )

        # Add grid
        ax.grid(False)

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

    def plot_fnc_degree_map(
        self, fnc_matrix, labels=None, title=None, save_path=None, ax=None
    ):
        """
        Plot degree map from FNC matrix

        Args:
            fnc_matrix: Functional connectivity matrix
            labels: Region labels
            title: Title for the plot
            save_path: Path to save the plot
            ax: Matplotlib axis to plot on

        Returns:
            Matplotlib axis object
        """
        # Calculate degree (sum of absolute connections)
        degree = np.sum(np.abs(fnc_matrix), axis=1)

        # Create figure if no axis provided
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 6))

        # Plot degree map
        bars = ax.bar(range(len(degree)), degree, color="steelblue")

        # Add labels if provided
        if labels:
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=90, fontsize=self.tick_fontsize)

        # Set title and labels
        if title:
            ax.set_title(title, fontsize=self.title_fontsize)
        else:
            ax.set_title("Network Degree Map", fontsize=self.title_fontsize)

        ax.set_xlabel("Brain Regions", fontsize=self.label_fontsize)
        ax.set_ylabel("Degree", fontsize=self.label_fontsize)
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

    def plot_fnc_group_comparison(
        self,
        fnc_group1,
        fnc_group2,
        labels=None,
        group_names=None,
        title=None,
        save_path=None,
    ):
        """
        Plot FNC group averages and differences

        Args:
            fnc_group1: FNC matrix for group 1
            fnc_group2: FNC matrix for group 2
            labels: Region labels
            group_names: Names of the groups
            title: Title for the plot
            save_path: Path to save the plot

        Returns:
            Matplotlib figure object
        """
        if group_names is None:
            group_names = ["Group 1", "Group 2"]

        # Calculate difference
        diff_matrix = fnc_group1 - fnc_group2

        # Create figure with subplots
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # Plot group 1
        im1 = axes[0].imshow(fnc_group1, cmap="coolwarm", vmin=-1, vmax=1)
        axes[0].set_title(f"{group_names[0]} Average", fontsize=self.title_fontsize)
        plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)

        # Plot group 2
        im2 = axes[1].imshow(fnc_group2, cmap="coolwarm", vmin=-1, vmax=1)
        axes[1].set_title(f"{group_names[1]} Average", fontsize=self.title_fontsize)
        plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

        # Plot difference
        max_diff = np.max(np.abs(diff_matrix))
        im3 = axes[2].imshow(
            diff_matrix, cmap="coolwarm", vmin=-max_diff, vmax=max_diff
        )
        axes[2].set_title(
            "Difference (Group 1 - Group 2)", fontsize=self.title_fontsize
        )
        plt.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)

        # Add labels if provided
        if labels:
            for ax in axes:
                ax.set_xticks(range(len(labels)))
                ax.set_yticks(range(len(labels)))
                ax.set_xticklabels(labels, rotation=90, fontsize=self.tick_fontsize)
                ax.set_yticklabels(labels, fontsize=self.tick_fontsize)

        # Set main title
        if title:
            plt.suptitle(title, fontsize=self.title_fontsize + 2)

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

    def plot_fnc_surface_projection(
        self, fnc_matrix, component_to_region_mapping=None, title=None, save_path=None
    ):
        """
        Project network strengths to cortical surface

        Args:
            fnc_matrix: Functional connectivity matrix
            component_to_region_mapping: Dictionary mapping components to regions
            title: Title for the plot
            save_path: Path to save the plot

        Returns:
            Matplotlib figure object
        """
        try:
            # Try to use fsaverage surface if nilearn is available
            fsaverage = datasets.fetch_surf_fsaverage()

            # Calculate degree map
            degree = np.sum(np.abs(fnc_matrix), axis=1)

            # Create a dummy volume for surface projection
            # This is a simplified approach - in practice you'd need proper component-to-region mapping
            dummy_volume = np.zeros((91, 109, 91))  # MNI space dimensions

            # If component_to_region_mapping is provided, use it
            if component_to_region_mapping:
                for comp_idx, region_coords in component_to_region_mapping.items():
                    if comp_idx < len(degree):
                        x, y, z = region_coords
                        if (
                            0 <= x < dummy_volume.shape[0]
                            and 0 <= y < dummy_volume.shape[1]
                            and 0 <= z < dummy_volume.shape[2]
                        ):
                            dummy_volume[x, y, z] = degree[comp_idx]

            # Create surface image
            texture = vol_to_surf(
                nib.Nifti1Image(dummy_volume, np.eye(4)), fsaverage.pial_left
            )

            # Plot surface
            fig = plt.figure(figsize=(10, 8))

            plotting.plot_surf(
                fsaverage.infl_left,
                texture,
                hemi="left",
                view="lateral",
                title=title if title else "Network Strength Surface Projection",
                threshold=0.1,
                cmap="jet",
                figure=fig,
            )

            plt.tight_layout()

        except Exception as e:
            print(f"Surface projection failed ({e}), falling back to heatmap")

            # Fallback to heatmap visualization
            fig, ax = plt.subplots(figsize=(10, 8))

            # Calculate degree map
            degree = np.sum(np.abs(fnc_matrix), axis=1)

            # Plot degree as heatmap
            im = ax.imshow(degree.reshape(1, -1), cmap="jet", aspect="auto")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            ax.set_title(
                title if title else "Network Strength", fontsize=self.title_fontsize
            )
            ax.set_xlabel("Components", fontsize=self.label_fontsize)
            ax.set_yticks([])

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

    def plot_meg_source_envelopes(
        self, meg_data, title=None, save_path=None, axes=None
    ):
        """
        Display MEG source-space amplitude envelopes in MNI space

        Args:
            meg_data: MEG source-space data (3D volume)
            title: Title for the plot
            save_path: Path to save the plot
            axes: Matplotlib axes array (creates new figure if None)

        Returns:
            Matplotlib axes array
        """
        # Ensure 3D data
        if len(meg_data.shape) > 3:
            meg_data = np.squeeze(meg_data)

        # Apply smoothing if needed
        from scipy.ndimage import gaussian_filter

        smoothed_data = gaussian_filter(meg_data, sigma=1.0)

        # Create figure if no axes provided
        if axes is None:
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Create NIfTI image
        img = nib.Nifti1Image(smoothed_data, affine=np.eye(4))

        # Plot orthogonal views
        plotting.plot_stat_map(
            img,
            cut_coords=8,
            display_mode="x",
            title="Sagittal",
            axes=axes[0],
            cmap=self.config.visualization.meg_colormap,
            threshold="auto",
        )

        plotting.plot_stat_map(
            img,
            cut_coords=8,
            display_mode="y",
            title="Coronal",
            axes=axes[1],
            cmap=self.config.visualization.meg_colormap,
            threshold="auto",
        )

        plotting.plot_stat_map(
            img,
            cut_coords=8,
            display_mode="z",
            title="Axial",
            axes=axes[2],
            cmap=self.config.visualization.meg_colormap,
            threshold="auto",
        )

        # Set main title if provided
        if title:
            plt.suptitle(title, fontsize=self.title_fontsize)

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

    def generate_brain_figure_set(
        self,
        grad_cam_data=None,
        saliency_data=None,
        fnc_matrices=None,
        meg_data=None,
        subject_id=None,
        output_dir=None,
    ):
        """
        Generate a comprehensive set of brain visualization figures

        Args:
            grad_cam_data: Grad-CAM volume data
            saliency_data: Saliency volume data
            fnc_matrices: Dictionary with FNC matrices for different groups
            meg_data: MEG source-space data
            subject_id: Subject ID for naming files
            output_dir: Output directory for figures

        Returns:
            Dictionary with paths to generated figures
        """
        if output_dir is None:
            output_dir = os.path.join(
                self.config.visualization.output_dir, "figures", "brain"
            )

        ensure_dir(output_dir)

        figure_paths = {}

        # Generate prefix for filenames
        prefix = f"subject_{subject_id}" if subject_id else "brain_analysis"

        # 1. Grad-CAM visualizations
        if grad_cam_data is not None:
            # Orthogonal slices
            grad_cam_path = os.path.join(output_dir, f"{prefix}_gradcam_slices.png")
            self.plot_grad_cam_slices(grad_cam_data, save_path=grad_cam_path)
            figure_paths["gradcam_slices"] = grad_cam_path

            # Slice montage
            grad_cam_montage_path = os.path.join(
                output_dir, f"{prefix}_gradcam_montage.png"
            )
            self.plot_slice_montage(
                grad_cam_data,
                title="Grad-CAM Slice Montage",
                save_path=grad_cam_montage_path,
            )
            figure_paths["gradcam_montage"] = grad_cam_montage_path

            # 3D rendering
            grad_cam_3d_path = os.path.join(output_dir, f"{prefix}_gradcam_3d.png")
            self.plot_3d_volume_rendering(
                grad_cam_data, title="Grad-CAM 3D Rendering", save_path=grad_cam_3d_path
            )
            figure_paths["gradcam_3d"] = grad_cam_3d_path

        # 2. Saliency visualizations
        if saliency_data is not None:
            # Orthogonal slices
            saliency_path = os.path.join(output_dir, f"{prefix}_saliency_slices.png")
            self.plot_saliency_slices(saliency_data, save_path=saliency_path)
            figure_paths["saliency_slices"] = saliency_path

            # Slice montage
            saliency_montage_path = os.path.join(
                output_dir, f"{prefix}_saliency_montage.png"
            )
            self.plot_slice_montage(
                saliency_data,
                title="Saliency Slice Montage",
                save_path=saliency_montage_path,
            )
            figure_paths["saliency_montage"] = saliency_montage_path

        # 3. FNC visualizations
        if fnc_matrices is not None:
            for group_name, fnc_matrix in fnc_matrices.items():
                # Heatmap
                fnc_heatmap_path = os.path.join(
                    output_dir, f"{prefix}_fnc_{group_name}_heatmap.png"
                )
                self.plot_fnc_heatmap(
                    fnc_matrix,
                    title=f"FNC Heatmap - {group_name}",
                    save_path=fnc_heatmap_path,
                )
                figure_paths[f"fnc_{group_name}_heatmap"] = fnc_heatmap_path

                # Degree map
                fnc_degree_path = os.path.join(
                    output_dir, f"{prefix}_fnc_{group_name}_degree.png"
                )
                self.plot_fnc_degree_map(
                    fnc_matrix,
                    title=f"FNC Degree Map - {group_name}",
                    save_path=fnc_degree_path,
                )
                figure_paths[f"fnc_{group_name}_degree"] = fnc_degree_path

                # Surface projection
                fnc_surface_path = os.path.join(
                    output_dir, f"{prefix}_fnc_{group_name}_surface.png"
                )
                self.plot_fnc_surface_projection(
                    fnc_matrix,
                    title=f"FNC Surface Projection - {group_name}",
                    save_path=fnc_surface_path,
                )
                figure_paths[f"fnc_{group_name}_surface"] = fnc_surface_path

            # Group comparison if multiple groups
            if len(fnc_matrices) > 1:
                group_names = list(fnc_matrices.keys())
                if len(group_names) == 2:
                    fnc_comparison_path = os.path.join(
                        output_dir, f"{prefix}_fnc_comparison.png"
                    )
                    self.plot_fnc_group_comparison(
                        fnc_matrices[group_names[0]],
                        fnc_matrices[group_names[1]],
                        group_names=group_names,
                        title="FNC Group Comparison",
                        save_path=fnc_comparison_path,
                    )
                    figure_paths["fnc_comparison"] = fnc_comparison_path

        # 4. MEG visualizations
        if meg_data is not None:
            meg_path = os.path.join(output_dir, f"{prefix}_meg_source_envelopes.png")
            self.plot_meg_source_envelopes(
                meg_data,
                title="MEG Source-Space Amplitude Envelopes",
                save_path=meg_path,
            )
            figure_paths["meg_source_envelopes"] = meg_path

        return figure_paths


# Convenience functions
def plot_grad_cam_slices(
    grad_cam_data, bg_img=None, slice_coords=None, title=None, save_path=None
):
    """
    Convenience function to plot Grad-CAM slices

    Args:
        grad_cam_data: 3D Grad-CAM volume
        bg_img: Background image (path or NIfTI image)
        slice_coords: Dictionary with slice coordinates for each view
        title: Title for the plot
        save_path: Path to save the plot
    """
    visualizer = BrainVisualizer()
    visualizer.plot_grad_cam_slices(
        grad_cam_data, bg_img, slice_coords, title, save_path
    )


def plot_saliency_slices(
    saliency_data, bg_img=None, slice_coords=None, title=None, save_path=None
):
    """
    Convenience function to plot saliency slices

    Args:
        saliency_data: 3D saliency volume
        bg_img: Background image (path or NIfTI image)
        slice_coords: Dictionary with slice coordinates for each view
        title: Title for the plot
        save_path: Path to save the plot
    """
    visualizer = BrainVisualizer()
    visualizer.plot_saliency_slices(
        saliency_data, bg_img, slice_coords, title, save_path
    )


def plot_fnc_heatmap(fnc_matrix, labels=None, title=None, save_path=None):
    """
    Convenience function to plot FNC heatmap

    Args:
        fnc_matrix: Functional connectivity matrix
        labels: Region labels
        title: Title for the plot
        save_path: Path to save the plot
    """
    visualizer = BrainVisualizer()
    visualizer.plot_fnc_heatmap(fnc_matrix, labels, title, save_path)


def generate_brain_report(
    grad_cam_data=None,
    saliency_data=None,
    fnc_matrices=None,
    meg_data=None,
    subject_id=None,
    output_dir=None,
):
    """
    Convenience function to generate brain visualization report

    Args:
        grad_cam_data: Grad-CAM volume data
        saliency_data: Saliency volume data
        fnc_matrices: Dictionary with FNC matrices for different groups
        meg_data: MEG source-space data
        subject_id: Subject ID for naming files
        output_dir: Output directory for figures

    Returns:
        Dictionary with paths to generated figures
    """
    visualizer = BrainVisualizer()
    return visualizer.generate_brain_figure_set(
        grad_cam_data, saliency_data, fnc_matrices, meg_data, subject_id, output_dir
    )
