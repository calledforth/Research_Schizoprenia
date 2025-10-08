"""
Feature to 3D conversion utilities for transforming FNC/SBM features to 96×96×96×1
"""

import numpy as np
import logging
from typing import Tuple, Dict, Any, Optional
from sklearn.manifold import MDS
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import pdist, squareform
from scipy.interpolate import RegularGridInterpolator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureTo3DConverter:
    """
    Convert FNC/SBM features to 3D representations (96×96×96×1)
    """

    def __init__(
        self,
        target_shape: Tuple[int, int, int, int] = (96, 96, 96, 1),
        method: str = "mds",
        random_state: int = 42,
    ):
        """
        Initialize the converter

        Args:
            target_shape (tuple): Target shape for 3D representation
            method (str): Conversion method ('mds', 'pca', 'graph', 'autoencoder')
            random_state (int): Random seed for reproducibility
        """
        self.target_shape = target_shape
        self.method = method
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.fitted = False
        # Fallback for tiny-sample cases where PCA/MDS cannot be fit
        self.fallback_brain_like = False

        # Initialize conversion method
        if method == "mds":
            self.converter = MDS(n_components=3, random_state=random_state)
        elif method == "pca":
            self.converter = PCA(n_components=3, random_state=random_state)
        elif method == "graph":
            self.converter = None  # Custom graph-based method
        elif method == "autoencoder":
            self.converter = None  # Custom autoencoder method
        else:
            raise ValueError(f"Unknown conversion method: {method}")

    def _create_connectivity_matrix(self, features: np.ndarray) -> np.ndarray:
        """
        Create a connectivity matrix from features

        Args:
            features (np.ndarray): Input features

        Returns:
            np.ndarray: Connectivity matrix
        """
        # Compute pairwise distances
        distances = pdist(features.reshape(1, -1))
        distance_matrix = squareform(distances)

        # Convert distances to similarities (connectivity)
        connectivity = 1 / (1 + distance_matrix)

        return connectivity

    def _graph_based_conversion(self, features: np.ndarray) -> np.ndarray:
        """
        Convert features to 3D using graph-based approach

        Args:
            features (np.ndarray): Input features

        Returns:
            np.ndarray: 3D coordinates
        """
        # Create connectivity matrix
        connectivity = self._create_connectivity_matrix(features)

        # Use eigendecomposition to get 3D embedding
        eigenvalues, eigenvectors = np.linalg.eigh(connectivity)

        # Get top 3 eigenvectors
        top_indices = np.argsort(eigenvalues)[-3:]
        coordinates = eigenvectors[:, top_indices]

        return coordinates

    def _autoencoder_conversion(self, features: np.ndarray) -> np.ndarray:
        """
        Convert features to 3D using simple autoencoder approach

        Args:
            features (np.ndarray): Input features

        Returns:
            np.ndarray: 3D coordinates
        """
        # Simple linear autoencoder using PCA
        pca = PCA(n_components=3, random_state=self.random_state)
        coordinates = pca.fit_transform(features.reshape(1, -1))

        return coordinates[0]

    def _interpolate_to_3d_grid(self, coordinates: np.ndarray) -> np.ndarray:
        """
        Interpolate 3D coordinates to a 96×96×96 grid

        Args:
            coordinates (np.ndarray): 3D coordinates

        Returns:
            np.ndarray: 3D grid representation
        """
        # Create a sparse 3D grid with points at the coordinates
        grid_size = self.target_shape[0]
        grid = np.zeros((grid_size, grid_size, grid_size))

        # Normalize coordinates to grid indices
        min_coords = np.min(coordinates, axis=0)
        max_coords = np.max(coordinates, axis=0)

        # Avoid division by zero
        if np.all(max_coords == min_coords):
            normalized_coords = np.ones_like(coordinates) * grid_size // 2
        else:
            normalized_coords = (coordinates - min_coords) / (max_coords - min_coords)
            normalized_coords = normalized_coords * (grid_size - 1)

        # Place values on the grid
        for i, coord in enumerate(normalized_coords):
            x, y, z = int(coord[0]), int(coord[1]), int(coord[2])
            if 0 <= x < grid_size and 0 <= y < grid_size and 0 <= z < grid_size:
                grid[x, y, z] = 1.0

        # Apply Gaussian smoothing to spread the values
        from scipy.ndimage import gaussian_filter

        grid = gaussian_filter(grid, sigma=2.0)

        # Normalize to [0, 1]
        if np.max(grid) > 0:
            grid = grid / np.max(grid)

        return grid

    def _create_brain_like_3d(self, features: np.ndarray) -> np.ndarray:
        """
        Create a brain-like 3D representation from features

        Args:
            features (np.ndarray): Input features

        Returns:
            np.ndarray: 3D brain-like representation
        """
        grid_size = self.target_shape[0]
        grid = np.zeros((grid_size, grid_size, grid_size))

        # Create a brain-like mask (ellipsoid)
        center = grid_size // 2
        for x in range(grid_size):
            for y in range(grid_size):
                for z in range(grid_size):
                    # Calculate distance from center
                    dist = np.sqrt(
                        (x - center) ** 2 + (y - center) ** 2 + (z - center) ** 2
                    )
                    # Create ellipsoid mask
                    if dist < grid_size * 0.4:
                        grid[x, y, z] = 1.0

        # Distribute features across the brain mask
        feature_indices = np.linspace(0, len(features) - 1, grid_size**3).astype(int)
        feature_values = features[feature_indices % len(features)]

        # Apply features to the brain mask
        grid[grid > 0] = feature_values[: np.sum(grid > 0)]

        # Apply Gaussian smoothing
        from scipy.ndimage import gaussian_filter

        grid = gaussian_filter(grid, sigma=1.5)

        # Normalize to [0, 1]
        if np.max(grid) > 0:
            grid = grid / np.max(grid)

        return grid

    def fit(self, features_list: list):
        """
        Fit the converter to a list of features

        Args:
            features_list (list): List of feature arrays
        """
        n_samples_input = len(features_list)
        logger.info(f"Fitting {self.method} converter to {n_samples_input} samples")

        # Stack all features for fitting
        all_features = np.vstack(
            [
                f.reshape(1, -1) if isinstance(f, np.ndarray) and f.ndim == 1 else f
                for f in features_list
            ]
        )
        n_samples, n_features = all_features.shape

        # Fit scaler
        self.scaler.fit(all_features)
        scaled_features = self.scaler.transform(all_features)

        # Handle dimensionality constraints
        if self.method == "pca":
            # n_components must be <= min(n_samples, n_features)
            n_comp = min(3, n_samples, n_features)
            if n_samples < 2 or n_comp < 1:
                self.fallback_brain_like = True
                self.fitted = True
                logger.warning(
                    "Insufficient samples/features for PCA; falling back to brain-like 3D conversion."
                )
                return
            # Recreate PCA with valid n_components if needed
            if (
                isinstance(self.converter, PCA)
                and getattr(self.converter, "n_components", None) != n_comp
            ):
                self.converter = PCA(
                    n_components=n_comp, random_state=self.random_state
                )
            self.converter.fit(scaled_features)

        elif self.method == "mds":
            # MDS requires at least 2 samples
            if n_samples < 2:
                self.fallback_brain_like = True
                self.fitted = True
                logger.warning(
                    "Only one sample available; falling back to brain-like 3D conversion."
                )
                return
            self.converter.fit(scaled_features)

        # Other methods ('graph', 'autoencoder') do not require fitting here
        self.fitted = True
        logger.info("Converter fitted successfully")

    def transform(self, features: np.ndarray) -> np.ndarray:
        """
        Transform features to 3D representation

        Args:
            features (np.ndarray): Input features

        Returns:
            np.ndarray: 3D representation (96×96×96×1)
        """
        # If not fitted, attempt to fit; handle tiny-sample edge cases
        if not self.fitted and self.method in ["mds", "pca"]:
            logger.warning("Converter not fitted, fitting on single sample")
            self.fit([features])

        # Scale features
        scaled_features = self.scaler.transform(features.reshape(1, -1))

        # Fallback path for insufficient samples/features
        if self.fallback_brain_like:
            grid = self._create_brain_like_3d(scaled_features[0])
            return grid.reshape(self.target_shape)

        # Convert to 3D coordinates, with robust fallback
        try:
            if self.method == "mds":
                coordinates = self.converter.fit_transform(scaled_features)[0]
            elif self.method == "pca":
                coordinates = self.converter.transform(scaled_features)[0]
            elif self.method == "graph":
                coordinates = self._graph_based_conversion(scaled_features[0])
            elif self.method == "autoencoder":
                coordinates = self._autoencoder_conversion(scaled_features[0])
            else:
                raise ValueError(f"Unknown conversion method: {self.method}")

            # Convert to 3D grid
            grid = self._interpolate_to_3d_grid(coordinates.reshape(1, -1))
            return grid.reshape(self.target_shape)
        except Exception as e:
            logger.warning(
                f"Conversion failed with method '{self.method}' ({e}); falling back to brain-like 3D."
            )
            grid = self._create_brain_like_3d(scaled_features[0])
            return grid.reshape(self.target_shape)

    def fit_transform(self, features_list: list) -> np.ndarray:
        """
        Fit the converter and transform features

        Args:
            features_list (list): List of feature arrays

        Returns:
            np.ndarray: 3D representations
        """
        self.fit(features_list)
        return np.array([self.transform(features) for features in features_list])


def convert_features_to_3d(
    features: np.ndarray,
    method: str = "mds",
    target_shape: Tuple[int, int, int, int] = (96, 96, 96, 1),
    random_state: int = 42,
) -> np.ndarray:
    """
    Convert features to 3D representation

    Args:
        features (np.ndarray): Input features
        method (str): Conversion method
        target_shape (tuple): Target shape for 3D representation
        random_state (int): Random seed for reproducibility

    Returns:
        np.ndarray: 3D representation
    """
    converter = FeatureTo3DConverter(
        target_shape=target_shape, method=method, random_state=random_state
    )

    return converter.transform(features)


def batch_convert_features_to_3d(
    features_list: list,
    method: str = "mds",
    target_shape: Tuple[int, int, int, int] = (96, 96, 96, 1),
    random_state: int = 42,
) -> np.ndarray:
    """
    Convert a batch of features to 3D representations

    Args:
        features_list (list): List of feature arrays
        method (str): Conversion method
        target_shape (tuple): Target shape for 3D representations
        random_state (int): Random seed for reproducibility

    Returns:
        np.ndarray: 3D representations
    """
    converter = FeatureTo3DConverter(
        target_shape=target_shape, method=method, random_state=random_state
    )

    return converter.fit_transform(features_list)
