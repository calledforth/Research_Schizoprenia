"""
Data augmentation techniques for neuroimaging data
"""

import numpy as np
import logging
from typing import Tuple, Union, List
from scipy.interpolate import interp1d

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def apply_rotation(data: np.ndarray, angle_range: Tuple[float, float]) -> np.ndarray:
    """
    Apply random rotation to neuroimaging data

    For FNC/SBM features, we simulate rotation by applying a transformation
    that preserves the statistical properties of the connectivity features.

    Args:
        data: Input neuroimaging features (1D array)
        angle_range (tuple): Range of rotation angles in degrees

    Returns:
        Rotated data (1D array)
    """
    if len(data.shape) != 1:
        raise ValueError("Rotation augmentation expects 1D feature array")

    # For 1D features, we simulate rotation by applying a smooth transformation
    # that preserves the overall distribution but slightly changes the values
    angle = np.random.uniform(angle_range[0], angle_range[1])

    # Create a smooth transformation based on the angle
    # This simulates the effect of rotation on connectivity features
    transform_factor = np.sin(np.radians(angle))

    # Apply a non-linear transformation that preserves the statistical properties
    transformed_data = data + transform_factor * np.sin(
        np.linspace(0, np.pi, len(data))
    )

    # Normalize to maintain similar scale
    transformed_data = (transformed_data - np.mean(transformed_data)) / np.std(
        transformed_data
    )
    transformed_data = transformed_data * np.std(data) + np.mean(data)

    return transformed_data


def apply_zoom(data: np.ndarray, zoom_range: Tuple[float, float]) -> np.ndarray:
    """
    Apply random zoom to neuroimaging data

    For FNC/SBM features, we simulate zoom by resampling the feature vector.

    Args:
        data: Input neuroimaging features (1D array)
        zoom_range (tuple): Range of zoom factors

    Returns:
        Zoomed data (1D array)
    """
    if len(data.shape) != 1:
        raise ValueError("Zoom augmentation expects 1D feature array")

    zoom_factor = np.random.uniform(zoom_range[0], zoom_range[1])

    # Create original indices
    original_indices = np.arange(len(data))

    # Create zoomed indices
    zoomed_length = int(len(data) * zoom_factor)
    zoomed_indices = np.linspace(0, len(data) - 1, zoomed_length)

    # Interpolate
    if zoom_factor > 1.0:
        # Zoom in - interpolate to get more points
        f = interp1d(original_indices, data, kind="cubic", fill_value="extrapolate")
        zoomed_data = f(zoomed_indices)
        # Resample back to original length
        f = interp1d(
            np.arange(zoomed_length),
            zoomed_data,
            kind="cubic",
            fill_value="extrapolate",
        )
        result = f(np.linspace(0, zoomed_length - 1, len(data)))
    else:
        # Zoom out - interpolate to get fewer points
        f = interp1d(original_indices, data, kind="cubic", fill_value="extrapolate")
        zoomed_data = f(zoomed_indices)
        # Resample back to original length
        f = interp1d(
            np.arange(zoomed_length),
            zoomed_data,
            kind="cubic",
            fill_value="extrapolate",
        )
        result = f(np.linspace(0, zoomed_length - 1, len(data)))

    return result


def apply_brightness_adjustment(
    data: np.ndarray, brightness_range: Tuple[float, float]
) -> np.ndarray:
    """
    Apply brightness adjustment to neuroimaging data

    For FNC/SBM features, this simulates changes in overall signal intensity.

    Args:
        data: Input neuroimaging features (1D array)
        brightness_range (tuple): Range of brightness adjustment factors

    Returns:
        Brightness-adjusted data (1D array)
    """
    if len(data.shape) != 1:
        raise ValueError("Brightness adjustment expects 1D feature array")

    brightness_factor = np.random.uniform(brightness_range[0], brightness_range[1])

    # Apply brightness adjustment
    adjusted_data = data * brightness_factor

    return adjusted_data


def add_gaussian_noise(
    data: np.ndarray, mean: float = 0.0, std: float = 0.01
) -> np.ndarray:
    """
    Add Gaussian noise to neuroimaging data

    Args:
        data: Input neuroimaging features (1D array)
        mean (float): Mean of the Gaussian noise
        std (float): Standard deviation of the Gaussian noise

    Returns:
        Data with added noise (1D array)
    """
    if len(data.shape) != 1:
        raise ValueError("Noise addition expects 1D feature array")

    # Calculate noise standard deviation relative to data standard deviation
    data_std = np.std(data)
    noise_std = std * data_std

    # Generate and add noise
    noise = np.random.normal(mean, noise_std, size=data.shape)
    noisy_data = data + noise

    return noisy_data


def apply_random_flip(data: np.ndarray, axis: int = 0) -> np.ndarray:
    """
    Apply random flip to neuroimaging data

    For FNC/SBM features, this simulates the effect of flipping by reversing
    the order of features or applying a sign change.

    Args:
        data: Input neuroimaging features (1D array)
        axis (int): Axis along which to flip (for 1D, this determines flip type)

    Returns:
        Flipped data (1D array)
    """
    if len(data.shape) != 1:
        raise ValueError("Flip augmentation expects 1D feature array")

    if np.random.random() > 0.5:
        # Apply flip
        if axis == 0:
            # Reverse the order of features
            flipped_data = data[::-1]
        else:
            # Apply sign change to simulate flipping
            flipped_data = -data
    else:
        # No flip
        flipped_data = data.copy()

    return flipped_data


def apply_feature_dropout(data: np.ndarray, dropout_rate: float = 0.1) -> np.ndarray:
    """
    Apply feature dropout to neuroimaging data

    Randomly sets a fraction of features to zero to simulate missing information.

    Args:
        data: Input neuroimaging features (1D array)
        dropout_rate (float): Fraction of features to drop

    Returns:
        Data with dropout applied (1D array)
    """
    if len(data.shape) != 1:
        raise ValueError("Dropout augmentation expects 1D feature array")

    # Create dropout mask
    mask = np.random.random(size=data.shape) > dropout_rate

    # Apply dropout
    dropped_data = data * mask

    return dropped_data


def apply_time_warping(data: np.ndarray, warp_factor: float = 0.1) -> np.ndarray:
    """
    Apply time warping to neuroimaging data

    Simulates temporal distortions in the signal.

    Args:
        data: Input neuroimaging features (1D array)
        warp_factor (float): Maximum warping factor

    Returns:
        Time-warped data (1D array)
    """
    if len(data.shape) != 1:
        raise ValueError("Time warping expects 1D feature array")

    # Create original indices
    original_indices = np.arange(len(data))

    # Create warped indices
    warp = np.random.uniform(-warp_factor, warp_factor, size=len(data))
    warped_indices = original_indices + warp * len(data)

    # Ensure indices are within bounds
    warped_indices = np.clip(warped_indices, 0, len(data) - 1)

    # Interpolate
    f = interp1d(original_indices, data, kind="cubic", fill_value="extrapolate")
    warped_data = f(warped_indices)

    return warped_data


def augment_features(data: np.ndarray, augmentation_config: dict = None) -> np.ndarray:
    """
    Apply a combination of augmentations to neuroimaging features

    Args:
        data: Input neuroimaging features (1D array)
        augmentation_config (dict): Configuration for augmentations

    Returns:
        Augmented data (1D array)
    """
    if augmentation_config is None:
        # Default configuration
        augmentation_config = {
            "rotation": {"enabled": True, "angle_range": (-5, 5)},
            "zoom": {"enabled": True, "zoom_range": (0.95, 1.05)},
            "brightness": {"enabled": True, "brightness_range": (0.9, 1.1)},
            "noise": {"enabled": True, "mean": 0.0, "std": 0.01},
            "flip": {"enabled": True, "axis": 0},
            "dropout": {"enabled": False, "dropout_rate": 0.1},
            "time_warping": {"enabled": False, "warp_factor": 0.1},
        }

    augmented_data = data.copy()

    # Apply rotation
    if augmentation_config.get("rotation", {}).get("enabled", False):
        angle_range = augmentation_config["rotation"].get("angle_range", (-5, 5))
        augmented_data = apply_rotation(augmented_data, angle_range)

    # Apply zoom
    if augmentation_config.get("zoom", {}).get("enabled", False):
        zoom_range = augmentation_config["zoom"].get("zoom_range", (0.95, 1.05))
        augmented_data = apply_zoom(augmented_data, zoom_range)

    # Apply brightness adjustment
    if augmentation_config.get("brightness", {}).get("enabled", False):
        brightness_range = augmentation_config["brightness"].get(
            "brightness_range", (0.9, 1.1)
        )
        augmented_data = apply_brightness_adjustment(augmented_data, brightness_range)

    # Add Gaussian noise
    if augmentation_config.get("noise", {}).get("enabled", False):
        mean = augmentation_config["noise"].get("mean", 0.0)
        std = augmentation_config["noise"].get("std", 0.01)
        augmented_data = add_gaussian_noise(augmented_data, mean, std)

    # Apply random flip
    if augmentation_config.get("flip", {}).get("enabled", False):
        axis = augmentation_config["flip"].get("axis", 0)
        augmented_data = apply_random_flip(augmented_data, axis)

    # Apply feature dropout
    if augmentation_config.get("dropout", {}).get("enabled", False):
        dropout_rate = augmentation_config["dropout"].get("dropout_rate", 0.1)
        augmented_data = apply_feature_dropout(augmented_data, dropout_rate)

    # Apply time warping
    if augmentation_config.get("time_warping", {}).get("enabled", False):
        warp_factor = augmentation_config["time_warping"].get("warp_factor", 0.1)
        augmented_data = apply_time_warping(augmented_data, warp_factor)

    return augmented_data
