"""
Data processing utilities
"""

import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split


def normalize_data(data, method="standard"):
    """
    Normalize data using various methods

    Args:
        data: Input data
        method (str): Normalization method ('standard', 'minmax', 'robust', 'unit')

    Returns:
        Normalized data
    """
    if method == "standard":
        scaler = StandardScaler()
    elif method == "minmax":
        scaler = MinMaxScaler()
    elif method == "robust":
        from sklearn.preprocessing import RobustScaler

        scaler = RobustScaler()
    elif method == "unit":
        # Unit norm scaling
        norm = np.linalg.norm(data, axis=-1, keepdims=True)
        return data / (norm + 1e-8)
    else:
        raise ValueError(f"Unknown normalization method: {method}")

    # Reshape data for sklearn if needed
    original_shape = data.shape
    if len(data.shape) > 2:
        data = data.reshape(data.shape[0], -1)

    # Fit and transform
    normalized_data = scaler.fit_transform(data)

    # Reshape back to original shape
    if len(original_shape) > 2:
        normalized_data = normalized_data.reshape(original_shape)

    return normalized_data


def denoise_data(data, method="gaussian", **kwargs):
    """
    Denoise data using various methods

    Args:
        data: Input data
        method (str): Denoising method ('gaussian', 'median', 'bilateral', 'wavelet')
        **kwargs: Additional parameters for denoising

    Returns:
        Denoised data
    """
    if method == "gaussian":
        from scipy.ndimage import gaussian_filter

        sigma = kwargs.get("sigma", 1.0)
        return gaussian_filter(data, sigma=sigma)
    elif method == "median":
        from scipy.ndimage import median_filter

        size = kwargs.get("size", 3)
        return median_filter(data, size=size)
    elif method == "bilateral":
        # This would require additional implementation
        raise NotImplementedError("Bilateral filter not implemented yet")
    elif method == "wavelet":
        import pywt

        wavelet = kwargs.get("wavelet", "db1")
        mode = kwargs.get("mode", "symmetric")

        # Decompose
        coeffs = pywt.wavedecn(data, wavelet=wavelet, mode=mode)

        # Threshold coefficients
        threshold = kwargs.get("threshold", 0.1)
        coeffs_thresh = list(coeffs)
        coeffs_thresh[1:] = [
            {
                key: pywt.threshold(detail, threshold, mode="soft")
                for key, detail in level.items()
            }
            for level in coeffs_thresh[1:]
        ]

        # Reconstruct
        return pywt.waverecn(coeffs_thresh, wavelet=wavelet, mode=mode)
    else:
        raise ValueError(f"Unknown denoising method: {method}")


def extract_patches(data, patch_size, stride=None):
    """
    Extract patches from data

    Args:
        data: Input data
        patch_size (tuple): Size of patches to extract
        stride (tuple): Stride for patch extraction

    Returns:
        Extracted patches
    """
    if stride is None:
        stride = patch_size

    patches = []

    if len(data.shape) == 3:  # 3D data
        x, y, z = data.shape
        px, py, pz = patch_size
        sx, sy, sz = stride

        for i in range(0, x - px + 1, sx):
            for j in range(0, y - py + 1, sy):
                for k in range(0, z - pz + 1, sz):
                    patch = data[i : i + px, j : j + py, k : k + pz]
                    patches.append(patch)
    elif len(data.shape) == 4:  # 4D data
        x, y, z, t = data.shape
        px, py, pz = patch_size
        sx, sy, sz = stride

        for i in range(0, x - px + 1, sx):
            for j in range(0, y - py + 1, sy):
                for k in range(0, z - pz + 1, sz):
                    for l in range(t):
                        patch = data[i : i + px, j : j + py, k : k + pz, l]
                        patches.append(patch)

    return np.array(patches)


def reconstruct_from_patches(patches, output_shape, patch_size, stride=None):
    """
    Reconstruct data from patches

    Args:
        patches: Array of patches
        output_shape (tuple): Shape of the output data
        patch_size (tuple): Size of patches
        stride (tuple): Stride used for patch extraction

    Returns:
        Reconstructed data
    """
    if stride is None:
        stride = patch_size

    # Initialize output and count arrays
    output = np.zeros(output_shape)
    count = np.zeros(output_shape)

    # Place patches back
    idx = 0
    if len(output_shape) == 3:  # 3D data
        x, y, z = output_shape
        px, py, pz = patch_size
        sx, sy, sz = stride

        for i in range(0, x - px + 1, sx):
            for j in range(0, y - py + 1, sy):
                for k in range(0, z - pz + 1, sz):
                    if idx < len(patches):
                        output[i : i + px, j : j + py, k : k + pz] += patches[idx]
                        count[i : i + px, j : j + py, k : k + pz] += 1
                        idx += 1
    elif len(output_shape) == 4:  # 4D data
        x, y, z, t = output_shape
        px, py, pz = patch_size
        sx, sy, sz = stride

        for i in range(0, x - px + 1, sx):
            for j in range(0, y - py + 1, sy):
                for k in range(0, z - pz + 1, sz):
                    for l in range(t):
                        if idx < len(patches):
                            output[i : i + px, j : j + py, k : k + pz, l] += patches[
                                idx
                            ]
                            count[i : i + px, j : j + py, k : k + pz, l] += 1
                            idx += 1

    # Normalize by count
    output = output / (count + 1e-8)

    return output


def resize_data(data, target_shape, method="linear"):
    """
    Resize data to target shape

    Args:
        data: Input data
        target_shape (tuple): Target shape
        method (str): Interpolation method

    Returns:
        Resized data
    """
    from scipy.ndimage import zoom

    # Calculate zoom factors
    zoom_factors = [t / s for t, s in zip(target_shape, data.shape)]

    # Apply zoom
    if method == "linear":
        order = 1
    elif method == "nearest":
        order = 0
    elif method == "cubic":
        order = 3
    else:
        raise ValueError(f"Unknown interpolation method: {method}")

    return zoom(data, zoom_factors, order=order)


def split_data_by_subject(data, labels, subject_ids, test_ratio=0.2, random_state=42):
    """
    Split data by subject to avoid data leakage

    Args:
        data: Input data
        labels: Input labels
        subject_ids: Subject IDs for each sample
        test_ratio (float): Ratio of test data
        random_state (int): Random seed

    Returns:
        Split data and labels
    """
    # Get unique subjects
    unique_subjects = np.unique(subject_ids)

    # Split subjects
    train_subjects, test_subjects = train_test_split(
        unique_subjects, test_size=test_ratio, random_state=random_state
    )

    # Create masks
    train_mask = np.isin(subject_ids, train_subjects)
    test_mask = np.isin(subject_ids, test_subjects)

    # Split data
    train_data = data[train_mask]
    test_data = data[test_mask]
    train_labels = labels[train_mask]
    test_labels = labels[test_mask]

    return train_data, test_data, train_labels, test_labels


def balance_data(data, labels, method="oversample"):
    """
    Balance imbalanced dataset

    Args:
        data: Input data
        labels: Input labels
        method (str): Balancing method ('oversample', 'undersample', 'smote')

    Returns:
        Balanced data and labels
    """
    from collections import Counter

    label_counts = Counter(labels)
    max_count = max(label_counts.values())

    if method == "oversample":
        balanced_data = []
        balanced_labels = []

        for label, count in label_counts.items():
            # Get all samples of this class
            class_data = data[labels == label]

            # Calculate how many samples to add
            samples_to_add = max_count - count

            # Randomly sample with replacement
            if samples_to_add > 0:
                indices = np.random.choice(
                    range(len(class_data)), size=samples_to_add, replace=True
                )
                additional_samples = class_data[indices]

                # Add original and additional samples
                balanced_data.append(class_data)
                balanced_data.append(additional_samples)
                balanced_labels.extend([label] * (count + samples_to_add))
            else:
                balanced_data.append(class_data)
                balanced_labels.extend([label] * count)

        return np.vstack(balanced_data), np.array(balanced_labels)

    elif method == "undersample":
        balanced_data = []
        balanced_labels = []

        for label, count in label_counts.items():
            # Get all samples of this class
            class_data = data[labels == label]

            # Randomly sample without replacement
            indices = np.random.choice(
                range(len(class_data)), size=max_count, replace=False
            )
            sampled_data = class_data[indices]

            balanced_data.append(sampled_data)
            balanced_labels.extend([label] * max_count)

        return np.vstack(balanced_data), np.array(balanced_labels)

    elif method == "smote":
        from imblearn.over_sampling import SMOTE

        # Reshape data for SMOTE if needed
        original_shape = data.shape
        if len(data.shape) > 2:
            data = data.reshape(data.shape[0], -1)

        # Apply SMOTE
        smote = SMOTE(random_state=42)
        data_resampled, labels_resampled = smote.fit_resample(data, labels)

        # Reshape back to original shape
        if len(original_shape) > 2:
            data_resampled = data_resampled.reshape(-1, *original_shape[1:])

        return data_resampled, labels_resampled

    else:
        raise ValueError(f"Unknown balancing method: {method}")


def encode_labels(labels, method="label"):
    """
    Encode labels

    Args:
        labels: Input labels
        method (str): Encoding method ('label', 'onehot')

    Returns:
        Encoded labels
    """
    if method == "label":
        encoder = LabelEncoder()
        return encoder.fit_transform(labels)
    elif method == "onehot":
        from sklearn.preprocessing import OneHotEncoder

        encoder = OneHotEncoder(sparse=False)
        return encoder.fit_transform(labels.reshape(-1, 1))
    else:
        raise ValueError(f"Unknown encoding method: {method}")


def calculate_data_statistics(data):
    """
    Calculate statistics of data

    Args:
        data: Input data

    Returns:
        Dictionary of statistics
    """
    return {
        "mean": np.mean(data),
        "std": np.std(data),
        "min": np.min(data),
        "max": np.max(data),
        "median": np.median(data),
        "q25": np.percentile(data, 25),
        "q75": np.percentile(data, 75),
        "shape": data.shape,
        "dtype": data.dtype,
    }
