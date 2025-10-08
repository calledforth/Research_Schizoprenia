"""
Data loading utilities for MLSP 2014 schizophrenia detection dataset
"""

import os
import csv
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Generator, Any
from sklearn.model_selection import train_test_split

# Import feature to 3D conversion
from .feature_to_3d import FeatureTo3DConverter, convert_features_to_3d

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_fmri_data(file_path: str) -> Dict[str, np.ndarray]:
    """
    Load FNC (Functional Network Connectivity) features from CSV file

    Args:
        file_path (str): Path to FNC CSV file

    Returns:
        Dictionary with subject IDs as keys and FNC features as values
    """
    logger.info(f"Loading FNC data from {file_path}")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"FNC file not found: {file_path}")

    fnc_data = {}

    with open(file_path, "r") as f:
        reader = csv.reader(f)
        header = next(reader)  # Skip header

        for row in reader:
            subject_id = row[0]
            features = np.array([float(val) for val in row[1:]])
            fnc_data[subject_id] = features

    logger.info(f"Loaded FNC data for {len(fnc_data)} subjects")
    return fnc_data


def load_meg_data(file_path: str) -> Dict[str, np.ndarray]:
    """
    Load SBM (Source-Based Morphometry) features from CSV file

    Args:
        file_path (str): Path to SBM CSV file

    Returns:
        Dictionary with subject IDs as keys and SBM features as values
    """
    logger.info(f"Loading SBM data from {file_path}")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"SBM file not found: {file_path}")

    sbm_data = {}

    with open(file_path, "r") as f:
        reader = csv.reader(f)
        header = next(reader)  # Skip header

        for row in reader:
            subject_id = row[0]
            features = np.array([float(val) for val in row[1:]])
            sbm_data[subject_id] = features

    logger.info(f"Loaded SBM data for {len(sbm_data)} subjects")
    return sbm_data


def load_labels(file_path: str) -> Dict[str, int]:
    """
    Load subject labels from CSV file

    Args:
        file_path (str): Path to labels CSV file

    Returns:
        Dictionary with subject IDs as keys and labels as values
    """
    logger.info(f"Loading labels from {file_path}")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Labels file not found: {file_path}")

    labels = {}

    with open(file_path, "r") as f:
        reader = csv.reader(f)
        header = next(reader)  # Skip header

        for row in reader:
            subject_id = row[0]
            label = int(row[1])
            labels[subject_id] = label

    logger.info(f"Loaded labels for {len(labels)} subjects")
    return labels


def load_subject_data(subject_id: str, data_root: str) -> Dict[str, Any]:
    """
    Load all data for a specific subject

    Args:
        subject_id (str): ID of the subject
        data_root (str): Root directory of the data

    Returns:
        Dictionary containing subject's FNC and SBM features
    """
    # Load FNC data
    fnc_file = os.path.join(data_root, "Train", "train_FNC.csv")
    fnc_data = load_fmri_data(fnc_file)

    # Load SBM data
    sbm_file = os.path.join(data_root, "Train", "train_SBM.csv")
    sbm_data = load_meg_data(sbm_file)

    # Load labels
    labels_file = os.path.join(data_root, "Train", "train_labels.csv")
    labels = load_labels(labels_file)

    if subject_id not in fnc_data:
        raise ValueError(f"Subject {subject_id} not found in FNC data")

    if subject_id not in sbm_data:
        raise ValueError(f"Subject {subject_id} not found in SBM data")

    if subject_id not in labels:
        raise ValueError(f"Subject {subject_id} not found in labels")

    return {
        "subject_id": subject_id,
        "fnc_features": fnc_data[subject_id],
        "sbm_features": sbm_data[subject_id],
        "label": labels[subject_id],
    }


def load_all_subjects(data_root: str) -> List[Dict[str, Any]]:
    """
    Load all subjects data

    Args:
        data_root (str): Root directory of the data

    Returns:
        List of dictionaries containing subject data
    """
    # Load FNC data
    fnc_file = os.path.join(data_root, "Train", "train_FNC.csv")
    fnc_data = load_fmri_data(fnc_file)

    # Load SBM data
    sbm_file = os.path.join(data_root, "Train", "train_SBM.csv")
    sbm_data = load_meg_data(sbm_file)

    # Load labels
    labels_file = os.path.join(data_root, "Train", "train_labels.csv")
    labels = load_labels(labels_file)

    # Find common subject IDs
    common_subjects = set(fnc_data.keys()) & set(sbm_data.keys()) & set(labels.keys())

    subjects_data = []
    for subject_id in common_subjects:
        subjects_data.append(
            {
                "subject_id": subject_id,
                "fnc_features": fnc_data[subject_id],
                "sbm_features": sbm_data[subject_id],
                "label": labels[subject_id],
            }
        )

    logger.info(f"Loaded data for {len(subjects_data)} subjects")
    return subjects_data


def create_data_generator(
    data: List[Dict[str, Any]], batch_size: int, shuffle: bool = True
) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
    """
    Create a data generator for training

    Args:
        data (list): List of subject data dictionaries
        batch_size (int): Batch size
        shuffle (bool): Whether to shuffle the data

    Yields:
        Tuple of (features, labels) for each batch
    """
    num_samples = len(data)
    indices = np.arange(num_samples)

    if shuffle:
        np.random.shuffle(indices)

    while True:
        for start_idx in range(0, num_samples, batch_size):
            end_idx = min(start_idx + batch_size, num_samples)
            batch_indices = indices[start_idx:end_idx]

            batch_features = []
            batch_labels = []

            for idx in batch_indices:
                # Combine FNC and SBM features
                fnc_features = data[idx]["fnc_features"]
                sbm_features = data[idx]["sbm_features"]
                combined_features = np.concatenate([fnc_features, sbm_features])

                batch_features.append(combined_features)
                batch_labels.append(data[idx]["label"])

            yield np.array(batch_features), np.array(batch_labels)


def create_3d_data_generator(
    data: List[Dict[str, Any]],
    batch_size: int,
    shuffle: bool = True,
    conversion_method: str = "mds",
) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
    """
    Create a 3D data generator for training with SSPNet model

    Args:
        data (list): List of subject data dictionaries
        batch_size (int): Batch size
        shuffle (bool): Whether to shuffle the data
        conversion_method (str): Method for 3D conversion ('mds', 'pca', 'graph', 'autoencoder')

    Yields:
        Tuple of (3D features, labels) for each batch
    """
    num_samples = len(data)
    indices = np.arange(num_samples)

    if shuffle:
        np.random.shuffle(indices)

    # Get all features for fitting the converter
    all_features = []
    for subject in data:
        fnc_features = subject["fnc_features"]
        sbm_features = subject["sbm_features"]
        combined_features = np.concatenate([fnc_features, sbm_features])
        all_features.append(combined_features)

    # Initialize and fit converter
    converter = FeatureTo3DConverter(method=conversion_method)
    converter.fit(all_features)

    while True:
        for start_idx in range(0, num_samples, batch_size):
            end_idx = min(start_idx + batch_size, num_samples)
            batch_indices = indices[start_idx:end_idx]

            batch_features = []
            batch_labels = []

            for idx in batch_indices:
                # Combine FNC and SBM features
                fnc_features = data[idx]["fnc_features"]
                sbm_features = data[idx]["sbm_features"]
                combined_features = np.concatenate([fnc_features, sbm_features])

                # Convert to 3D representation
                features_3d = converter.transform(combined_features)
                batch_features.append(features_3d)
                batch_labels.append(data[idx]["label"])

            yield np.array(batch_features), np.array(batch_labels)


def split_data(
    data: List[Dict[str, Any]],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Split data into train, validation, and test sets

    Args:
        data (list): List of subject data dictionaries
        train_ratio (float): Ratio of training data
        val_ratio (float): Ratio of validation data
        test_ratio (float): Ratio of test data
        random_state (int): Random seed for reproducibility

    Returns:
        Tuple of (train_data, val_data, test_data)
    """
    # Validate ratios
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("Train, validation, and test ratios must sum to 1.0")

    # First split: train + val vs test
    train_val_data, test_data = train_test_split(
        data,
        test_size=test_ratio,
        random_state=random_state,
        stratify=[d["label"] for d in data],
    )

    # Second split: train vs val
    val_size_adjusted = val_ratio / (train_ratio + val_ratio)
    train_data, val_data = train_test_split(
        train_val_data,
        test_size=val_size_adjusted,
        random_state=random_state,
        stratify=[d["label"] for d in train_val_data],
    )

    logger.info(
        f"Data split: {len(train_data)} train, {len(val_data)} validation, {len(test_data)} test"
    )
    return train_data, val_data, test_data
