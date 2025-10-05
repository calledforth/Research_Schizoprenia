"""
Data loading utilities
"""


def load_fmri_data(file_path):
    """
    Load fMRI data from file

    Args:
        file_path (str): Path to fMRI file

    Returns:
        Loaded fMRI data
    """
    pass


def load_meg_data(file_path):
    """
    Load MEG data from file

    Args:
        file_path (str): Path to MEG file

    Returns:
        Loaded MEG data
    """
    pass


def load_subject_data(subject_id, data_root):
    """
    Load all data for a specific subject

    Args:
        subject_id (str): ID of the subject
        data_root (str): Root directory of the data

    Returns:
        Dictionary containing subject's fMRI and MEG data
    """
    pass


def create_data_generator(data_paths, labels, batch_size, shuffle=True):
    """
    Create a data generator for training

    Args:
        data_paths (list): List of data file paths
        labels (list): List of corresponding labels
        batch_size (int): Batch size
        shuffle (bool): Whether to shuffle the data

    Returns:
        Data generator
    """
    pass


def split_data(data_paths, labels, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """
    Split data into train, validation, and test sets

    Args:
        data_paths (list): List of data file paths
        labels (list): List of corresponding labels
        train_ratio (float): Ratio of training data
        val_ratio (float): Ratio of validation data
        test_ratio (float): Ratio of test data

    Returns:
        Split data paths and labels
    """
    pass
