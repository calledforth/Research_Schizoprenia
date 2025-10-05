"""
File handling utilities
"""

import os
import json
import pickle
import numpy as np
import nibabel as nib
from pathlib import Path


def ensure_dir(directory):
    """
    Ensure that a directory exists

    Args:
        directory (str): Directory path
    """
    Path(directory).mkdir(parents=True, exist_ok=True)


def load_nifti(file_path):
    """
    Load a NIfTI file

    Args:
        file_path (str): Path to the NIfTI file

    Returns:
        NIfTI image object
    """
    return nib.load(file_path)


def save_nifti(data, file_path, affine=None):
    """
    Save data as a NIfTI file

    Args:
        data: Data to save
        file_path (str): Path to save the NIfTI file
        affine: Affine matrix for the NIfTI image
    """
    if affine is None:
        affine = np.eye(4)

    img = nib.Nifti1Image(data, affine)
    nib.save(img, file_path)


def load_json(file_path):
    """
    Load data from a JSON file

    Args:
        file_path (str): Path to the JSON file

    Returns:
        Loaded data
    """
    with open(file_path, "r") as f:
        return json.load(f)


def save_json(data, file_path):
    """
    Save data to a JSON file

    Args:
        data: Data to save
        file_path (str): Path to save the JSON file
    """
    with open(file_path, "w") as f:
        json.dump(data, f, indent=2)


def load_pickle(file_path):
    """
    Load data from a pickle file

    Args:
        file_path (str): Path to the pickle file

    Returns:
        Loaded data
    """
    with open(file_path, "rb") as f:
        return pickle.load(f)


def save_pickle(data, file_path):
    """
    Save data to a pickle file

    Args:
        data: Data to save
        file_path (str): Path to save the pickle file
    """
    with open(file_path, "wb") as f:
        pickle.dump(data, f)


def list_files(directory, pattern=None, recursive=False):
    """
    List files in a directory

    Args:
        directory (str): Directory path
        pattern (str): File pattern to match
        recursive (bool): Whether to search recursively

    Returns:
        List of file paths
    """
    if recursive:
        files = []
        for root, _, filenames in os.walk(directory):
            for filename in filenames:
                if pattern is None or filename.endswith(pattern):
                    files.append(os.path.join(root, filename))
        return files
    else:
        if pattern is None:
            return [
                os.path.join(directory, f)
                for f in os.listdir(directory)
                if os.path.isfile(os.path.join(directory, f))
            ]
        else:
            return [
                os.path.join(directory, f)
                for f in os.listdir(directory)
                if os.path.isfile(os.path.join(directory, f)) and f.endswith(pattern)
            ]


def get_file_size(file_path):
    """
    Get the size of a file in MB

    Args:
        file_path (str): Path to the file

    Returns:
        File size in MB
    """
    size_bytes = os.path.getsize(file_path)
    return size_bytes / (1024 * 1024)


def check_file_exists(file_path):
    """
    Check if a file exists

    Args:
        file_path (str): Path to the file

    Returns:
        bool: True if file exists, False otherwise
    """
    return os.path.isfile(file_path)


def get_file_extension(file_path):
    """
    Get the extension of a file

    Args:
        file_path (str): Path to the file

    Returns:
        str: File extension
    """
    return os.path.splitext(file_path)[1]


def create_symlink(source, target):
    """
    Create a symbolic link

    Args:
        source (str): Source path
        target (str): Target path
    """
    os.symlink(source, target)


def copy_file(source, target):
    """
    Copy a file

    Args:
        source (str): Source path
        target (str): Target path
    """
    import shutil

    shutil.copy2(source, target)


def move_file(source, target):
    """
    Move a file

    Args:
        source (str): Source path
        target (str): Target path
    """
    import shutil

    shutil.move(source, target)


def delete_file(file_path):
    """
    Delete a file

    Args:
        file_path (str): Path to the file
    """
    os.remove(file_path)


def get_absolute_path(file_path):
    """
    Get the absolute path of a file

    Args:
        file_path (str): Path to the file

    Returns:
        str: Absolute path
    """
    return os.path.abspath(file_path)


def get_relative_path(file_path, base_path):
    """
    Get the relative path of a file with respect to a base path

    Args:
        file_path (str): Path to the file
        base_path (str): Base path

    Returns:
        str: Relative path
    """
    return os.path.relpath(file_path, base_path)


def join_paths(*paths):
    """
    Join multiple path components

    Args:
        *paths: Path components

    Returns:
        str: Joined path
    """
    return os.path.join(*paths)


def split_path(file_path):
    """
    Split a path into directory, filename, and extension

    Args:
        file_path (str): Path to split

    Returns:
        tuple: (directory, filename, extension)
    """
    directory = os.path.dirname(file_path)
    filename = os.path.basename(file_path)
    name, ext = os.path.splitext(filename)
    return directory, name, ext
