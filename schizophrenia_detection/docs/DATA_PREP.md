# Data Preparation Guide

This guide covers the complete data preparation workflow for the schizophrenia detection pipeline, including directory structure, supported formats, metadata requirements, and data loading examples.

## Table of Contents

1. [Directory Structure](#directory-structure)
2. [Supported Data Formats](#supported-data-formats)
3. [Metadata Requirements](#metadata-requirements)
4. [Data Validation](#data-validation)
5. [Data Loading Examples](#data-loading-examples)
6. [Precomputed Features](#precomputed-features)
7. [Quality Control](#quality-control)
8. [Common Issues](#common-issues)

## Directory Structure

### Google Colab Layout

When using Google Colab, organize your data as follows under `/content/drive/MyDrive/schizophrenia_detection/`:

```
schizophrenia_detection/
├── data/
│   ├── fmri/                    # fMRI data files
│   │   ├── sub-01_func.nii.gz
│   │   ├── sub-02_func.nii.gz
│   │   └── ...
│   ├── meg/                     # MEG data files
│   │   ├── sub-01_meg.fif
│   │   ├── sub-02_meg.fif
│   │   └── ...
│   └── metadata/
│       ├── participants.csv     # Subject metadata
│       ├── scan_info.csv        # Scan parameters
│       └── site_info.csv        # Site-specific information
├── checkpoints/                 # Model checkpoints
├── results/                     # Analysis results
├── visualizations/              # Generated figures
└── logs/                        # Log files
```

### Configuration Paths

Update paths in [`config.py`](../config.py:1) to match your structure:

```python
# In config.py
config.data.data_root = "./data"
config.data.fmri_data_dir = "./data/fmri"
config.data.meg_data_dir = "./data/meg"
```

## Supported Data Formats

### fMRI Data

**Supported Formats:**
- NIfTI (.nii, .nii.gz) - preferred
- Analyze (.hdr, .img) - legacy support

**File Naming Convention:**
```
sub-<subject_id>_task-<task_name>_run-<run_number>_bold.nii.gz
```

Examples:
- `sub-001_task-rest_run-01_bold.nii.gz`
- `sub-002_task-emotion_run-01_bold.nii.gz`

**Required Properties:**
- 4D volume (x, y, z, time)
- BOLD contrast
- Preprocessed or raw data acceptable (pipeline handles preprocessing)

### MEG Data

**Supported Formats:**
- FIF (.fif) - preferred (MNE-Python native)
- CTF (.ds) - CTF systems
- BrainVision (.vhdr, .vmrk, .eeg) - BrainVision systems
- European Data Format (.edf) - limited support

**File Naming Convention:**
```
sub-<subject_id>_task-<task_name>_run-<run_number>_meg.fif
```

**Required Properties:**
- Sensor space data
- Continuous recording
- Contains MEG channels (magnetometers/gradiometers)
- Optional EOG/ECG channels for artifact detection

## Metadata Requirements

### Participants CSV

Create `data/metadata/participants.csv` with the following structure:

```csv
participant_id,age,sex,diagnosis,site,medication,scan_date
sub-001,25,M,SCZ,site_A,antipsychotic,2023-01-15
sub-002,30,F,CTL,site_B,none,2023-01-16
sub-003,28,M,SCZ,site_A,antipsychotic,2023-01-17
```

**Required Columns:**
- `participant_id`: Unique subject identifier (matches file names)
- `diagnosis`: Clinical diagnosis (`SCZ` for schizophrenia, `CTL` for control)
- `age`: Age in years
- `sex`: Biological sex (`M` or `F`)

**Optional Columns:**
- `site`: Scanning site identifier
- `medication`: Medication status
- `scan_date`: Date of scan (YYYY-MM-DD)
- Additional demographic or clinical variables

### Scan Information CSV

Create `data/metadata/scan_info.csv` for scan parameters:

```csv
participant_id,modality,task,run,tr,voxel_size,duration
sub-001,fmri,rest,1,2.0,2.5x2.5x2.5,600
sub-001,meg,rest,1,,,600
sub-002,fmri,rest,1,2.0,2.5x2.5x2.5,600
```

## Data Validation

### Basic Validation Script

```python
import os
import pandas as pd
import nibabel as nib
from schizophrenia_detection.utils.data_utils import validate_data_structure

def validate_dataset(data_root):
    """Validate dataset structure and files"""
    
    # Check directory structure
    required_dirs = ['fmri', 'meg', 'metadata']
    for dir_name in required_dirs:
        dir_path = os.path.join(data_root, dir_name)
        if not os.path.exists(dir_path):
            print(f"Missing directory: {dir_path}")
            return False
    
    # Validate metadata files
    participants_file = os.path.join(data_root, 'metadata', 'participants.csv')
    if not os.path.exists(participants_file):
        print("Missing participants.csv")
        return False
    
    # Load and validate participants
    participants = pd.read_csv(participants_file)
    required_columns = ['participant_id', 'diagnosis', 'age', 'sex']
    for col in required_columns:
        if col not in participants.columns:
            print(f"Missing column in participants.csv: {col}")
            return False
    
    # Validate data files exist
    fmri_dir = os.path.join(data_root, 'fmri')
    meg_dir = os.path.join(data_root, 'meg')
    
    for _, subject in participants.iterrows():
        subject_id = subject['participant_id']
        
        # Check fMRI file
        fmri_file = os.path.join(fmri_dir, f"{subject_id}_func.nii.gz")
        if not os.path.exists(fmri_file):
            print(f"Missing fMRI file for {subject_id}")
            return False
        
        # Check MEG file
        meg_file = os.path.join(meg_dir, f"{subject_id}_meg.fif")
        if not os.path.exists(meg_file):
            print(f"Missing MEG file for {subject_id}")
            return False
    
    print("Dataset validation passed!")
    return True

# Usage
validate_dataset("./data")
```

### fMRI Quality Checks

```python
def validate_fmri_data(file_path):
    """Validate fMRI data quality"""
    try:
        img = nib.load(file_path)
        data = img.get_fdata()
        
        # Check dimensions
        if len(data.shape) != 4:
            print(f"Expected 4D data, got {len(data.shape)}D")
            return False
        
        # Check for reasonable values
        if np.any(np.isnan(data)):
            print("fMRI data contains NaN values")
            return False
        
        # Check temporal dimension
        if data.shape[3] < 50:
            print(f"Too few time points: {data.shape[3]}")
            return False
        
        print(f"fMRI validation passed: {data.shape}")
        return True
        
    except Exception as e:
        print(f"fMRI validation error: {e}")
        return False
```

### MEG Quality Checks

```python
import mne

def validate_meg_data(file_path):
    """Validate MEG data quality"""
    try:
        raw = mne.io.read_raw_fif(file_path, preload=False)
        
        # Check for MEG channels
        meg_channels = mne.pick_types(raw.info, meg=True)
        if len(meg_channels) == 0:
            print("No MEG channels found")
            return False
        
        # Check recording duration
        duration = raw.times[-1]
        if duration < 300:  # Less than 5 minutes
            print(f"Short recording: {duration:.1f} seconds")
            return False
        
        # Check sampling rate
        sfreq = raw.info['sfreq']
        if sfreq < 100:
            print(f"Low sampling rate: {sfreq} Hz")
            return False
        
        print(f"MEG validation passed: {len(meg_channels)} channels, {duration:.1f}s")
        return True
        
    except Exception as e:
        print(f"MEG validation error: {e}")
        return False
```

## Data Loading Examples

### Loading Single Subject Data

```python
from schizophrenia_detection.data_processing.data_loader import load_subject_data

# Load all data for a single subject
subject_data = load_subject_data(
    subject_id="sub-001",
    data_root="./data"
)

print(f"Loaded subject data keys: {list(subject_data.keys())}")
print(f"fMRI shape: {subject_data['fmri'].shape}")
print(f"MEG shape: {subject_data['meg'].shape}")
```

### Loading Multiple Subjects

```python
from schizophrenia_detection.data_processing.data_loader import load_fmri_data, load_meg_data
import pandas as pd

# Load participants list
participants = pd.read_csv("./data/metadata/participants.csv")

# Load all fMRI data
fmri_data = {}
for subject_id in participants['participant_id']:
    fmri_path = f"./data/fmri/{subject_id}_func.nii.gz"
    fmri_data[subject_id] = load_fmri_data(fmri_path)

# Load all MEG data
meg_data = {}
for subject_id in participants['participant_id']:
    meg_path = f"./data/meg/{subject_id}_meg.fif"
    meg_data[subject_id] = load_meg_data(meg_path)

print(f"Loaded {len(fmri_data)} fMRI datasets")
print(f"Loaded {len(meg_data)} MEG datasets")
```

### Creating Data Generators

```python
from schizophrenia_detection.data_processing.data_loader import create_data_generator

# Prepare data paths and labels
data_paths = [
    f"./data/fmri/{sid}_func.nii.gz" 
    for sid in participants['participant_id']
]
labels = participants['diagnosis'].map({'SCZ': 1, 'CTL': 0}).tolist()

# Create training generator
train_generator = create_data_generator(
    data_paths=data_paths,
    labels=labels,
    batch_size=4,
    shuffle=True
)

# Use in training
for batch_x, batch_y in train_generator:
    print(f"Batch shape: {batch_x.shape}, Labels: {batch_y}")
    break
```

### Data Splitting

```python
from schizophrenia_detection.data_processing.data_loader import split_data

# Split data into train/val/test
train_paths, val_paths, test_paths, train_labels, val_labels, test_labels = split_data(
    data_paths=data_paths,
    labels=labels,
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15
)

print(f"Training: {len(train_paths)} subjects")
print(f"Validation: {len(val_paths)} subjects")
print(f"Test: {len(test_paths)} subjects")
```

## Precomputed Features

### Functional Network Connectivity (FNC)

If you have precomputed FNC matrices:

```python
import numpy as np

def load_fnc_matrix(subject_id, data_root):
    """Load precomputed FNC matrix"""
    fnc_path = os.path.join(data_root, 'features', 'fnc', f"{subject_id}_fnc.npy")
    if os.path.exists(fnc_path):
        return np.load(fnc_path)
    return None

# Example usage
fnc_matrix = load_fnc_matrix("sub-001", "./data")
if fnc_matrix is not None:
    print(f"FNC matrix shape: {fnc_matrix.shape}")
```

### Source-Based Morphometry (SBM)

If you have SBM features:

```python
def load_sbm_features(subject_id, data_root):
    """Load precomputed SBM features"""
    sbm_path = os.path.join(data_root, 'features', 'sbm', f"{subject_id}_sbm.npy")
    if os.path.exists(sbm_path):
        return np.load(sbm_path)
    return None

# Example usage
sbm_features = load_sbm_features("sub-001", "./data")
if sbm_features is not None:
    print(f"SBM features shape: {sbm_features.shape}")
```

### Integrating Precomputed Features

```python
def load_subject_with_features(subject_id, data_root):
    """Load subject data with optional precomputed features"""
    
    # Load raw data
    subject_data = load_subject_data(subject_id, data_root)
    
    # Add precomputed features if available
    fnc_matrix = load_fnc_matrix(subject_id, data_root)
    if fnc_matrix is not None:
        subject_data['fnc'] = fnc_matrix
    
    sbm_features = load_sbm_features(subject_id, data_root)
    if sbm_features is not None:
        subject_data['sbm'] = sbm_features
    
    return subject_data
```

## Quality Control

### Visual Inspection

```python
import matplotlib.pyplot as plt
from nilearn import plotting

def visualize_fmri_slice(fmri_path, slice_idx=50):
    """Visualize a single slice of fMRI data"""
    img = nib.load(fmri_path)
    data = img.get_fdata()
    
    plt.figure(figsize=(10, 8))
    plt.imshow(data[:, :, slice_idx, 0], cmap='gray')
    plt.title(f"fMRI Slice {slice_idx}")
    plt.axis('off')
    plt.show()

# Usage
visualize_fmri_slice("./data/fmri/sub-001_func.nii.gz")
```

### Signal Quality Metrics

```python
def calculate_fmri_quality_metrics(fmri_path):
    """Calculate basic fMRI quality metrics"""
    img = nib.load(fmri_path)
    data = img.get_fdata()
    
    # Calculate metrics
    mean_signal = np.mean(data)
    std_signal = np.std(data)
    snr = mean_signal / std_signal
    
    # Temporal SNR
    temporal_mean = np.mean(data, axis=3)
    temporal_std = np.std(data, axis=3)
    temporal_snr = np.mean(temporal_mean / temporal_std)
    
    metrics = {
        'mean_signal': mean_signal,
        'std_signal': std_signal,
        'snr': snr,
        'temporal_snr': temporal_snr
    }
    
    return metrics

# Usage
metrics = calculate_fmri_quality_metrics("./data/fmri/sub-001_func.nii.gz")
print(f"fMRI Quality Metrics: {metrics}")
```

### MEG Quality Metrics

```python
def calculate_meg_quality_metrics(meg_path):
    """Calculate basic MEG quality metrics"""
    raw = mne.io.read_raw_fif(meg_path, preload=True)
    
    # Pick MEG channels
    meg_channels = mne.pick_types(raw.info, meg=True)
    meg_data = raw.get_data(picks=meg_channels)
    
    # Calculate metrics
    peak_to_peak = np.max(meg_data, axis=1) - np.min(meg_data, axis=1)
    rms = np.sqrt(np.mean(meg_data**2, axis=1))
    
    metrics = {
        'peak_to_peak_mean': np.mean(peak_to_peak),
        'peak_to_peak_std': np.std(peak_to_peak),
        'rms_mean': np.mean(rms),
        'rms_std': np.std(rms)
    }
    
    return metrics

# Usage
metrics = calculate_meg_quality_metrics("./data/meg/sub-001_meg.fif")
print(f"MEG Quality Metrics: {metrics}")
```

## Common Issues

### Missing Files

**Problem**: Data files don't match expected naming convention

**Solution**:
```python
import os
import glob

def find_data_files(data_root, pattern):
    """Find files matching a pattern"""
    search_path = os.path.join(data_root, pattern)
    return glob.glob(search_path)

# Find all fMRI files
fmri_files = find_data_files("./data/fmri", "*.nii*")
print(f"Found {len(fmri_files)} fMRI files")

# Find all MEG files
meg_files = find_data_files("./data/meg", "*.fif")
print(f"Found {len(meg_files)} MEG files")
```

### Format Conversion

**Problem**: Data in unsupported format

**Solution**: Use appropriate conversion tools

```python
# Convert DICOM to NIfTI (requires dcm2niix)
import subprocess

def convert_dicom_to_nifti(dicom_dir, output_path):
    """Convert DICOM series to NIfTI"""
    cmd = f"dcm2niix -z y -o {os.path.dirname(output_path)} -f {os.path.splitext(os.path.basename(output_path))[0]} {dicom_dir}"
    subprocess.run(cmd, shell=True)

# Usage
# convert_dicom_to_nifti("./dicom/sub-001", "./data/fmri/sub-001_func.nii.gz")
```

### Memory Issues

**Problem**: Large datasets cause memory errors

**Solution**: Use data generators and chunked loading

```python
def load_fmri_chunked(file_path, chunk_size=10):
    """Load fMRI data in chunks"""
    img = nib.load(file_path)
    data = img.get_fdata()
    
    n_timepoints = data.shape[3]
    for i in range(0, n_timepoints, chunk_size):
        chunk = data[..., i:i+chunk_size]
        yield chunk

# Usage
for chunk in load_fmri_chunked("./data/fmri/sub-001_func.nii.gz"):
    print(f"Processing chunk of shape: {chunk.shape}")
    # Process chunk
```

### Data Corruption

**Problem**: Corrupted or incomplete files

**Solution**: Validate files before processing

```python
def validate_file_integrity(file_path):
    """Check if file is corrupted"""
    try:
        if file_path.endswith('.nii.gz'):
            img = nib.load(file_path)
            data = img.get_fdata()
            return not np.any(np.isnan(data))
        elif file_path.endswith('.fif'):
            raw = mne.io.read_raw_fif(file_path, preload=False)
            return True
        else:
            return False
    except Exception as e:
        print(f"File validation error: {e}")
        return False

# Usage
is_valid = validate_file_integrity("./data/fmri/sub-001_func.nii.gz")
print(f"File valid: {is_valid}")
```

## Next Steps

After preparing your data:

1. Run data validation scripts to ensure quality
2. Execute [data_exploration.ipynb](../notebooks/data_exploration.ipynb:1) to understand your dataset
3. Follow preprocessing guides:
   - [FMRI_PREPROCESSING.md](FMRI_PREPROCESSING.md:1)
   - [MEG_PREPROCESSING.md](MEG_PREPROCESSING.md:1)
4. Proceed to model training using [model_training.ipynb](../notebooks/model_training.ipynb:1)

For additional help, see the [FAQ.md](FAQ.md:1) or open an issue on the GitHub repository.