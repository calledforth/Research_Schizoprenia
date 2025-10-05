# fMRI Preprocessing Guide

This guide covers the complete fMRI preprocessing pipeline for schizophrenia detection, including motion correction, slice timing correction, spatial smoothing, coregistration, normalization, and SSP map generation.

## Table of Contents

1. [Overview](#overview)
2. [Preprocessing Pipeline](#preprocessing-pipeline)
3. [Step-by-Step Processing](#step-by-step-processing)
4. [SSP Map Generation](#ssp-map-generation)
5. [Quality Control](#quality-control)
6. [Code Examples](#code-examples)
7. [Configuration](#configuration)
8. [Troubleshooting](#troubleshooting)

## Overview

The fMRI preprocessing pipeline implements a comprehensive workflow that transforms raw fMRI data into normalized, quality-controlled volumes ready for SSPNet 3D CNN training. The pipeline follows standard neuroimaging best practices with specialized components for schizophrenia detection.

### Key Features

- **Motion Correction**: Rigid-body registration to correct head motion
- **Slice Timing Correction**: Accounts for interleaved slice acquisition
- **Spatial Smoothing**: Gaussian smoothing for noise reduction
- **Coregistration**: Alignment to structural images
- **Normalization**: Transformation to MNI standard space
- **SPM Analysis**: Statistical Parametric Mapping for activation detection
- **SSP Maps**: Spatial Source Phase maps for specialized analysis

### Input Requirements

- **Format**: NIfTI (.nii, .nii.gz)
- **Dimensions**: 4D (x, y, z, time)
- **Voxel Size**: Recommended 2-3mm isotropic
- **TR**: 2.0 seconds (default, configurable)
- **Duration**: Minimum 5 minutes recommended

## Preprocessing Pipeline

### Pipeline Architecture

```
Raw fMRI → Motion Correction → Slice Timing → Spatial Smoothing → Coregistration → Normalization → [SPM/SSP] → Output
```

### Implementation Location

The complete pipeline is implemented in [`fmri_preprocessing.py`](../data_processing/fmri_preprocessing.py:1) with the main function:

```python
preprocess_fmri_pipeline(
    fmri_img, struct_img=None, tr=2.0, fwhm=6.0, 
    slice_order="ascending", perform_normalization=True,
    perform_spm=False, design_matrix=None, contrast_matrix=None,
    generate_ssp=False, output_dir=None
)
```

## Step-by-Step Processing

### 1. Motion Correction and Realignment

**Purpose**: Correct for head movement during scanning

**Method**: Rigid-body registration using mutual information

**Key Parameters**:
- `reference_vol`: Reference volume for alignment (default: middle volume)
- `cost_function`: 'mutual_info', 'corr', or 'normcorr'
- `interp`: 'trilinear', 'nearest', or 'cubic'

**Code Example**:
```python
from schizophrenia_detection.data_processing.fmri_preprocessing import realign_motion_correction

# Apply motion correction
corrected_img = realign_motion_correction(
    fmri_img="data/fmri/sub-001_func.nii.gz",
    reference_vol=None,  # Use middle volume
    cost_function="mutual_info",
    interp="trilinear"
)

print(f"Motion correction completed: {corrected_img.shape}")
```

**Output**: Motion-corrected 4D fMRI volume

**Quality Metrics**:
- Maximum displacement (mm)
- Mean displacement
- Motion parameters (6 DOF: 3 translations, 3 rotations)

### 2. Slice Timing Correction

**Purpose**: Correct for differences in slice acquisition times

**Method**: Temporal interpolation to reference slice time

**Key Parameters**:
- `tr`: Repetition time in seconds
- `slice_order`: 'ascending', 'descending', or 'custom'
- `reference_slice`: Reference slice for correction
- `interleaved`: Whether acquisition was interleaved

**Code Example**:
```python
from schizophrenia_detection.data_processing.fmri_preprocessing import slice_timing_correction

# Apply slice timing correction
stc_img = slice_timing_correction(
    fmri_img=corrected_img,
    tr=2.0,
    slice_order="ascending",
    reference_slice=None,  # Use middle slice
    interleaved=False
)

print(f"Slice timing correction completed: {stc_img.shape}")
```

**Output**: Slice timing corrected 4D volume

**Quality Metrics**:
- Temporal SNR improvement
- Residual slice timing artifacts

### 3. Spatial Smoothing

**Purpose**: Reduce noise and improve signal-to-noise ratio

**Method**: Gaussian smoothing kernel

**Key Parameters**:
- `fwhm`: Full width at half maximum in mm (default: 6.0mm)

**Code Example**:
```python
from schizophrenia_detection.data_processing.fmri_preprocessing import spatial_smoothing

# Apply spatial smoothing
smoothed_img = spatial_smoothing(
    fmri_img=stc_img,
    fwhm=6.0
)

print(f"Spatial smoothing completed: {smoothed_img.shape}")
```

**Output**: Spatially smoothed 4D volume

**Quality Metrics**:
- Effective smoothing
- SNR improvement
- Spatial resolution preservation

### 4. Coregistration

**Purpose**: Align functional data to structural anatomy

**Method**: Rigid-body registration to structural image

**Key Parameters**:
- `struct_img`: Structural anatomical image
- `cost_function`: Registration cost function
- `interp`: Interpolation method

**Code Example**:
```python
from schizophrenia_detection.data_processing.fmri_preprocessing import coregistration

# Apply coregistration (if structural image available)
if struct_img:
    coreg_img = coregistration(
        func_img=smoothed_img,
        struct_img="data/anat/sub-001_T1w.nii.gz",
        cost_function="mutual_info",
        interp="trilinear"
    )
    print(f"Coregistration completed: {coreg_img.shape}")
else:
    coreg_img = smoothed_img
    print("Skipping coregistration (no structural image)")
```

**Output**: Coregistered 4D volume

**Quality Metrics**:
- Registration quality
- Overlap metrics
- Visual inspection of alignment

### 5. Normalization to MNI Space

**Purpose**: Transform to standard anatomical space

**Method**: Affine transformation to MNI template

**Key Parameters**:
- `template_img`: MNI template image
- `voxel_size`: Target voxel size (default: 2.0mm isotropic)

**Code Example**:
```python
from schizophrenia_detection.data_processing.fmri_preprocessing import normalize_to_mni

# Apply normalization
normalized_img = normalize_to_mni(
    fmri_img=coreg_img,
    template_img=None,  # Use default MNI template
    voxel_size=(2.0, 2.0, 2.0)
)

print(f"Normalization completed: {normalized_img.shape}")
```

**Output**: Normalized 4D volume in MNI space

**Quality Metrics**:
- Transformation quality
- Overlay with MNI template
- Anatomical accuracy

## SSP Map Generation

### Spatial Source Phase (SSP) Maps

**Purpose**: Generate phase-based features for specialized analysis

**Method**: Complex-valued analysis with phase extraction

**Key Components**:
1. **Complex Representation**: Hilbert transform for complex signal
2. **Phase Extraction**: Angle of complex signal
3. **Amplitude Weighting**: Phase weighted by signal amplitude
4. **Reverse Phase De-ambiguization**: Phase optimization

### SSP Map Generation

```python
from schizophrenia_detection.data_processing.fmri_preprocessing import generate_ssp_maps

# Generate SSP maps
ssp_map = generate_ssp_maps(
    fmri_img=normalized_img,
    mask_img=None  # Auto-generate brain mask
)

print(f"SSP map generated: {ssp_map.shape}")
```

### Reverse Phase De-ambiguization

```python
from schizophrenia_detection.data_processing.fmri_preprocessing import reverse_phase_deambiguization

# Apply reverse phase de-ambiguization
deambiguated_ssp = reverse_phase_deambiguization(
    ssp_map_img=ssp_map,
    amplitude_threshold=0.5
)

print(f"De-ambiguated SSP map: {deambiguated_ssp.shape}")
```

### SSP Map Quality Control

```python
import matplotlib.pyplot as plt
import nibabel as nib

# Visualize SSP map
ssp_data = deambiguated_ssp.get_fdata()

# Show middle slice
middle_slice = ssp_data[:, :, ssp_data.shape[2]//2]

plt.figure(figsize=(10, 8))
plt.imshow(middle_slice, cmap='RdBu_r')
plt.title("SSP Map - Middle Slice")
plt.colorbar()
plt.axis('off')
plt.show()
```

## Statistical Parametric Mapping (SPM)

### SPM Analysis

**Purpose**: Statistical analysis of activation patterns

**Method**: General Linear Model (GLM) with contrast testing

**Requirements**:
- Design matrix (timepoints × regressors)
- Contrast matrix (contrasts × regressors)

### SPM Implementation

```python
from schizophrenia_detection.data_processing.fmri_preprocessing import statistical_parametric_mapping
import numpy as np

# Create design matrix (example)
n_timepoints = normalized_img.shape[3]
design_matrix = np.ones((n_timepoints, 2))  # Intercept + task
design_matrix[n_timepoints//2:, 1] = 1  # Task block

# Create contrast matrix
contrast_matrix = np.array([[0, 1]])  # Task vs baseline

# Perform SPM analysis
spm_results = statistical_parametric_mapping(
    fmri_img=normalized_img,
    design_matrix=design_matrix,
    contrast_matrix=contrast_matrix,
    mask_img=None
)

print(f"SPM analysis completed: {list(spm_results.keys())}")
```

### SPM Results

```python
# Access SPM results
for contrast_name, results in spm_results.items():
    stat_map = results["stat_map"]
    p_map = results["p_map"]
    t_stat = results["t_stat"]
    
    print(f"Contrast: {contrast_name}")
    print(f"Max t-statistic: {np.max(np.abs(t_stat)):.3f}")
    print(f"Significant voxels (p<0.05): {np.sum(p_map < 0.05)}")
```

## Complete Pipeline Example

### Full Pipeline Execution

```python
from schizophrenia_detection.data_processing.fmri_preprocessing import preprocess_fmri_pipeline

# Execute complete pipeline
results = preprocess_fmri_pipeline(
    fmri_img="data/fmri/sub-001_func.nii.gz",
    struct_img="data/anat/sub-001_T1w.nii.gz",
    tr=2.0,
    fwhm=6.0,
    slice_order="ascending",
    perform_normalization=True,
    perform_spm=False,  # Set to True if you have design matrix
    design_matrix=None,
    contrast_matrix=None,
    generate_ssp=True,  # Generate SSP maps
    output_dir="data/preprocessed/sub-001"
)

# Access results
print("Pipeline results:")
for step, result in results.items():
    if hasattr(result, 'shape'):
        print(f"  {step}: {result.shape}")
    else:
        print(f"  {step}: {type(result)}")
```

### Batch Processing

```python
import os
import pandas as pd

# Load participants list
participants = pd.read_csv("data/metadata/participants.csv")

# Process all subjects
for _, subject in participants.iterrows():
    subject_id = subject['participant_id']
    
    # Input paths
    fmri_path = f"data/fmri/{subject_id}_func.nii.gz"
    struct_path = f"data/anat/{subject_id}_T1w.nii.gz"
    
    # Check if files exist
    if not os.path.exists(fmri_path):
        print(f"Skipping {subject_id}: fMRI file not found")
        continue
    
    # Output directory
    output_dir = f"data/preprocessed/{subject_id}"
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Process subject
        results = preprocess_fmri_pipeline(
            fmri_img=fmri_path,
            struct_img=struct_path if os.path.exists(struct_path) else None,
            tr=2.0,
            fwhm=6.0,
            generate_ssp=True,
            output_dir=output_dir
        )
        
        print(f"Completed {subject_id}")
        
    except Exception as e:
        print(f"Error processing {subject_id}: {e}")
        continue
```

## Quality Control

### Motion Parameters Visualization

```python
import matplotlib.pyplot as plt
import numpy as np

def plot_motion_parameters(motion_params):
    """Plot motion parameters over time"""
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    
    # Translations
    axes[0, 0].plot(motion_params[:, 0])
    axes[0, 0].set_title('X Translation (mm)')
    axes[0, 0].set_ylabel('Displacement')
    
    axes[0, 1].plot(motion_params[:, 1])
    axes[0, 1].set_title('Y Translation (mm)')
    
    axes[0, 2].plot(motion_params[:, 2])
    axes[0, 2].set_title('Z Translation (mm)')
    
    # Rotations
    axes[1, 0].plot(motion_params[:, 3])
    axes[1, 0].set_title('X Rotation (deg)')
    axes[1, 0].set_ylabel('Rotation')
    
    axes[1, 1].plot(motion_params[:, 4])
    axes[1, 1].set_title('Y Rotation (deg)')
    
    axes[1, 2].plot(motion_params[:, 5])
    axes[1, 2].set_title('Z Rotation (deg)')
    
    plt.tight_layout()
    plt.show()

# Usage (if motion parameters are available)
# plot_motion_parameters(motion_params)
```

### Signal Quality Metrics

```python
def calculate_quality_metrics(fmri_img):
    """Calculate fMRI quality metrics"""
    import nibabel as nib
    
    img = nib.load(fmri_img) if isinstance(fmri_img, str) else fmri_img
    data = img.get_fdata()
    
    # Temporal SNR
    temporal_mean = np.mean(data, axis=3)
    temporal_std = np.std(data, axis=3)
    temporal_snr = np.mean(temporal_mean / temporal_std)
    
    # Global signal
    global_signal = np.mean(data, axis=(0, 1, 2))
    global_signal_std = np.std(global_signal)
    
    # DVARS (standardized DVARS)
    diff_data = np.diff(data, axis=3)
    dvars = np.mean(np.std(diff_data, axis=(0, 1, 2)))
    
    metrics = {
        'temporal_snr': temporal_snr,
        'global_signal_std': global_signal_std,
        'dvars': dvars,
        'mean_signal': np.mean(data),
        'std_signal': np.std(data)
    }
    
    return metrics

# Usage
metrics = calculate_quality_metrics("data/preprocessed/sub-001/05_normalized.nii.gz")
print("Quality Metrics:")
for metric, value in metrics.items():
    print(f"  {metric}: {value:.3f}")
```

## Configuration

### Parameter Configuration

Update [`config.py`](../config.py:1) to adjust preprocessing parameters:

```python
# In config.py
config.data.fmri_voxel_size = (2.0, 2.0, 2.0)  # mm
config.data.fmri_tr = 2.0  # seconds
config.data.fmri_shape = (96, 96, 96, 4)  # Output shape
```

### Advanced Configuration

```python
# Custom preprocessing parameters
preprocessing_config = {
    'motion_correction': {
        'cost_function': 'mutual_info',
        'interp': 'trilinear'
    },
    'slice_timing': {
        'slice_order': 'ascending',
        'interleaved': False
    },
    'smoothing': {
        'fwhm': 6.0
    },
    'normalization': {
        'voxel_size': (2.0, 2.0, 2.0)
    },
    'ssp': {
        'amplitude_threshold': 0.5
    }
}
```

## Troubleshooting

### Common Issues

#### Memory Errors

**Problem**: Out of memory during preprocessing

**Solution**:
```python
# Reduce data size
config.data.fmri_shape = (64, 64, 64, 4)  # Smaller volumes

# Process in chunks
def process_in_chunks(fmri_img, chunk_size=50):
    img = nib.load(fmri_img)
    data = img.get_fdata()
    
    for i in range(0, data.shape[3], chunk_size):
        chunk = data[..., i:i+chunk_size]
        # Process chunk
        yield process_chunk(chunk)
```

#### Motion Artifacts

**Problem**: Excessive head motion

**Solution**:
```python
# Check motion parameters
motion_params = extract_motion_parameters(fmri_img)
max_displacement = np.max(np.abs(motion_params[:, :3]))

if max_displacement > 3.0:  # 3mm threshold
    print("Warning: Excessive motion detected")
    # Consider scrubbing or exclusion
```

#### Normalization Failures

**Problem**: Poor normalization quality

**Solution**:
```python
# Check brain extraction
from nilearn import masking

brain_mask = masking.compute_brain_mask(normalized_img)
brain_volume = np.sum(brain_mask.get_fdata())

if brain_volume < 500000:  # voxels
    print("Warning: Small brain volume detected")
    # Check normalization parameters
```

#### SSP Map Issues

**Problem**: Blank or noisy SSP maps

**Solution**:
```python
# Check signal quality
ssp_data = ssp_map.get_fdata()
signal_range = np.max(ssp_data) - np.min(ssp_data)

if signal_range < 0.1:
    print("Warning: Low signal range in SSP map")
    # Adjust amplitude_threshold parameter
```

### Debugging Tools

```python
def debug_preprocessing_step(input_img, output_img, step_name):
    """Debug preprocessing step with visual inspection"""
    
    input_data = input_img.get_fdata()
    output_data = output_img.get_fdata()
    
    print(f"Debug: {step_name}")
    print(f"  Input shape: {input_data.shape}")
    print(f"  Output shape: {output_data.shape}")
    print(f"  Input range: [{np.min(input_data):.3f}, {np.max(input_data):.3f}]")
    print(f"  Output range: [{np.min(output_data):.3f}, {np.max(output_data):.3f}]")
    
    # Visual comparison
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Input
    input_slice = input_data[:, :, input_data.shape[2]//2, 0]
    axes[0].imshow(input_slice, cmap='gray')
    axes[0].set_title(f'{step_name} - Input')
    axes[0].axis('off')
    
    # Output
    output_slice = output_data[:, :, output_data.shape[2]//2, 0]
    axes[1].imshow(output_slice, cmap='gray')
    axes[1].set_title(f'{step_name} - Output')
    axes[1].axis('off')
    
    plt.show()

# Usage in pipeline
debug_preprocessing_step(raw_img, corrected_img, "Motion Correction")
```

## Next Steps

After completing fMRI preprocessing:

1. Validate preprocessing quality using QC metrics
2. Proceed with MEG preprocessing using [MEG_PREPROCESSING.md](MEG_PREPROCESSING.md:1)
3. Train the SSPNet model using [model_training.ipynb](../notebooks/model_training.ipynb:1)
4. Evaluate results using [results_analysis.ipynb](../notebooks/results_analysis.ipynb:1)

For additional help, see the [FAQ.md](FAQ.md:1) or open an issue on the GitHub repository.