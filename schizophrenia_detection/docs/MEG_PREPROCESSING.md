# MEG Preprocessing Guide

This guide covers the complete MEG preprocessing pipeline for schizophrenia detection, including artifact removal, filtering, source localization, and spatial processing.

## Table of Contents

1. [Overview](#overview)
2. [Preprocessing Pipeline](#preprocessing-pipeline)
3. [Step-by-Step Processing](#step-by-step-processing)
4. [Source Space Processing](#source-space-processing)
5. [Beamformer Analysis](#beamformer-analysis)
6. [Quality Control](#quality-control)
7. [Code Examples](#code-examples)
8. [Configuration](#configuration)
9. [Troubleshooting](#troubleshooting)

## Overview

The MEG preprocessing pipeline implements a comprehensive workflow that transforms raw MEG sensor data into source-localized, frequency-specific volumes ready for SSPNet 3D CNN training. The pipeline follows MNE-Python best practices with specialized components for schizophrenia detection.

### Key Features

- **Maxfilter**: Artifact removal and head movement correction
- **SSP**: Signal-Space Projection for cardiac and blink artifacts
- **Bandpass Filtering**: Four frequency bands (delta, theta, alpha, beta)
- **Covariance Regularization**: Robust covariance matrix estimation
- **Source Localization**: 6-mm³ grid with beamformer
- **MNI Normalization**: Standard space transformation
- **Hilbert Envelopes**: Amplitude envelope extraction
- **Spatial Processing**: Smoothing and resampling

### Input Requirements

- **Format**: FIF (.fif) preferred, CTF (.ds), BrainVision (.vhdr) supported
- **Channels**: MEG channels required, EOG/ECG optional
- **Sampling Rate**: Minimum 100 Hz, recommended 250-1000 Hz
- **Duration**: Minimum 5 minutes recommended
- **Head Position**: Optional but recommended for movement correction

## Preprocessing Pipeline

### Pipeline Architecture

```
Raw MEG → Maxfilter → SSP → Bandpass Filtering → Covariance → Source Space → Beamformer → Hilbert → Spatial Processing → MNI → Output
```

### Implementation Location

The complete pipeline is implemented in [`meg_preprocessing.py`](../data_processing/meg_preprocessing.py:1) with the main class:

```python
MEGPreprocessor(config=None)
preprocessor.preprocess_meg_data(
    file_path, head_pos=None, subject="fsaverage", 
    trans=None, save_output=False, output_dir=None
)
```

## Step-by-Step Processing

### 1. Data Loading

**Purpose**: Load raw MEG data with validation

**Supported Formats**:
- FIF (.fif) - MNE-Python native format
- CTF (.ds) - CTF systems
- BrainVision (.vhdr) - BrainVision systems

**Code Example**:
```python
from schizophrenia_detection.data_processing.meg_preprocessing import MEGPreprocessor

# Initialize preprocessor
preprocessor = MEGPreprocessor()

# Load MEG data
raw = preprocessor.load_meg_data(
    file_path="data/meg/sub-001_meg.fif",
    preload=False  # Load on demand
)

print(f"Loaded MEG data: {raw}")
print(f"Channels: {len(raw.ch_names)}")
print(f"Duration: {raw.times[-1]:.1f} seconds")
print(f"Sampling rate: {raw.info['sfreq']} Hz")
```

### 2. Maxfilter Processing

**Purpose**: Remove environmental noise and correct head movement

**Method**: Signal Space Separation (SSS) with movement compensation

**Key Parameters**:
- `head_pos`: Head position file for movement correction
- `downsample_freq`: Target frequency for downsampling

**Code Example**:
```python
# Apply Maxfilter
raw_filtered = preprocessor.apply_maxfilter(
    raw=raw,
    head_pos="data/meg/sub-001_head.pos",  # Optional
    downsample_freq=250.0
)

print(f"Maxfilter applied: {raw_filtered}")
print(f"New sampling rate: {raw_filtered.info['sfreq']} Hz")
```

**Output**: Noise-reduced, movement-corrected MEG data

**Quality Metrics**:
- Noise reduction factor
- Movement compensation quality
- Signal preservation

### 3. Signal-Space Projection (SSP)

**Purpose**: Remove physiological artifacts (cardiac, blink)

**Method**: Projection vectors computed from ECG/EOG channels

**Key Parameters**:
- `ecg_ch`: ECG channel name
- `eog_ch`: List of EOG channel names

**Code Example**:
```python
# Apply SSP for artifact removal
raw_ssp = preprocessor.compute_ssp_projections(
    raw=raw_filtered,
    ecg_ch=None,  # Auto-detect
    eog_ch=None   # Auto-detect
)

print(f"SSP applied: {raw_ssp}")
print(f"Number of projections: {len(raw_ssp.info['projs'])}")
```

**Output**: Artifact-corrected MEG data

**Quality Metrics**:
- Artifact reduction factor
- Signal preservation
- Projection quality

### 4. Bandpass Filtering

**Purpose**: Extract frequency-specific information

**Frequency Bands**:
- **Delta**: 1-4 Hz
- **Theta**: 5-9 Hz
- **Alpha**: 10-15 Hz
- **Beta**: 16-29 Hz

**Code Example**:
```python
# Apply bandpass filtering for all frequency bands
filtered_data = preprocessor.apply_bandpass_filter(
    raw=raw_ssp,
    freq_bands=None  # Use default bands
)

print(f"Filtered data keys: {list(filtered_data.keys())}")
for band_name, band_data in filtered_data.items():
    print(f"  {band_name}: {band_data}")
```

**Output**: Dictionary of filtered datasets for each frequency band

**Quality Metrics**:
- Filter quality (roll-off, ripple)
- Signal preservation
- Frequency response

### 5. Covariance Matrix Computation

**Purpose**: Estimate sensor covariance for beamformer

**Method**: Regularized covariance estimation

**Key Parameters**:
- `method`: Covariance estimation method
- `reg_factor`: Regularization factor

**Code Example**:
```python
# Compute regularized covariance matrices
covariances = {}
for band_name, band_data in filtered_data.items():
    cov = preprocessor.compute_regularized_covariance(
        raw=band_data,
        method="shrunk"
    )
    covariances[band_name] = cov
    print(f"Computed covariance for {band_name}")
```

**Output**: Regularized covariance matrices for each frequency band

**Quality Metrics**:
- Condition number
- Eigenvalue spectrum
- Regularization effectiveness

## Source Space Processing

### 1. Source Space Setup

**Purpose**: Create source space for beamformer

**Method**: 6-mm³ grid on cortical surface

**Code Example**:
```python
# Setup source space
src = preprocessor.setup_source_space(
    subject="fsaverage",
    spacing="oct6"  # ~6mm spacing
)

print(f"Source space created: {src}")
print(f"Number of sources: {len(src)}")
```

### 2. BEM Model Creation

**Purpose**: Create forward model for beamformer

**Method**: Single-shell Boundary Element Model

**Code Example**:
```python
# Create BEM model
bem = preprocessor.create_bem_model(
    subject="fsaverage",
    conductivity=(0.3, 0.006, 0.3)
)

print(f"BEM model created: {bem}")
```

### 3. Forward Solution

**Purpose**: Compute forward solution for beamformer

**Code Example**:
```python
# Create forward solution
fwd = preprocessor.create_forward_solution(
    raw=raw_ssp,
    src=src,
    bem=bem,
    trans=None  # Use default transformation
)

print(f"Forward solution created: {fwd}")
print(f"Sources: {fwd['nsource']}, Channels: {fwd['nchan']}")
```

## Beamformer Analysis

### 1. LCMV Beamformer

**Purpose**: Localize neural activity

**Method**: Linearly Constrained Minimum Variance beamformer

**Key Parameters**:
- `reg`: Regularization parameter
- `pick_ori`: Orientation picking method

**Code Example**:
```python
# Apply LCMV beamformer for each frequency band
source_estimates = {}
for band_name, band_data in filtered_data.items():
    cov = covariances[band_name]
    
    stc = preprocessor.apply_lcmv_beamformer(
        raw=band_data,
        fwd=fwd,
        cov=cov,
        reg=0.05,
        pick_ori="max-power"
    )
    
    source_estimates[band_name] = stc
    print(f"Beamformer applied for {band_name}: {stc}")
```

**Output**: Source time courses for each frequency band

### 2. Hilbert Transform

**Purpose**: Extract amplitude envelopes

**Code Example**:
```python
# Apply Hilbert transform for amplitude envelopes
envelopes = {}
for band_name, stc in source_estimates.items():
    envelope_data = preprocessor.apply_hilbert_transform(stc)
    envelopes[band_name] = envelope_data
    print(f"Hilbert transform applied for {band_name}")
```

**Output**: Amplitude envelopes for each frequency band

### 3. Spatial Processing

**Purpose**: Apply spatial smoothing and resampling

**Key Parameters**:
- `smooth_fwhm`: Smoothing FWHM in mm
- `target_voxel_size`: Target voxel size

**Code Example**:
```python
# Apply spatial processing
processed_data = {}
for band_name, envelope_data in envelopes.items():
    for env_name, env_stc in envelope_data.items():
        processed_stc = preprocessor.apply_spatial_processing(
            stc=env_stc,
            smooth_fwhm=6.0,
            target_voxel_size=(3.0, 3.0, 3.0)
        )
        
        if band_name not in processed_data:
            processed_data[band_name] = {}
        processed_data[band_name][env_name] = processed_stc
        
        print(f"Spatial processing applied for {band_name}-{env_name}")
```

### 4. MNI Normalization

**Purpose**: Transform to standard anatomical space

**Code Example**:
```python
# Normalize to MNI space
mni_data = {}
for band_name, band_data in processed_data.items():
    mni_data[band_name] = {}
    for env_name, env_stc in band_data.items():
        mni_stc = preprocessor.normalize_to_mni(
            stc=env_stc,
            subject="fsaverage"
        )
        mni_data[band_name][env_name] = mni_stc
        print(f"MNI normalization applied for {band_name}-{env_name}")
```

**Output**: MNI-normalized source estimates

## Complete Pipeline Example

### Full Pipeline Execution

```python
from schizophrenia_detection.data_processing.meg_preprocessing import MEGPreprocessor

# Initialize preprocessor
preprocessor = MEGPreprocessor()

# Execute complete pipeline
results = preprocessor.preprocess_meg_data(
    file_path="data/meg/sub-001_meg.fif",
    head_pos="data/meg/sub-001_head.pos",  # Optional
    subject="fsaverage",
    trans=None,  # Use default transformation
    save_output=True,
    output_dir="data/preprocessed/meg/sub-001"
)

# Access results
print("Pipeline results:")
for band_name, band_data in results.items():
    print(f"  {band_name}:")
    for env_name, env_stc in band_data.items():
        print(f"    {env_name}: {env_stc}")
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
    
    # Input path
    meg_path = f"data/meg/{subject_id}_meg.fif"
    head_pos_path = f"data/meg/{subject_id}_head.pos"
    
    # Check if file exists
    if not os.path.exists(meg_path):
        print(f"Skipping {subject_id}: MEG file not found")
        continue
    
    # Output directory
    output_dir = f"data/preprocessed/meg/{subject_id}"
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Process subject
        results = preprocessor.preprocess_meg_data(
            file_path=meg_path,
            head_pos=head_pos_path if os.path.exists(head_pos_path) else None,
            subject="fsaverage",
            save_output=True,
            output_dir=output_dir
        )
        
        print(f"Completed {subject_id}")
        
    except Exception as e:
        print(f"Error processing {subject_id}: {e}")
        continue
```

## Quality Control

### Channel Information

```python
def analyze_channels(raw):
    """Analyze MEG channel information"""
    
    # Get channel types
    meg_channels = mne.pick_types(raw.info, meg=True)
    eog_channels = mne.pick_types(raw.info, eog=True)
    ecg_channels = mne.pick_types(raw.info, ecg=True)
    
    print(f"Channel Analysis:")
    print(f"  Total channels: {len(raw.ch_names)}")
    print(f"  MEG channels: {len(meg_channels)}")
    print(f"  EOG channels: {len(eog_channels)}")
    print(f"  ECG channels: {len(ecg_channels)}")
    
    # Channel names
    if len(meg_channels) > 0:
        print(f"  MEG channel names: {[raw.ch_names[i] for i in meg_channels[:5]]}...")
    
    return {
        'meg_channels': len(meg_channels),
        'eog_channels': len(eog_channels),
        'ecg_channels': len(ecg_channels)
    }

# Usage
channel_info = analyze_channels(raw)
```

### Signal Quality Metrics

```python
def calculate_signal_quality(raw):
    """Calculate MEG signal quality metrics"""
    
    # Pick MEG channels
    meg_channels = mne.pick_types(raw.info, meg=True)
    meg_data = raw.get_data(picks=meg_channels)
    
    # Calculate metrics
    peak_to_peak = np.max(meg_data, axis=1) - np.min(meg_data, axis=1)
    rms = np.sqrt(np.mean(meg_data**2, axis=1))
    
    # Channel-wise metrics
    metrics = {
        'peak_to_peak_mean': np.mean(peak_to_peak),
        'peak_to_peak_std': np.std(peak_to_peak),
        'rms_mean': np.mean(rms),
        'rms_std': np.std(rms),
        'dynamic_range': np.max(peak_to_peak) - np.min(peak_to_peak)
    }
    
    return metrics

# Usage
quality_metrics = calculate_signal_quality(raw)
print("Signal Quality Metrics:")
for metric, value in quality_metrics.items():
    print(f"  {metric}: {value:.3e}")
```

### Power Spectral Density

```python
import matplotlib.pyplot as plt

def plot_psd(raw, fmin=1, fmax=50):
    """Plot power spectral density"""
    
    # Pick MEG channels
    meg_channels = mne.pick_types(raw.info, meg=True)
    
    # Compute PSD
    psd, freqs = mne.time_frequency.psd_welch(
        raw, fmin=fmin, fmax=fmax, picks=meg_channels
    )
    
    # Plot
    plt.figure(figsize=(12, 6))
    
    # Mean PSD across channels
    mean_psd = np.mean(psd, axis=0)
    
    plt.semilogy(freqs, mean_psd)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power Spectral Density')
    plt.title('MEG Power Spectral Density')
    plt.grid(True)
    
    # Mark frequency bands
    plt.axvspan(1, 4, alpha=0.2, color='blue', label='Delta')
    plt.axvspan(5, 9, alpha=0.2, color='green', label='Theta')
    plt.axvspan(10, 15, alpha=0.2, color='orange', label='Alpha')
    plt.axvspan(16, 29, alpha=0.2, color='red', label='Beta')
    
    plt.legend()
    plt.show()

# Usage
plot_psd(raw)
```

### Source Space Visualization

```python
def plot_source_estimate(stc, subject="fsaverage"):
    """Plot source estimate on brain"""
    
    try:
        from nilearn import plotting
        
        # Convert to NIfTI for visualization
        img = stc.as_volume(src, mni_resolution=3)
        
        # Plot
        plotting.plot_stat_map(
            img,
            title="Source Estimate",
            cut_coords=(0, 0, 0),
            black_bg=True,
            display_mode='ortho'
        )
        plt.show()
        
    except ImportError:
        print("Nilearn not available for visualization")

# Usage
if 'alpha' in source_estimates:
    plot_source_estimate(source_estimates['alpha'])
```

## Configuration

### Parameter Configuration

Update [`config.py`](../config.py:1) to adjust MEG preprocessing parameters:

```python
# In config.py
config.data.meg_lowpass = 40.0  # Hz
config.data.meg_highpass = 0.1  # Hz
config.data.meg_shape = (306, 100, 1000)  # sensors x time x frequency
```

### Advanced Configuration

```python
# Custom MEG preprocessing parameters
meg_config = {
    'maxfilter': {
        'downsample_freq': 250.0,
        'int_order': 8,
        'skip_by_annotation': 'bad'
    },
    'ssp': {
        'ecg_threshold': 0.25,
        'eog_threshold': 0.25
    },
    'bands': {
        'delta': (1, 4),
        'theta': (5, 9),
        'alpha': (10, 15),
        'beta': (16, 29)
    },
    'beamformer': {
        'reg': 0.05,
        'pick_ori': 'max-power'
    },
    'spatial': {
        'smooth_fwhm': 6.0,
        'target_voxel_size': (3.0, 3.0, 3.0)
    }
}
```

## Troubleshooting

### Common Issues

#### FSL Not Available

**Problem**: FSL required for MNI normalization

**Solution**:
```python
# Check FSL availability
from schizophrenia_detection.data_processing.meg_preprocessing import FSL_AVAILABLE

if not FSL_AVAILABLE:
    print("FSL not available. MNI normalization will be skipped.")
    print("Install FSL or use alternative normalization methods.")
    
    # Alternative: Use MNE's built-in normalization
    # This is a simplified implementation
    stc_mni = stc.copy()  # No transformation applied
```

#### Memory Errors

**Problem**: Out of memory during processing

**Solution**:
```python
# Reduce data size
config.data.meg_shape = (306, 50, 500)  # Smaller time windows

# Process in chunks
def process_in_chunks(raw, chunk_size=30):
    """Process MEG data in time chunks"""
    
    n_samples = len(raw.times)
    chunks = []
    
    for i in range(0, n_samples, chunk_size * int(raw.info['sfreq'])):
        start = i
        end = min(i + chunk_size * int(raw.info['sfreq']), n_samples)
        
        chunk = raw.copy().crop(start, end)
        chunks.append(chunk)
    
    return chunks

# Usage
chunks = process_in_chunks(raw, chunk_size=30)
for i, chunk in enumerate(chunks):
    print(f"Processing chunk {i+1}/{len(chunks)}")
    # Process chunk
```

#### Bad Channels

**Problem**: Noisy or bad channels affecting quality

**Solution**:
```python
# Detect bad channels
import numpy as np

def detect_bad_channels(raw, threshold=5.0):
    """Detect bad channels based on variance"""
    
    meg_channels = mne.pick_types(raw.info, meg=True)
    meg_data = raw.get_data(picks=meg_channels)
    
    # Calculate variance for each channel
    variances = np.var(meg_data, axis=1)
    median_var = np.median(variances)
    
    # Mark channels with high variance as bad
    bad_channels = []
    for i, var in enumerate(variances):
        if var > threshold * median_var:
            bad_channels.append(raw.ch_names[meg_channels[i]])
    
    return bad_channels

# Detect and mark bad channels
bad_channels = detect_bad_channels(raw)
if bad_channels:
    print(f"Detected bad channels: {bad_channels}")
    raw.info['bads'] = bad_channels
    
    # Interpolate bad channels
    raw.interpolate_bads()
```

#### Covariance Issues

**Problem**: Singular covariance matrix

**Solution**:
```python
# Increase regularization
preprocessor.reg_factor = 10.0  # Increase from default 4.0

# Use alternative covariance estimation
cov = preprocessor.compute_regularized_covariance(
    raw=band_data,
    method="oas"  # Oracle Approximating Shrinkage
)
```

### Debugging Tools

```python
def debug_meg_step(input_data, output_data, step_name):
    """Debug MEG preprocessing step"""
    
    print(f"Debug: {step_name}")
    
    if hasattr(input_data, 'get_data'):
        input_data_array = input_data.get_data()
        print(f"  Input shape: {input_data_array.shape}")
        print(f"  Input range: [{np.min(input_data_array):.3e}, {np.max(input_data_array):.3e}]")
    
    if hasattr(output_data, 'get_data'):
        output_data_array = output_data.get_data()
        print(f"  Output shape: {output_data_array.shape}")
        print(f"  Output range: [{np.min(output_data_array):.3e}, {np.max(output_data_array):.3e}]")
    
    elif hasattr(output_data, 'data'):
        print(f"  Output shape: {output_data.data.shape}")
        print(f"  Output range: [{np.min(output_data.data):.3e}, {np.max(output_data.data):.3e}]")

# Usage in pipeline
debug_meg_step(raw, raw_filtered, "Maxfilter")
debug_meg_step(raw_filtered, raw_ssp, "SSP")
```

## Next Steps

After completing MEG preprocessing:

1. Validate preprocessing quality using QC metrics
2. Combine with fMRI data for multimodal analysis
3. Train the SSPNet model using [model_training.ipynb](../notebooks/model_training.ipynb:1)
4. Evaluate results using [results_analysis.ipynb](../notebooks/results_analysis.ipynb:1)

For additional help, see the [FAQ.md](FAQ.md:1) or open an issue on the GitHub repository.