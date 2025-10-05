# Google Colab Setup Guide

This guide provides detailed instructions for setting up the schizophrenia detection pipeline on Google Colab, including GPU configuration, Drive mounting, package installation, and optimization tips.

## Prerequisites

- Google account with access to Google Drive
- Basic familiarity with Jupyter notebooks
- Neuroimaging data uploaded to Google Drive (see [DATA_PREP.md](DATA_PREP.md:1))

## 1. Runtime Configuration

### Selecting GPU Runtime

1. Open any notebook in Colab
2. Go to **Runtime** → **Change runtime type**
3. Select **GPU** as hardware accelerator
4. Choose GPU type (recommended):
   - **T4** (default, good balance of memory and speed)
   - **A100** (if available, for large datasets)
   - **V100** (alternative high-performance option)

### Verifying GPU Setup

```python
# Check GPU availability and type
import tensorflow as tf
print("TensorFlow version:", tf.__version__)
print("GPU Available:", tf.config.list_physical_devices('GPU'))

# Get GPU details
!nvidia-smi
```

Expected output should show:
- TensorFlow version 2.x
- At least one GPU detected
- GPU memory information

## 2. Google Drive Mounting

### Mounting Drive

```python
from google.colab import drive
drive.mount('/content/drive')
```

You'll be prompted to:
1. Click the authorization link
2. Sign in to your Google account
3. Copy the authorization code
4. Paste it in the provided box

### Navigating to Project Directory

```python
# Navigate to the project directory
%cd /content/drive/MyDrive/schizophrenia_detection

# Verify current directory
!pwd
!ls -la
```

### Directory Structure Verification

Ensure your Drive has the expected structure:
```
/content/drive/MyDrive/schizophrenia_detection/
├── data/
│   ├── fmri/
│   ├── meg/
│   └── metadata/
├── notebooks/
├── requirements.txt
└── config.py
```

## 3. Package Installation

### Basic Installation

```python
# Install basic requirements
!pip install -r requirements.txt
```

### Additional Dependencies for Colab

```python
# Install additional packages for Colab environment
!pip install --upgrade pip
!pip install ipywidgets  # for interactive widgets
!pip install tqdm        # progress bars
!pip install matplotlib seaborn plotly  # visualization
```

### Neuroimaging Packages

```python
# Install neuroimaging packages
!pip install nilearn nibabel mne
```

### Documentation Tools (Optional)

```python
# For building documentation locally
!pip install mkdocs mkdocs-material
```

## 4. Environment Verification

### Verify Core Packages

```python
# Check key package versions
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nibabel as nib
import nilearn
import mne

print("NumPy:", np.__version__)
print("Pandas:", pd.__version__)
print "Matplotlib:", matplotlib.__version__)
print("Nibabel:", nib.__version__)
print("Nilearn:", nilearn.__version__)
print("MNE:", mne.__version__)
```

### Test GPU with TensorFlow

```python
import tensorflow as tf

# Test GPU computation
if tf.config.list_physical_devices('GPU'):
    # Create a simple tensor on GPU
    with tf.device('/GPU:0'):
        a = tf.random.normal([1000, 1000])
        b = tf.random.normal([1000, 1000])
        c = tf.matmul(a, b)
    print("GPU computation successful!")
else:
    print("No GPU detected. Using CPU.")
```

### Test Neuroimaging Libraries

```python
# Test nibabel
import nibabel as nib
print("Nibabel test:", nib.__version__)

# Test nilearn
from nilearn import datasets
print("Nilearn test: OK")

# Test MNE
import mne
print("MNE test:", mne.__version__)
```

## 5. Memory Optimization

### Monitoring Memory Usage

```python
# Check system memory
!free -h

# Check GPU memory
!nvidia-smi
```

### Memory Management Tips

1. **Clear unused variables**:
```python
import gc
del large_variable
gc.collect()
```

2. **Use mixed precision training**:
```python
from tensorflow.keras import mixed_precision
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)
```

3. **Reduce batch size** in [`config.py`](../config.py:1):
```python
# In config.py
data.batch_size = 2  # Reduce from default 4
```

4. **Use data generators** instead of loading all data at once:
```python
from schizophrenia_detection.data_processing.data_loader import create_data_generator

generator = create_data_generator(
    data_paths, labels, 
    batch_size=2,  # Small batch size
    shuffle=True
)
```

## 6. Session Management

### Preventing Session Timeouts

1. **Keep the tab active** - Colab disconnects after 90 minutes of inactivity
2. **Use JavaScript to keep session alive** (run in console):
```javascript
function ClickConnect(){
    console.log("Working");
    document.querySelector("colab-toolbar-button#connect").click()
}
setInterval(ClickConnect, 60000)
```

3. **Save checkpoints frequently**:
```python
# In config.py
training.save_best_only = True
training.checkpoint_dir = "./checkpoints"
```

### Handling Disconnections

1. **Save progress frequently**:
```python
# Save model checkpoints
model.save('checkpoint.h5')

# Save training history
import json
with open('training_history.json', 'w') as f:
    json.dump(history.history, f)
```

2. **Use resume training**:
```python
# Load checkpoint if exists
if os.path.exists('checkpoint.h5'):
    model = tf.keras.models.load_model('checkpoint.h5')
    print("Resumed from checkpoint")
```

## 7. Data Loading Optimization

### Efficient Data Loading

```python
# Use data generators for large datasets
from schizophrenia_detection.data_processing.data_loader import create_data_generator

# Create generator with optimal settings
train_generator = create_data_generator(
    train_paths, train_labels,
    batch_size=2,  # Adjust based on GPU memory
    shuffle=True
)

# Use generator in training
model.fit(
    train_generator,
    steps_per_epoch=len(train_paths) // 2,
    epochs=50
)
```

### Caching Data

```python
# Cache preprocessed data to avoid recomputation
import os
import pickle

def cache_preprocessed_data(data, cache_path):
    """Cache preprocessed data to avoid recomputation"""
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, 'wb') as f:
        pickle.dump(data, f)

def load_cached_data(cache_path):
    """Load cached data if available"""
    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as f:
            return pickle.load(f)
    return None
```

## 8. Common Issues and Solutions

### Out of Memory (OOM) Errors

**Symptoms**: `ResourceExhaustedError` or CUDA OOM

**Solutions**:
1. Reduce batch size in [`config.py`](../config.py:1)
2. Enable mixed precision training
3. Use smaller input volumes
4. Clear GPU memory between runs:
```python
tf.keras.backend.clear_session()
```

### Slow I/O Operations

**Symptoms**: Long loading times for data

**Solutions**:
1. Cache preprocessed data to local Colab storage
2. Use compressed data formats
3. Preprocess data in batches
4. Use Google Drive streaming for large files

### Package Installation Failures

**Symptoms**: `pip install` errors

**Solutions**:
1. Upgrade pip first: `!pip install --upgrade pip`
2. Install packages one by one to identify conflicts
3. Use specific versions if needed
4. Restart runtime after installation

### GPU Not Detected

**Symptoms**: No GPU shown in `nvidia-smi`

**Solutions**:
1. Change runtime type to GPU
2. Restart runtime
3. Check if GPU quota is exceeded
4. Try again later if resources are unavailable

## 9. Performance Monitoring

### Monitor Training Progress

```python
# Set up TensorBoard
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard

tensorboard_callback = TensorBoard(
    log_dir='./logs/tensorboard',
    histogram_freq=1
)

# Use in model.fit
model.fit(
    train_generator,
    callbacks=[tensorboard_callback]
)
```

### Monitor Resource Usage

```python
# Monitor GPU memory during training
import GPUtil

def monitor_gpu():
    gpus = GPUtil.getGPUs()
    for gpu in gpus:
        print(f"GPU {gpu.id}: {gpu.memoryUtil*100:.1f}% memory used")

# Call periodically during training
monitor_gpu()
```

## 10. Best Practices

### Code Organization

1. **Use relative imports** for project modules
2. **Keep notebooks focused** on specific tasks
3. **Use configuration files** for parameters
4. **Document your experiments** in markdown cells

### Experiment Tracking

```python
# Create experiment log
import datetime
experiment_log = {
    'date': datetime.datetime.now().isoformat(),
    'parameters': config.__dict__,
    'results': {}
}

# Save experiment details
with open(f'experiment_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.json', 'w') as f:
    json.dump(experiment_log, f, indent=2)
```

### Collaboration

1. **Share notebooks** with appropriate permissions
2. **Use version control** for code changes
3. **Document data sources** and preprocessing steps
4. **Share configuration files** for reproducibility

## 11. Running the Pipeline

### Quick Start

```python
# 1. Mount Drive and navigate
from google.colab import drive
drive.mount('/content/drive')
%cd /content/drive/MyDrive/schizophrenia_detection

# 2. Install dependencies
!pip install -r requirements.txt

# 3. Verify environment
import tensorflow as tf
print("GPU Available:", tf.config.list_physical_devices('GPU'))

# 4. Run notebooks in order:
#    - data_exploration.ipynb
#    - model_training.ipynb
#    - results_analysis.ipynb
```

### Running Individual Components

```python
# Preprocess fMRI data
from schizophrenia_detection.data_processing import fmri_preprocessing
results = fmri_preprocessing.preprocess_fmri_pipeline(
    fmri_img="data/fmri/sub-01_func.nii.gz",
    output_dir="data/preprocessed/"
)

# Train model
from schizophrenia_detection.models import sspnet_3d_cnn
from schizophrenia_detection.training import trainer

model = sspnet_3d_cnn.SSPNet3DCNN()
trainer.train_model(model, train_data, val_data, config)
```

## 12. Troubleshooting Checklist

Before running the pipeline, verify:

- [ ] GPU runtime selected
- [ ] Google Drive mounted successfully
- [ ] All packages installed without errors
- [ ] Data directories exist and contain files
- [ ] Configuration file paths are correct
- [ ] Sufficient GPU memory available
- [ ] Session timeout prevention enabled

## Next Steps

After completing the setup:

1. Review [DATA_PREP.md](DATA_PREP.md:1) for data preparation
2. Run [data_exploration.ipynb](../notebooks/data_exploration.ipynb:1) to understand your data
3. Follow [model_training.ipynb](../notebooks/model_training.ipynb:1) for training
4. Use [results_analysis.ipynb](../notebooks/results_analysis.ipynb:1) for evaluation

For additional help, see the [FAQ.md](FAQ.md:1) or open an issue on the GitHub repository.