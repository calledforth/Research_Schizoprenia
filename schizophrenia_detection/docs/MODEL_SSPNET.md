# SSPNet 3D CNN Model Guide

This guide covers the SSPNet 3D CNN architecture for schizophrenia detection, including model structure, configuration, training, and interpretability.

## Overview

SSPNet is a specialized 3D convolutional neural network designed for multimodal neuroimaging analysis. It processes fMRI and MEG data to identify biomarkers associated with schizophrenia.

## Architecture

### Model Structure

```
Input (96×96×96×4) → Spatial Filters → Spectral Filters → Pooling → Dense Layers → Output (2 classes)
```

### Key Components

- **Spatial Filters**: 3D convolutions for spatial feature extraction
- **Spectral Filters**: Temporal/spectral processing
- **Pooling**: Spatial downsampling
- **Dense Layers**: Classification head

## Usage

### Basic Model Creation

```python
from schizophrenia_detection.models.sspnet_3d_cnn import SSPNet3DCNN

# Create model with default configuration
model = SSPNet3DCNN()
model.summary()
```

### Custom Configuration

```python
from schizophrenia_detection.config import Config

# Load configuration
config = Config()

# Modify model parameters
config.model.input_shape = (96, 96, 96, 4)
config.model.num_classes = 2
config.model.dropout_rate = 0.5

# Create model with custom config
model = SSPNet3DCNN(config.model)
```

## Training

### Basic Training

```python
from schizophrenia_detection.training import trainer

# Train model
history = trainer.train_model(
    model=model,
    train_data=train_generator,
    val_data=val_generator,
    config=config
)
```

### Model Saving/Loading

```python
# Save model
model.save('sspnet_model.h5')

# Load model
from tensorflow.keras.models import load_model
loaded_model = load_model('sspnet_model.h5')
```

## Interpretability

### Grad-CAM

```python
from schizophrenia_detection.visualization.model_visualization import generate_grad_cam

# Generate Grad-CAM heatmap
heatmap = generate_grad_cam(
    model=model,
    data_sample=x_test[0],
    layer_name='conv3d_3'
)
```

### Saliency Maps

```python
from schizophrenia_detection.visualization.model_visualization import generate_saliency_map

# Generate saliency map
saliency = generate_saliency_map(
    model=model,
    data_sample=x_test[0]
)
```

## Configuration

### Model Parameters

Update [`config.py`](../config.py:1):

```python
config.model.num_spatial_filters = 32
config.model.num_spectral_filters = 16
config.model.spatial_kernel_size = (3, 3, 3)
config.model.spectral_kernel_size = (1, 1, 3)
config.model.dense_units = [512, 256]
```

### Training Parameters

```python
config.training.epochs = 100
config.training.learning_rate = 0.001
config.training.batch_size = 4
config.training.mixed_precision = True
```

## Tips

- Use mixed precision training for GPU memory efficiency
- Start with smaller batch sizes if encountering OOM errors
- Enable early stopping to prevent overfitting
- Use cross-validation for robust evaluation

## Next Steps

1. Configure model parameters in [`config.py`](../config.py:1)
2. Train using [model_training.ipynb](../notebooks/model_training.ipynb:1)
3. Evaluate with [results_analysis.ipynb](../notebooks/results_analysis.ipynb:1)
4. Visualize results using [VISUALIZATION_GUIDE.md](VISUALIZATION_GUIDE.md:1)