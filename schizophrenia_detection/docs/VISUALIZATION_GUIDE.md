# Visualization Guide

This guide covers visualization tools for analyzing neuroimaging data and model results.

## Static Plots

### Result Plots

```python
from schizophrenia_detection.visualization.result_plots import plot_training_history, plot_confusion_matrix

# Plot training history
plot_training_history(history, save_path="./visualizations/training_history.png")

# Plot confusion matrix
plot_confusion_matrix(y_true, y_pred, save_path="./visualizations/confusion_matrix.png")
```

### Performance Metrics

```python
from schizophrenia_detection.visualization.result_plots import plot_roc_curve, plot_metrics_comparison

# ROC curve
plot_roc_curve(y_true, y_scores, save_path="./visualizations/roc_curve.png")

# Metrics comparison
plot_metrics_comparison(metrics_dict, save_path="./visualizations/metrics_comparison.png")
```

## Brain Visualization

### fMRI Brain Maps

```python
from schizophrenia_detection.visualization.brain_visualization import plot_fmri_activation

# Plot activation map
plot_fmri_activation(
    stat_map="results/activation_map.nii.gz",
    title="Schizophrenia vs Control",
    save_path="./visualizations/brain_activation.png"
)
```

### MEG Source Maps

```python
from schizophrenia_detection.visualization.brain_visualization import plot_meg_source

# Plot MEG source activity
plot_meg_source(
    source_estimate=stc,
    subject="fsaverage",
    save_path="./visualizations/meg_source.png"
)
```

## Model Visualization

### Architecture

```python
from schizophrenia_detection.visualization.model_visualization import plot_model_architecture

# Plot model architecture
plot_model_architecture(
    model=model,
    save_path="./visualizations/model_architecture.png"
)
```

### Saliency Maps

```python
from schizophrenia_detection.visualization.model_visualization import plot_saliency_map

# Plot saliency map
plot_saliency_map(
    model=model,
    data_sample=x_test[0],
    save_path="./visualizations/saliency_map.png"
)
```

### Grad-CAM

```python
from schizophrenia_detection.visualization.model_visualization import plot_grad_cam

# Plot Grad-CAM heatmap
plot_grad_cam(
    model=model,
    data_sample=x_test[0],
    layer_name="conv3d_3",
    save_path="./visualizations/grad_cam.png"
)
```

## Interactive Plots

### Interactive Dashboard

```python
from schizophrenia_detection.visualization.interactive_plots import create_interactive_dashboard

# Create dashboard
dashboard = create_interactive_dashboard(
    data=data,
    model=model,
    results=results
)
dashboard.show()
```

### 3D Brain Viewer

```python
from schizophrenia_detection.visualization.interactive_plots import plot_3d_brain

# Interactive 3D brain
plot_3d_brain(
    volume=brain_volume,
    threshold=2.0,
    interactive=True
)
```

## Configuration

### Visualization Parameters

```python
# In config.py
config.visualization.figure_size = (10, 8)
config.visualization.dpi = 300
config.visualization.fmri_colormap = "viridis"
config.visualization.meg_colormap = "plasma"
config.visualization.output_dir = "./visualizations"
```

## Tips

- Use high DPI (300) for publication-quality figures
- Save plots in multiple formats (PNG, PDF, SVG)
- Use consistent colormaps across visualizations
- Include colorbars and axis labels
- Add descriptive titles and legends

## Output Locations

Visualizations are saved to:
- `./visualizations/` (configurable)
- Organized by type: `training/`, `results/`, `brain/`, `model/`

## Next Steps

1. Generate plots after training completion
2. Use [results_analysis.ipynb](../notebooks/results_analysis.ipynb:1) for comprehensive analysis
3. Create publication-ready figures for reporting