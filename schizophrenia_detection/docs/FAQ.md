# Frequently Asked Questions

## Google Colab Issues

### Q: Out of Memory (OOM) errors
**A**: Reduce batch size in [`config.py`](../config.py:1):
```python
config.data.batch_size = 2  # Reduce from 4
```
Enable mixed precision:
```python
config.training.mixed_precision = True
```

### Q: Session disconnects frequently
**A**: Keep tab active and save checkpoints:
```python
config.training.save_best_only = True
config.training.checkpoint_dir = "./checkpoints"
```

### Q: Slow Drive I/O
**A**: Cache data to local Colab storage:
```python
# Cache to /content/local_cache/
local_cache = "/content/local_cache"
os.makedirs(local_cache, exist_ok=True)
```

## Data Issues

### Q: Missing data files
**A**: Check file paths and naming:
```python
import os
expected_files = ["sub-001_func.nii.gz", "sub-001_meg.fif"]
for f in expected_files:
    if not os.path.exists(f"data/fmri/{f}"):
        print(f"Missing: {f}")
```

### Q: Shape mismatches
**A**: Verify input shapes match model:
```python
# Expected: (96, 96, 96, 4)
print(f"Input shape: {data.shape}")
```

### Q: Corrupted files
**A**: Validate file integrity:
```python
import nibabel as nib
try:
    img = nib.load("data/fmri/sub-001_func.nii.gz")
    data = img.get_fdata()
    assert not np.any(np.isnan(data))
except Exception as e:
    print(f"Corrupted file: {e}")
```

## Model Training

### Q: Poor convergence
**A**: Try these fixes:
- Lower learning rate: `config.training.learning_rate = 0.0001`
- Add dropout: `config.model.dropout_rate = 0.5`
- Enable early stopping: `config.training.early_stopping = True`

### Q: Overfitting
**A**: Regularization strategies:
- Data augmentation: `config.data.augmentation_enabled = True`
- Dropout: `config.model.dropout_rate = 0.5`
- Cross-validation: `config.evaluation.cv_enabled = True`

### Q: Training too slow
**A**: Speed up training:
- Mixed precision: `config.training.mixed_precision = True`
- Smaller batch size: `config.data.batch_size = 2`
- Fewer epochs: `config.training.epochs = 50`

## Visualization

### Q: Grad-CAM shows blank output
**A**: Check layer name and input:
```python
# Verify layer exists
print([layer.name for layer in model.layers])

# Use correct layer name
layer_name = "conv3d_3"  # Adjust as needed
```

### Q: Brain visualization fails
**A**: Install missing dependencies:
```python
!pip install nilearn
```

### Q: Interactive plots not working
**A**: Enable in Colab:
```python
%matplotlib inline
import plotly.graph_objects as go
```

## MEG Processing

### Q: FSL not available
**A**: MNI normalization will be skipped. Install FSL or use alternative methods.

### Q: MEG preprocessing fails
**A**: Check channel types:
```python
import mne
raw = mne.io.read_raw_fif("data/meg/sub-001_meg.fif")
print(f"MEG channels: {mne.pick_types(raw.info, meg=True)}")
```

## Configuration

### Q: Where are results saved?
**A**: Check [`config.py`](../config.py:1):
```python
config.evaluation.results_dir = "./results"
config.visualization.output_dir = "./visualizations"
config.training.checkpoint_dir = "./checkpoints"
```

### Q: How to change model parameters?
**A**: Edit configuration:
```python
config.model.num_spatial_filters = 64
config.model.dense_units = [1024, 512]
```

## Performance

### Q: Low accuracy
**A**: Improve performance:
- Increase data quality
- Add more training data
- Tune hyperparameters
- Use cross-validation

### Q: High variance in results
**A**: Stabilize results:
- Fix random seeds (see [REPRODUCIBILITY.md](REPRODUCIBILITY.md:1))
- Use cross-validation
- Increase dataset size

## Getting Help

### Q: Where to find more information?
**A**: 
- Check other documentation files
- Review notebook examples
- Examine error messages carefully
- Check GitHub issues

### Q: How to report bugs?
**A**: Include:
- Error message
- Configuration used
- Data format
- Steps to reproduce

## Quick Fixes

### Common Solutions
1. **Restart runtime** - Clear memory issues
2. **Reduce batch size** - Fix OOM errors
3. **Check file paths** - Fix missing file errors
4. **Update packages** - Fix compatibility issues
5. **Verify data format** - Fix shape mismatches

## Next Steps

1. Check specific guide for detailed help
2. Review example notebooks
3. Experiment with configuration parameters
4. Search GitHub issues for similar problems