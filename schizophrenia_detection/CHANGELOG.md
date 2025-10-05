# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2025-10-05

### Added
- Initial release of schizophrenia detection pipeline
- Complete fMRI preprocessing pipeline with SSP map generation
- Complete MEG preprocessing pipeline with beamformer and source localization
- SSPNet 3D CNN architecture implementation
- Training and evaluation framework with cross-validation
- Comprehensive visualization tools for results and brain imaging
- Google Colab compatibility with notebook workflows
- Configuration system for easy parameter management
- Documentation suite with detailed guides
- MIT License
- Contributing guidelines
- Citation file with project metadata

### Features
- **fMRI Processing**
  - Motion correction and realignment
  - Slice timing correction
  - Spatial smoothing
  - Coregistration and normalization
  - Statistical Parametric Mapping (SPM)
  - Spatial Source Phase (SSP) map generation
  - Reverse phase de-ambiguization

- **MEG Processing**
  - Maxfilter for artifact removal and head movement correction
  - Signal-Space Projection (SSP) for cardiac and blink artifacts
  - Bandpass filtering for delta, theta, alpha, and beta frequency ranges
  - Covariance matrix processing with regularization
  - Source space processing using 6-mm³ grid
  - Beamformer projection with single-shell BEM
  - MNI normalization using FLIRT
  - Hilbert transform for amplitude envelopes
  - Spatial processing with smoothing and resampling

- **Model Architecture**
  - SSPNet 3D CNN with spatial and spectral filters
  - Configurable input shapes and parameters
  - Mixed precision training support
  - Model checkpointing and serialization

- **Training and Evaluation**
  - Cross-validation framework
  - Hyperparameter tuning utilities
  - Comprehensive metrics (ACC, SEN, SPEC, ROC/AUC)
  - Early stopping and learning rate scheduling
  - Distributed training support

- **Visualization**
  - Static plots for results
  - Brain imaging visualization
  - Model architecture visualization
  - Interactive plotting tools
  - Saliency maps and Grad-CAM

- **Documentation**
  - Comprehensive README with quickstart guide
  - Detailed documentation in /docs directory
  - Colab setup instructions
  - API documentation
  - FAQ and troubleshooting guide

### Technical Details
- Python 3.8+ compatibility
- TensorFlow/Keras for deep learning
- MNE-Python for MEG processing
- Nilearn for fMRI processing
- NiBabel for neuroimaging I/O
- NumPy, SciPy for numerical computations
- Matplotlib, Plotly for visualization

### Notebooks
- `data_exploration.ipynb` - Data exploration and visualization
- `model_training.ipynb` - Model training workflow
- `results_analysis.ipynb` - Results analysis and interpretation

### Configuration
- Centralized configuration system in `config.py`
- Support for JSON-based configuration files
- Configurable data paths, model parameters, and training settings

### Project Structure
- Modular design with clear separation of concerns
- Organized into logical modules (data_processing, models, training, visualization, utils)
- Comprehensive error handling and logging

### Known Limitations
- MNI normalization for MEG requires FSL (graceful degradation when unavailable)
- Large memory requirements for high-resolution fMRI data
- Training time can be significant for large datasets

### Future Plans
- Support for additional neuroimaging modalities
- Advanced model architectures
- Cloud deployment options
- Real-time processing capabilities
- Extended documentation and tutorials