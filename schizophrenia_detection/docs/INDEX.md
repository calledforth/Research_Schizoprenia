# Schizophrenia Detection Documentation

Welcome to the comprehensive documentation for the Schizophrenia Detection project. This collection of guides covers everything from setup to advanced usage of the SSPNet 3D CNN pipeline for multimodal neuroimaging analysis.

## Table of Contents

1. [Getting Started](#getting-started)
2. [Data Preparation](#data-preparation)
3. [Preprocessing](#preprocessing)
4. [Model Architecture](#model-architecture)
5. [Training and Evaluation](#training-and-evaluation)
6. [Visualization](#visualization)
7. [Advanced Topics](#advanced-topics)
8. [Troubleshooting](#troubleshooting)

## Getting Started

### [SETUP_COLAB.md](SETUP_COLAB.md)
Complete Google Colab setup guide including:
- GPU configuration and runtime selection
- Google Drive mounting and navigation
- Package installation and environment verification
- Memory optimization tips for large datasets
- Mixed precision training setup
- Session timeout handling strategies

### [REPRODUCIBILITY.md](REPRODUCIBILITY.md)
Ensuring reproducible results:
- Random seed configuration across frameworks
- Deterministic operations in TensorFlow and CUDA
- Environment and dependency management with exact version pinning
- Cross-validation protocol documentation
- Results archiving best practices

## Data Preparation

### [DATA_PREP.md](DATA_PREP.md)
Comprehensive data organization guide:
- Expected directory structure for Google Colab
- Supported input formats (fMRI .nii/.nii.gz, MEG .fif)
- Metadata CSV format and required fields
- FNC/SBM feature integration options
- Sample data loading scripts using [`data_loader.py`](../data_processing/data_loader.py:1)
- Data validation and quality checks

## Preprocessing

### [FMRI_PREPROCESSING.md](FMRI_PREPROCESSING.md)
Detailed fMRI preprocessing pipeline:
- Step-by-step preprocessing workflow
- Motion correction and realignment procedures
- Slice timing correction methods
- Spatial smoothing parameters
- Coregistration and normalization techniques
- Statistical Parametric Mapping (SPM) implementation
- SSP map generation and reverse phase de-ambiguization
- Code examples using [`fmri_preprocessing.py`](../data_processing/fmri_preprocessing.py:1)
- Expected outputs and quality metrics

### [MEG_PREPROCESSING.md](MEG_PREPROCESSING.md)
Comprehensive MEG preprocessing guide:
- Maxfilter configuration for artifact removal
- Signal-Space Projection (SSP) for cardiac and blink artifacts
- Bandpass filtering for delta, theta, alpha, and beta ranges
- Covariance matrix computation with regularization
- Source space setup using 6-mm³ grid
- Beamformer projection with single-shell BEM
- MNI normalization using FLIRT
- Hilbert transform for amplitude envelope extraction
- Spatial processing with smoothing and resampling
- Implementation examples from [`meg_preprocessing.py`](../data_processing/meg_preprocessing.py:1)
- Troubleshooting common MEG processing issues

## Model Architecture

### [MODEL_SSPNET.md](MODEL_SSPNET.md)
SSPNet 3D CNN architecture details:
- Model architecture overview and design principles
- Layer shapes and parameter budget
- Input volume assumptions and preprocessing requirements
- Model instantiation and configuration options
- Save/load functionality for model checkpoints
- Saliency map and Grad-CAM generation
- Training stability considerations and initialization
- Code examples using [`sspnet_3d_cnn.py`](../models/sspnet_3d_cnn.py:1)
- Performance optimization tips

## Training and Evaluation

### [TRAIN_EVAL.md](TRAIN_EVAL.md)
Complete training and evaluation workflow:
- Notebook-based training using [`model_training.ipynb`](../notebooks/model_training.ipynb:1)
- Command-line training options
- Cross-validation setup and execution
- Hyperparameter tuning strategies
- Metrics calculation (ACC, SEN, SPEC, ROC/AUC)
- Confusion matrix generation and interpretation
- Model evaluation using [`evaluator.py`](../training/evaluator.py:1)
- Results reporting and visualization
- Best practices for model selection

## Visualization

### [VISUALIZATION_GUIDE.md](VISUALIZATION_GUIDE.md)
Comprehensive visualization toolkit:
- Static result plots using [`result_plots.py`](../visualization/result_plots.py:1)
- Brain imaging visualization with [`brain_visualization.py`](../visualization/brain_visualization.py:1)
- Model architecture visualization via [`model_visualization.py`](../visualization/model_visualization.py:1)
- Interactive dashboards using [`interactive_plots.py`](../visualization/interactive_plots.py:1)
- Saliency maps and interpretability tools
- Expected input formats and output file patterns
- Custom visualization examples
- Publication-ready figure generation

## Advanced Topics

### [REPRODUCIBILITY.md](REPRODUCIBILITY.md)
Advanced reproducibility techniques:
- Docker containerization options
- Workflow automation with scripts
- Version control best practices
- Computational resource management
- Large dataset handling strategies

## Troubleshooting

### [FAQ.md](FAQ.md)
Frequently asked questions and solutions:
- Google Colab-specific issues (OOM, session timeouts)
- Data loading and format problems
- Model training convergence issues
- Visualization rendering problems
- Grad-CAM blank output solutions
- Shape mismatch debugging
- Performance optimization tips
- Where to find help and support

## Additional Resources

### Configuration
- Main configuration file: [`config.py`](../config.py:1)
- Path management and parameter tuning
- Environment-specific settings

### Notebooks
- [data_exploration.ipynb](../notebooks/data_exploration.ipynb:1) - Initial data exploration
- [model_training.ipynb](../notebooks/model_training.ipynb:1) - Training workflow
- [results_analysis.ipynb](../notebooks/results_analysis.ipynb:1) - Results interpretation

### API Reference
- Module documentation in source files
- Function signatures and parameter descriptions
- Usage examples and best practices

## Building Documentation Locally

To build and view the documentation locally:

```bash
pip install mkdocs mkdocs-material
mkdocs serve
```

Then open the provided local URL in your browser.

## Contributing to Documentation

We welcome contributions to the documentation! Please see the [Contributing Guidelines](../CONTRIBUTING.md:1) for details on:
- Documentation style guidelines
- Pull request process
- Code examples and testing
- Review process

## Getting Help

If you need help with the project:

1. Check the relevant documentation section above
2. Search the [FAQ](FAQ.md:1) for common issues
3. Open an issue on the GitHub repository
4. Join our community discussions

---

*This documentation is maintained alongside the codebase and updated with each release. For the latest version, see the project's GitHub repository.*