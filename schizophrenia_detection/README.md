# Schizophrenia Detection with SSPNet 3D CNN (fMRI + MEG)

This repository implements a multimodal neuroimaging pipeline for schizophrenia detection using a 3D convolutional neural network based on SSPNet. The code processes fMRI and MEG data and trains a classifier to distinguish schizophrenia from control cohorts. It is designed to run reproducibly on Google Colab and locally. Core model: [sspnet_3d_cnn.SSPNet3DCNN()](schizophrenia_detection/models/sspnet_3d_cnn.py:1).

Note: This repository aligns with the methodology described in the associated research paper. See [CITATION.cff](schizophrenia_detection/CITATION.cff:1) for the preferred citation once metadata is added.

## Table of Contents

- Overview and Architecture
- Quickstart on Google Colab
- Dataset Preparation
- Training and Evaluation
- Artifacts, Figures, and Logs
- Troubleshooting (Colab)
- Documentation Index (/docs)
- Build Docs Locally (MkDocs)
- Contributing, Changelog, License, Citation

For detailed docs, start at [INDEX.md](schizophrenia_detection/docs/INDEX.md:1).

## High-level Architecture and Pipeline

Pipeline overview (see detailed guides in [/docs](schizophrenia_detection/docs/INDEX.md:1)):

```
+-------------------+      +------------------------------+      +-------------------+      +---------------------------+      +--------------------+
| Raw Data (fMRI,   | ---> | Preprocessing                | ---> | Features/Volumes  | ---> | SSPNet 3D CNN             | ---> | Metrics & Reports  |
| MEG, metadata)    |      | - fMRI: realign, STC,        |      | - fMRI SSP maps   |      | - Training (cross-val)    |      | - ACC/SEN/SPEC     |
|                   |      |   smooth, coreg, normalize   |      | - MEG envelopes   |      | - Evaluation              |      | - ROC/AUC, CM      |
+-------------------+      | - MEG: Maxfilter, SSP,       |      | - Normalized MNI  |      | - Checkpoints             |      | - Visualizations   |
                           |   bandpass, beamformer, MNI  |      |                   |      |                           |      |                    |
                           +------------------------------+      +-------------------+      +---------------------------+      +--------------------+
```

Key modules:
- fMRI preprocessing: [fmri_preprocessing.py](schizophrenia_detection/data_processing/fmri_preprocessing.py:1)
- MEG preprocessing: [meg_preprocessing.py](schizophrenia_detection/data_processing/meg_preprocessing.py:1)
- Data loading & utils: [data_loader.py](schizophrenia_detection/data_processing/data_loader.py:1), [file_utils.py](schizophrenia_detection/utils/file_utils.py:1), [data_utils.py](schizophrenia_detection/utils/data_utils.py:1)
- Model: [sspnet_3d_cnn.py](schizophrenia_detection/models/sspnet_3d_cnn.py:1), helpers in [model_utils.py](schizophrenia_detection/models/model_utils.py:1)
- Training/Eval: [trainer.py](schizophrenia_detection/training/trainer.py:1), [evaluator.py](schizophrenia_detection/training/evaluator.py:1), [cross_validation.py](schizophrenia_detection/training/cross_validation.py:1), [hyperparameter_tuning.py](schizophrenia_detection/training/hyperparameter_tuning.py:1)
- Visualization: [result_plots.py](schizophrenia_detection/visualization/result_plots.py:1), [brain_visualization.py](schizophrenia_detection/visualization/brain_visualization.py:1), [model_visualization.py](schizophrenia_detection/visualization/model_visualization.py:1), [interactive_plots.py](schizophrenia_detection/visualization/interactive_plots.py:1)

Notebooks (Colab-friendly):
- Exploration: [data_exploration.ipynb](schizophrenia_detection/notebooks/data_exploration.ipynb:1)
- Training: [model_training.ipynb](schizophrenia_detection/notebooks/model_training.ipynb:1)
- Analysis: [results_analysis.ipynb](schizophrenia_detection/notebooks/results_analysis.ipynb:1)

## Quickstart on Google Colab

1) Runtime and GPU
- Runtime -> Change runtime type -> Hardware accelerator: GPU (T4/A100 if available)

2) Mount Drive and navigate
```python
from google.colab import drive
drive.mount('/content/drive')
%cd /content/drive/MyDrive/schizophrenia_detection
```

3) Install dependencies
```python
!pip install -r requirements.txt
# Optional: tools for docs and visualization in Colab
!pip install mkdocs mkdocs-material
```

4) Verify environment
```python
import tensorflow as tf; print('TF:', tf.__version__, 'GPUs:', tf.config.list_physical_devices('GPU'))
import mne, nilearn, nibabel as nib; print('MNE:', mne.__version__, 'Nilearn:', nilearn.__version__, 'Nibabel:', nib.__version__)
```

5) Run notebooks in order
- 1. Exploration: open [data_exploration.ipynb](schizophrenia_detection/notebooks/data_exploration.ipynb:1)
- 2. Training: open [model_training.ipynb](schizophrenia_detection/notebooks/model_training.ipynb:1)
- 3. Analysis: open [results_analysis.ipynb](schizophrenia_detection/notebooks/results_analysis.ipynb:1)

See the detailed Colab setup guide: [SETUP_COLAB.md](schizophrenia_detection/docs/SETUP_COLAB.md:1).

## Dataset Preparation (summary)

Default paths are configured in [config.py](schizophrenia_detection/config.py:1) under `Config().data`:
- data_root: ./data
- fmri_data_dir: ./data/fmri
- meg_data_dir: ./data/meg

Recommended Drive layout when using Colab (after %cd to project root):
```
/content/drive/MyDrive/schizophrenia_detection/
├── data/
│   ├── fmri/                # NIfTI (.nii/.nii.gz)
│   ├── meg/                 # FIF/CTF/VHDR, etc.
│   └── metadata/
│       └── participants.csv # subject_id,label,age,sex,site,...
├── checkpoints/
├── results/
├── visualizations/
└── logs/
```

Supported inputs:
- fMRI: NIfTI (.nii, .nii.gz)
- MEG: FIF (.fif) and selected vendor formats
- Optional precomputed features: FNC/SBM matrices or ROI timeseries (if available)

Detailed guide: [DATA_PREP.md](schizophrenia_detection/docs/DATA_PREP.md:1).

## Training and Evaluation (notebooks)

Primary workflow uses notebooks:
- Train model: [model_training.ipynb](schizophrenia_detection/notebooks/model_training.ipynb:1)
- Evaluate and visualize: [results_analysis.ipynb](schizophrenia_detection/notebooks/results_analysis.ipynb:1)

Programmatic entry points (advanced users):
- Model: [sspnet_3d_cnn.SSPNet3DCNN()](schizophrenia_detection/models/sspnet_3d_cnn.py:1)
- Training: [trainer.py](schizophrenia_detection/training/trainer.py:1), evaluation: [evaluator.py](schizophrenia_detection/training/evaluator.py:1)
- Cross-validation: [cross_validation.py](schizophrenia_detection/training/cross_validation.py:1), hyperparams: [hyperparameter_tuning.py](schizophrenia_detection/training/hyperparameter_tuning.py:1)

Metrics reported: Accuracy, Sensitivity (Recall for positive class), Specificity, ROC/AUC, and confusion matrix. See [TRAIN_EVAL.md](schizophrenia_detection/docs/TRAIN_EVAL.md:1).

## Where outputs are saved (from config)

Configuration: [config.py](schizophrenia_detection/config.py:1) and `Config().*`:
- Checkpoints: `training.checkpoint_dir` → ./checkpoints
- Results (metrics/reports): `evaluation.results_dir` → ./results
- Visualizations: `visualization.output_dir` → ./visualizations
- Logs: `logging.log_file` → ./logs/schizophrenia_detection.log
- TensorBoard: `logging.tensorboard_dir` → ./logs/tensorboard

Update paths by editing [config.py](schizophrenia_detection/config.py:1) or loading/saving a JSON config via `Config.save()`/`Config.load()`.

## Troubleshooting (Colab)

- OOM on GPU/CPU:
  - Lower `data.batch_size` in [config.py](schizophrenia_detection/config.py:1)
  - Enable/keep mixed precision: `training.mixed_precision=True`
  - Reduce input sizes or use fewer timepoints/voxels
- Session timeouts/disconnects:
  - Periodically save checkpoints to Drive (`save_best_only=True`)
  - Keep Colab tab active; consider shorter `epochs`
- Slow Drive I/O:
  - Cache intermediate files under `/content` and sync important outputs back to Drive
- MNE/FSL tools unavailable:
  - The pipeline degrades gracefully where possible (e.g., MNI normalization for MEG uses placeholders when FSL is missing). See [MEG_PREPROCESSING.md](schizophrenia_detection/docs/MEG_PREPROCESSING.md:1).
- Grad-CAM/Saliency shows blank maps:
  - Verify correct input shapes and layer names; see [MODEL_SSPNET.md](schizophrenia_detection/docs/MODEL_SSPNET.md:1)
- Shape mismatches:
  - Confirm `model.input_shape` matches data (default (96,96,96,4)) in [config.py](schizophrenia_detection/config.py:1)

## Documentation

The full documentation set lives in [/docs](schizophrenia_detection/docs/INDEX.md:1):
- Colab setup: [SETUP_COLAB.md](schizophrenia_detection/docs/SETUP_COLAB.md:1)
- Data prep: [DATA_PREP.md](schizophrenia_detection/docs/DATA_PREP.md:1)
- fMRI preprocessing: [FMRI_PREPROCESSING.md](schizophrenia_detection/docs/FMRI_PREPROCESSING.md:1)
- MEG preprocessing: [MEG_PREPROCESSING.md](schizophrenia_detection/docs/MEG_PREPROCESSING.md:1)
- SSPNet model: [MODEL_SSPNET.md](schizophrenia_detection/docs/MODEL_SSPNET.md:1)
- Train & eval: [TRAIN_EVAL.md](schizophrenia_detection/docs/TRAIN_EVAL.md:1)
- Visualization: [VISUALIZATION_GUIDE.md](schizophrenia_detection/docs/VISUALIZATION_GUIDE.md:1)
- Reproducibility: [REPRODUCIBILITY.md](schizophrenia_detection/docs/REPRODUCIBILITY.md:1)
- FAQ: [FAQ.md](schizophrenia_detection/docs/FAQ.md:1)

## Build docs locally (MkDocs)

The repository includes a MkDocs configuration to browse docs locally.
```bash
pip install mkdocs mkdocs-material
mkdocs serve
```
Then open the printed local URL in a browser.

## Contributing, Changelog, License, Citation

- Contribution guide: [CONTRIBUTING.md](schizophrenia_detection/CONTRIBUTING.md:1)
- Changelog: [CHANGELOG.md](schizophrenia_detection/CHANGELOG.md:1)
- License: [LICENSE](schizophrenia_detection/LICENSE:1)
- Citation: [CITATION.cff](schizophrenia_detection/CITATION.cff:1)

---

Contact: open an issue or PR. For email-based queries, add your contact in [README.md](schizophrenia_detection/README.md:1).