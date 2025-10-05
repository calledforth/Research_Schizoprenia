"""
Configuration parameters for the schizophrenia detection model
"""

import os
from dataclasses import dataclass
from typing import Tuple, List


@dataclass
class DataConfig:
    """Configuration for data processing"""

    # Data paths
    data_root: str = "./data"
    fmri_data_dir: str = "./data/fmri"
    meg_data_dir: str = "./data/meg"

    # Data parameters
    fmri_shape: Tuple[int, int, int, int] = (96, 96, 96, 4)  # 3D spatial + time
    meg_shape: Tuple[int, int, int] = (306, 100, 1000)  # sensors x time x frequency

    # Preprocessing parameters
    fmri_voxel_size: Tuple[float, float, float] = (2.0, 2.0, 2.0)
    fmri_tr: float = 2.0  # Repetition time in seconds
    meg_lowpass: float = 40.0  # Hz
    meg_highpass: float = 0.1  # Hz

    # Data split
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    # Batch size
    batch_size: int = 4

    # Data augmentation
    augmentation_enabled: bool = True
    rotation_range: float = 10.0  # degrees
    zoom_range: float = 0.1
    brightness_range: List[float] = None


@dataclass
class ModelConfig:
    """Configuration for the SSPNet 3D CNN model"""

    # Model architecture
    input_shape: Tuple[int, int, int, int] = (96, 96, 96, 4)
    num_classes: int = 2  # Schizophrenia vs Control
    dropout_rate: float = 0.5

    # SSPNet specific parameters
    num_spatial_filters: int = 32
    num_spectral_filters: int = 16
    spatial_kernel_size: Tuple[int, int, int] = (3, 3, 3)
    spectral_kernel_size: Tuple[int, int, int] = (1, 1, 3)

    # Pooling parameters
    pool_size: Tuple[int, int, int] = (2, 2, 2)

    # Dense layers
    dense_units: List[int] = (512, 256)

    # Activation functions
    activation: str = "relu"
    final_activation: str = "softmax"


@dataclass
class TrainingConfig:
    """Configuration for training the model"""

    # Training parameters
    epochs: int = 100
    learning_rate: float = 0.001
    optimizer: str = "adam"
    loss_function: str = "categorical_crossentropy"

    # Learning rate scheduling
    lr_scheduler: str = "reduce_on_plateau"
    lr_patience: int = 10
    lr_factor: float = 0.5

    # Early stopping
    early_stopping: bool = True
    patience: int = 20

    # Model checkpointing
    save_best_only: bool = True
    checkpoint_dir: str = "./checkpoints"

    # Mixed precision training
    mixed_precision: bool = True

    # Distributed training
    distributed: bool = False


@dataclass
class EvaluationConfig:
    """Configuration for model evaluation"""

    # Metrics to compute
    metrics: List[str] = None

    # Cross-validation
    cv_folds: int = 5
    cv_enabled: bool = True

    # Threshold for binary classification
    classification_threshold: float = 0.5

    # Output directory
    results_dir: str = "./results"


@dataclass
class VisualizationConfig:
    """Configuration for visualization"""

    # Brain visualization
    brain_template: str = "MNI152NLin2009cAsym"
    brain_view: str = "axial"

    # Plotting parameters
    figure_size: Tuple[int, int] = (10, 8)
    dpi: int = 300

    # Color maps
    fmri_colormap: str = "viridis"
    meg_colormap: str = "plasma"

    # Output directory
    output_dir: str = "./visualizations"


@dataclass
class LoggingConfig:
    """Configuration for logging"""

    # Logging level
    level: str = "INFO"

    # Log file
    log_file: str = "./logs/schizophrenia_detection.log"

    # Console logging
    console_logging: bool = True

    # TensorBoard
    tensorboard_dir: str = "./logs/tensorboard"
    tensorboard_enabled: bool = True


class Config:
    """Main configuration class"""

    def __init__(self):
        self.data = DataConfig()
        self.model = ModelConfig()
        self.training = TrainingConfig()
        self.evaluation = EvaluationConfig()
        self.visualization = VisualizationConfig()
        self.logging = LoggingConfig()

        # Initialize evaluation metrics if not provided
        if self.evaluation.metrics is None:
            self.evaluation.metrics = [
                "accuracy",
                "precision",
                "recall",
                "f1_score",
                "auc",
            ]

        # Initialize brightness range if not provided
        if self.data.augmentation_enabled and self.data.brightness_range is None:
            self.data.brightness_range = [0.8, 1.2]

    def update_from_dict(self, config_dict: dict):
        """Update configuration from a dictionary"""
        for section, values in config_dict.items():
            if hasattr(self, section):
                section_obj = getattr(self, section)
                for key, value in values.items():
                    if hasattr(section_obj, key):
                        setattr(section_obj, key, value)

    def to_dict(self) -> dict:
        """Convert configuration to a dictionary"""
        return {
            "data": self.data.__dict__,
            "model": self.model.__dict__,
            "training": self.training.__dict__,
            "evaluation": self.evaluation.__dict__,
            "visualization": self.visualization.__dict__,
            "logging": self.logging.__dict__,
        }

    def save(self, path: str):
        """Save configuration to a file"""
        import json

        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str):
        """Load configuration from a file"""
        import json

        config = cls()
        with open(path, "r") as f:
            config_dict = json.load(f)
        config.update_from_dict(config_dict)
        return config


# Default configuration instance
default_config = Config()
