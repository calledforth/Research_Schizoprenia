"""
Comprehensive training script for the SSPNet 3D CNN model
"""

import os
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau,
    TensorBoard,
    CSVLogger,
    LearningRateScheduler,
)
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.mixed_precision import experimental as mixed_precision
from ..models.sspnet_3d_cnn import create_sspnet_model
from ..models.loss_functions import (
    focal_loss,
    weighted_binary_crossentropy,
    combined_loss,
)
from ..models.metrics import Sensitivity, Specificity, AUC, AUPRC
from ..config import default_config
from ..utils.logging_utils import setup_logger


class ModelTrainer:
    """
    Comprehensive class for training the SSPNet 3D CNN model
    """

    def __init__(self, config=None):
        """
        Initialize the model trainer

        Args:
            config: Configuration object
        """
        self.config = config if config is not None else default_config
        self.model = None
        self.history = None
        self.logger = setup_logger(
            "trainer", self.config.logging.log_file, self.config.logging.level
        )

        # Enable mixed precision training if specified
        if self.config.training.mixed_precision:
            policy = mixed_precision.Policy("mixed_float16")
            mixed_precision.set_policy(policy)
            self.logger.info("Mixed precision training enabled")

    def build_model(self):
        """
        Build and compile the SSPNet 3D CNN model
        """
        self.logger.info("Building SSPNet 3D CNN model...")

        # Create the model
        self.model = create_sspnet_model(
            input_shape=self.config.model.input_shape,
            num_classes=self.config.model.num_classes,
        )

        # Get the optimizer
        optimizer = self._get_optimizer()

        # Get the loss function
        loss_fn = self._get_loss_function()

        # Get the metrics
        metrics = self._get_metrics()

        # Compile the model
        self.model.compile(
            optimizer=optimizer,
            loss=loss_fn,
            metrics=metrics,
        )

        # Log model summary
        self.model.model_summary()

        # Log parameter count
        param_counts = self.model.count_parameters()
        self.logger.info(f"Model parameters: {param_counts['total']} total")

        return self.model

    def _get_optimizer(self):
        """
        Get the optimizer based on configuration
        """
        optimizer_name = self.config.training.optimizer.lower()
        learning_rate = self.config.training.learning_rate

        if optimizer_name == "adam":
            return Adam(learning_rate=learning_rate)
        elif optimizer_name == "sgd":
            return SGD(learning_rate=learning_rate, momentum=0.9)
        elif optimizer_name == "rmsprop":
            return RMSprop(learning_rate=learning_rate)
        else:
            self.logger.warning(f"Unknown optimizer: {optimizer_name}. Using Adam.")
            return Adam(learning_rate=learning_rate)

    def _get_loss_function(self):
        """
        Get the loss function based on configuration
        """
        loss_name = self.config.training.loss_function.lower()

        if loss_name == "categorical_crossentropy":
            return "categorical_crossentropy"
        elif loss_name == "binary_crossentropy":
            return "binary_crossentropy"
        elif loss_name == "focal_loss":
            return focal_loss()
        elif loss_name == "weighted_binary_crossentropy":
            # Default class weights for schizophrenia detection
            class_weights = {0: 1.0, 1: 1.5}  # Higher weight for positive class
            return weighted_binary_crossentropy(class_weights)
        elif loss_name == "combined_loss":
            # Combine focal loss with weighted binary cross-entropy
            class_weights = {0: 1.0, 1: 1.5}
            return combined_loss(
                [focal_loss(), weighted_binary_crossentropy(class_weights)], [0.7, 0.3]
            )
        else:
            self.logger.warning(
                f"Unknown loss function: {loss_name}. Using focal_loss."
            )
            return focal_loss()

    def _get_metrics(self):
        """
        Get the metrics based on configuration
        """
        metrics = ["accuracy"]

        # Add custom metrics if specified in evaluation config
        if "sensitivity" in self.config.evaluation.metrics:
            metrics.append(Sensitivity())
        if "specificity" in self.config.evaluation.metrics:
            metrics.append(Specificity())
        if "auc" in self.config.evaluation.metrics:
            metrics.append(AUC())
        if "auprc" in self.config.evaluation.metrics:
            metrics.append(AUPRC())

        return metrics

    def setup_callbacks(self):
        """
        Setup training callbacks
        """
        callbacks = []

        # Create necessary directories
        os.makedirs(self.config.training.checkpoint_dir, exist_ok=True)
        os.makedirs(self.config.logging.tensorboard_dir, exist_ok=True)
        os.makedirs(os.path.dirname(self.config.logging.log_file), exist_ok=True)

        # Model checkpoint
        checkpoint_callback = ModelCheckpoint(
            filepath=os.path.join(
                self.config.training.checkpoint_dir,
                "model_{epoch:02d}_{val_accuracy:.4f}.h5",
            ),
            monitor="val_accuracy",
            save_best_only=self.config.training.save_best_only,
            mode="max",
            verbose=1,
            save_weights_only=False,
        )
        callbacks.append(checkpoint_callback)

        # Early stopping
        if self.config.training.early_stopping:
            early_stopping_callback = EarlyStopping(
                monitor="val_accuracy",
                patience=self.config.training.patience,
                mode="max",
                verbose=1,
                restore_best_weights=True,
            )
            callbacks.append(early_stopping_callback)

        # Learning rate scheduler
        if self.config.training.lr_scheduler == "reduce_on_plateau":
            lr_scheduler_callback = ReduceLROnPlateau(
                monitor="val_accuracy",
                factor=self.config.training.lr_factor,
                patience=self.config.training.lr_patience,
                mode="max",
                verbose=1,
                min_lr=1e-7,
            )
            callbacks.append(lr_scheduler_callback)
        elif self.config.training.lr_scheduler == "cosine_decay":
            # Cosine annealing learning rate scheduler
            def cosine_decay(epoch, lr):
                initial_lr = self.config.training.learning_rate
                epochs = self.config.training.epochs
                return initial_lr * (0.5 * (1 + np.cos(np.pi * epoch / epochs)))

            lr_scheduler_callback = LearningRateScheduler(cosine_decay, verbose=1)
            callbacks.append(lr_scheduler_callback)

        # TensorBoard
        if self.config.logging.tensorboard_enabled:
            tensorboard_callback = TensorBoard(
                log_dir=self.config.logging.tensorboard_dir,
                histogram_freq=1,
                write_graph=True,
                write_images=True,
                update_freq="epoch",
            )
            callbacks.append(tensorboard_callback)

        # CSV Logger
        csv_logger = CSVLogger(
            filename=os.path.join(
                self.config.logging.tensorboard_dir, "training_log.csv"
            ),
            append=True,
        )
        callbacks.append(csv_logger)

        return callbacks

    def train(self, train_data, val_data, class_weights=None):
        """
        Train the model with comprehensive logging and monitoring

        Args:
            train_data: Training data generator or dataset
            val_data: Validation data generator or dataset
            class_weights: Optional class weights for imbalanced data

        Returns:
            Training history
        """
        if self.model is None:
            self.build_model()

        callbacks = self.setup_callbacks()

        # Log training start
        self.logger.info("Starting model training...")
        start_time = time.time()

        # Train the model
        self.history = self.model.fit(
            train_data,
            epochs=self.config.training.epochs,
            validation_data=val_data,
            callbacks=callbacks,
            verbose=1,
            class_weight=class_weights,
        )

        # Log training completion
        end_time = time.time()
        training_time = end_time - start_time
        self.logger.info(f"Training completed in {training_time:.2f} seconds")

        # Log final metrics
        final_metrics = {}
        for metric in self.model.metrics_names:
            if metric in self.history.history:
                final_metrics[metric] = self.history.history[metric][-1]

        self.logger.info(f"Final training metrics: {final_metrics}")

        return self.history

    def save_model(self, path):
        """
        Save the trained model

        Args:
            path (str): Path to save the model
        """
        if self.model is not None:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(path), exist_ok=True)

            # Save the entire model
            self.model.save(path)
            self.logger.info(f"Model saved to {path}")
        else:
            self.logger.error("No model to save. Please train the model first.")

    def load_model(self, path):
        """
        Load a trained model

        Args:
            path (str): Path to the saved model
        """
        try:
            self.model = tf.keras.models.load_model(
                path,
                custom_objects={
                    "Sensitivity": Sensitivity,
                    "Specificity": Specificity,
                    "AUC": AUC,
                    "AUPRC": AUPRC,
                    "focal_loss": focal_loss(),
                },
            )
            self.logger.info(f"Model loaded from {path}")
            return self.model
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            raise

    def get_training_history(self):
        """
        Get the training history

        Returns:
            Training history dictionary
        """
        if self.history is not None:
            return self.history.history
        else:
            self.logger.warning(
                "No training history available. Please train the model first."
            )
            return None

    def plot_training_history(self, save_path=None):
        """
        Plot the training history

        Args:
            save_path (str): Path to save the plot
        """
        if self.history is None:
            self.logger.warning(
                "No training history available. Please train the model first."
            )
            return

        import matplotlib.pyplot as plt

        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Plot training & validation accuracy
        axes[0, 0].plot(self.history.history["accuracy"])
        if "val_accuracy" in self.history.history:
            axes[0, 0].plot(self.history.history["val_accuracy"])
        axes[0, 0].set_title("Model Accuracy")
        axes[0, 0].set_ylabel("Accuracy")
        axes[0, 0].set_xlabel("Epoch")
        axes[0, 0].legend(["Train", "Validation"], loc="upper left")

        # Plot training & validation loss
        axes[0, 1].plot(self.history.history["loss"])
        if "val_loss" in self.history.history:
            axes[0, 1].plot(self.history.history["val_loss"])
        axes[0, 1].set_title("Model Loss")
        axes[0, 1].set_ylabel("Loss")
        axes[0, 1].set_xlabel("Epoch")
        axes[0, 1].legend(["Train", "Validation"], loc="upper left")

        # Plot sensitivity if available
        if "sensitivity" in self.history.history:
            axes[1, 0].plot(self.history.history["sensitivity"])
            if "val_sensitivity" in self.history.history:
                axes[1, 0].plot(self.history.history["val_sensitivity"])
            axes[1, 0].set_title("Model Sensitivity")
            axes[1, 0].set_ylabel("Sensitivity")
            axes[1, 0].set_xlabel("Epoch")
            axes[1, 0].legend(["Train", "Validation"], loc="upper left")

        # Plot specificity if available
        if "specificity" in self.history.history:
            axes[1, 1].plot(self.history.history["specificity"])
            if "val_specificity" in self.history.history:
                axes[1, 1].plot(self.history.history["val_specificity"])
            axes[1, 1].set_title("Model Specificity")
            axes[1, 1].set_ylabel("Specificity")
            axes[1, 1].set_xlabel("Epoch")
            axes[1, 1].legend(["Train", "Validation"], loc="upper left")

        plt.tight_layout()

        if save_path:
            plt.savefig(
                save_path, dpi=self.config.visualization.dpi, bbox_inches="tight"
            )
            self.logger.info(f"Training history plot saved to {save_path}")

        plt.show()


def train_model(train_data, val_data, config=None, class_weights=None):
    """
    Convenience function to train the model

    Args:
        train_data: Training data
        val_data: Validation data
        config: Configuration object
        class_weights: Optional class weights for imbalanced data

    Returns:
        Trained model and training history
    """
    trainer = ModelTrainer(config)
    model = trainer.build_model()
    history = trainer.train(train_data, val_data, class_weights)

    return model, history, trainer
