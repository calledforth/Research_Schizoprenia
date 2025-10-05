"""
SSPNet 3D CNN architecture for schizophrenia detection
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, backend as K
import matplotlib.pyplot as plt
import cv2
from typing import Tuple, Optional, Union, List


class SSPNet3DCNN(models.Model):
    """
    SSPNet 3D CNN model for schizophrenia detection

    This model implements a 3D CNN architecture specifically designed for analyzing
    3D SSP (Statistical Parametric Mapping) maps for schizophrenia detection.

    Architecture:
    - Two convolutional layers (3×3×3 kernels, 8 and 16 filters respectively)
    - Two max pooling layers (2×2×2 filter size)
    - Two fully connected layers (64 nodes each)
    - Output layer (2 nodes for binary classification)
    - ReLU activation for convolutional layers
    - Softmax activation for output layer

    Parameter counts:
    - Convolutional layers: 3,696 parameters
    - Fully Connected layers: 1,577,154 parameters
    - Batch Normalization: 256 parameters
    - Total: 1,581,106 parameters
    """

    def __init__(
        self,
        input_shape: Tuple[int, int, int, int] = (96, 96, 96, 1),
        num_classes: int = 2,
        **kwargs,
    ):
        """
        Initialize the SSPNet 3D CNN model

        Args:
            input_shape (tuple): Shape of input data (x, y, z, channels)
            num_classes (int): Number of output classes (default: 2 for binary classification)
            **kwargs: Additional keyword arguments
        """
        super(SSPNet3DCNN, self).__init__(**kwargs)
        self.input_shape = input_shape
        self.num_classes = num_classes

        # Define the model architecture
        self._build_model()

    def _build_model(self):
        """Build the model architecture with the specified layers"""
        # First convolutional block
        self.conv1 = layers.Conv3D(
            filters=8,
            kernel_size=(3, 3, 3),
            activation="relu",
            padding="same",
            kernel_initializer="he_normal",
            name="conv1",
        )
        self.pool1 = layers.MaxPooling3D(pool_size=(2, 2, 2), name="pool1")
        self.bn1 = layers.BatchNormalization(name="bn1")

        # Second convolutional block
        self.conv2 = layers.Conv3D(
            filters=16,
            kernel_size=(3, 3, 3),
            activation="relu",
            padding="same",
            kernel_initializer="he_normal",
            name="conv2",
        )
        self.pool2 = layers.MaxPooling3D(pool_size=(2, 2, 2), name="pool2")
        self.bn2 = layers.BatchNormalization(name="bn2")

        # Flatten layer
        self.flatten = layers.Flatten(name="flatten")

        # First fully connected layer
        self.fc1 = layers.Dense(
            units=64, activation="relu", kernel_initializer="he_normal", name="fc1"
        )
        self.dropout1 = layers.Dropout(0.5, name="dropout1")

        # Second fully connected layer
        self.fc2 = layers.Dense(
            units=64, activation="relu", kernel_initializer="he_normal", name="fc2"
        )
        self.dropout2 = layers.Dropout(0.5, name="dropout2")

        # Output layer
        self.output_layer = layers.Dense(
            units=self.num_classes,
            activation="softmax",
            kernel_initializer="he_normal",
            name="output_layer",
        )

    def call(self, inputs, training=None):
        """
        Forward pass of the model

        Args:
            inputs: Input tensor of shape (batch_size, x, y, z, channels)
            training: Whether in training mode

        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        # First convolutional block
        x = self.conv1(inputs)
        x = self.pool1(x)
        x = self.bn1(x, training=training)

        # Second convolutional block
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.bn2(x, training=training)

        # Flatten and fully connected layers
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.dropout1(x, training=training)
        x = self.fc2(x)
        x = self.dropout2(x, training=training)

        # Output layer
        outputs = self.output_layer(x)

        return outputs

    def model_summary(self):
        """Print a summary of the model architecture"""
        # Build a temporary model to show the summary
        x = layers.Input(shape=self.input_shape)
        model = models.Model(inputs=[x], outputs=self.call(x))
        model.summary()

    def count_parameters(self):
        """
        Count the total number of parameters in the model

        Returns:
            dict: Dictionary containing parameter counts for different layers
        """
        total_params = self.count_params()

        # Count parameters for each layer type
        conv_params = sum(
            [
                layer.count_params()
                for layer in self.layers
                if isinstance(layer, layers.Conv3D)
            ]
        )
        fc_params = sum(
            [
                layer.count_params()
                for layer in self.layers
                if isinstance(layer, layers.Dense)
            ]
        )
        bn_params = sum(
            [
                layer.count_params()
                for layer in self.layers
                if isinstance(layer, layers.BatchNormalization)
            ]
        )

        return {
            "total": total_params,
            "convolutional": conv_params,
            "fully_connected": fc_params,
            "batch_normalization": bn_params,
        }

    def generate_saliency_map(
        self, input_data: np.ndarray, class_idx: int = 0
    ) -> np.ndarray:
        """
        Generate a saliency map for the given input data

        Args:
            input_data: 3D SSP map of shape (x, y, z, channels)
            class_idx: Index of the class to generate saliency for

        Returns:
            Saliency map of the same shape as input_data
        """
        # Ensure input has batch dimension
        if len(input_data.shape) == 4:
            input_data = np.expand_dims(input_data, axis=0)

        # Convert to tensor
        input_tensor = tf.convert_to_tensor(input_data, dtype=tf.float32)

        with tf.GradientTape() as tape:
            tape.watch(input_tensor)
            predictions = self(input_tensor, training=False)
            loss = predictions[:, class_idx]

        # Compute gradients
        gradients = tape.gradient(loss, input_tensor)

        # Take absolute values and max across channels
        saliency_map = tf.abs(gradients[0])
        if len(saliency_map.shape) == 4:
            saliency_map = tf.reduce_max(saliency_map, axis=-1)

        # Normalize to [0, 1]
        saliency_map = (saliency_map - tf.reduce_min(saliency_map)) / (
            tf.reduce_max(saliency_map) - tf.reduce_min(saliency_map) + 1e-8
        )

        return saliency_map.numpy()

    def generate_grad_cam(
        self, input_data: np.ndarray, class_idx: int = 0, layer_name: str = "conv2"
    ) -> np.ndarray:
        """
        Generate a Grad-CAM heatmap for the given input data

        Args:
            input_data: 3D SSP map of shape (x, y, z, channels)
            class_idx: Index of the class to generate Grad-CAM for
            layer_name: Name of the layer to use for Grad-CAM

        Returns:
            Grad-CAM heatmap
        """
        # Ensure input has batch dimension
        if len(input_data.shape) == 4:
            input_data = np.expand_dims(input_data, axis=0)

        # Convert to tensor
        input_tensor = tf.convert_to_tensor(input_data, dtype=tf.float32)

        # Get the target layer
        target_layer = None
        for layer in self.layers:
            if layer.name == layer_name:
                target_layer = layer
                break

        if target_layer is None:
            raise ValueError(f"Layer '{layer_name}' not found in the model")

        # Create a model that outputs both the target layer and the predictions
        grad_model = models.Model(
            inputs=self.input, outputs=[target_layer.output, self.output]
        )

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(input_tensor)
            loss = predictions[:, class_idx]

        # Compute gradients
        grads = tape.gradient(loss, conv_outputs)

        # Global average pooling of gradients
        weights = tf.reduce_mean(grads, axis=[1, 2, 3, 4])

        # Weighted combination of activation maps
        cam = tf.reduce_sum(
            tf.multiply(
                weights[:, tf.newaxis, tf.newaxis, tf.newaxis, tf.newaxis], conv_outputs
            ),
            axis=-1,
        )

        # Apply ReLU
        cam = tf.nn.relu(cam)

        # Normalize to [0, 1]
        cam = cam[0]  # Remove batch dimension
        cam = (cam - tf.reduce_min(cam)) / (
            tf.reduce_max(cam) - tf.reduce_min(cam) + 1e-8
        )

        # Resize to original input size
        input_size = input_data.shape[1:4]  # x, y, z dimensions
        cam_resized = tf.expand_dims(cam, -1)  # Add channel dimension for resizing

        # Use numpy for resizing as TensorFlow doesn't have direct 3D resize
        cam_resized = cam_resized.numpy()
        resized_cam = np.zeros(input_size)

        for i in range(input_size[2]):  # Iterate through z dimension
            # Resize each 2D slice
            slice_2d = cv2.resize(
                cam_resized[:, :, i, 0], (input_size[1], input_size[0])
            )
            resized_cam[:, :, i] = slice_2d

        return resized_cam

    def visualize_saliency_map(
        self,
        input_data: np.ndarray,
        class_idx: int = 0,
        slice_idx: int = None,
        axis: int = 2,
    ):
        """
        Visualize the saliency map for the given input data

        Args:
            input_data: 3D SSP map of shape (x, y, z, channels)
            class_idx: Index of the class to generate saliency for
            slice_idx: Index of the slice to visualize (if None, middle slice)
            axis: Axis along which to take the slice (0: x, 1: y, 2: z)
        """
        saliency_map = self.generate_saliency_map(input_data, class_idx)

        # Remove channel dimension if present
        if len(saliency_map.shape) == 4:
            saliency_map = saliency_map[:, :, :, 0]

        # Determine slice to visualize
        if slice_idx is None:
            slice_idx = saliency_map.shape[axis] // 2

        # Take the slice
        if axis == 0:
            saliency_slice = saliency_map[slice_idx, :, :]
            input_slice = (
                input_data[slice_idx, :, :, 0]
                if len(input_data.shape) == 4
                else input_data[slice_idx, :, :]
            )
        elif axis == 1:
            saliency_slice = saliency_map[:, slice_idx, :]
            input_slice = (
                input_data[:, slice_idx, :, 0]
                if len(input_data.shape) == 4
                else input_data[:, slice_idx, :]
            )
        else:  # axis == 2
            saliency_slice = saliency_map[:, :, slice_idx]
            input_slice = (
                input_data[:, :, slice_idx, 0]
                if len(input_data.shape) == 4
                else input_data[:, :, slice_idx]
            )

        # Plot
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        # Original input
        axes[0].imshow(input_slice, cmap="gray")
        axes[0].set_title(f"Original Input (Slice {slice_idx}, Axis {axis})")
        axes[0].axis("off")

        # Saliency map
        im = axes[1].imshow(saliency_slice, cmap="hot")
        axes[1].set_title(f"Saliency Map (Class {class_idx})")
        axes[1].axis("off")
        plt.colorbar(im, ax=axes[1])

        plt.tight_layout()
        plt.show()

    def visualize_grad_cam(
        self,
        input_data: np.ndarray,
        class_idx: int = 0,
        slice_idx: int = None,
        axis: int = 2,
        layer_name: str = "conv2",
    ):
        """
        Visualize the Grad-CAM heatmap for the given input data

        Args:
            input_data: 3D SSP map of shape (x, y, z, channels)
            class_idx: Index of the class to generate Grad-CAM for
            slice_idx: Index of the slice to visualize (if None, middle slice)
            axis: Axis along which to take the slice (0: x, 1: y, 2: z)
            layer_name: Name of the layer to use for Grad-CAM
        """
        grad_cam = self.generate_grad_cam(input_data, class_idx, layer_name)

        # Remove channel dimension if present
        if len(input_data.shape) == 4:
            input_data = input_data[:, :, :, 0]

        # Determine slice to visualize
        if slice_idx is None:
            slice_idx = grad_cam.shape[axis] // 2

        # Take the slice
        if axis == 0:
            grad_cam_slice = grad_cam[slice_idx, :, :]
            input_slice = input_data[slice_idx, :, :]
        elif axis == 1:
            grad_cam_slice = grad_cam[:, slice_idx, :]
            input_slice = input_data[:, slice_idx, :]
        else:  # axis == 2
            grad_cam_slice = grad_cam[:, :, slice_idx]
            input_slice = input_data[:, :, slice_idx]

        # Plot
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # Original input
        axes[0].imshow(input_slice, cmap="gray")
        axes[0].set_title(f"Original Input (Slice {slice_idx}, Axis {axis})")
        axes[0].axis("off")

        # Grad-CAM heatmap
        im1 = axes[1].imshow(grad_cam_slice, cmap="jet")
        axes[1].set_title(f"Grad-CAM Heatmap (Class {class_idx})")
        axes[1].axis("off")
        plt.colorbar(im1, ax=axes[1])

        # Overlay
        axes[2].imshow(input_slice, cmap="gray")
        im2 = axes[2].imshow(grad_cam_slice, cmap="jet", alpha=0.5)
        axes[2].set_title(f"Overlay")
        axes[2].axis("off")
        plt.colorbar(im2, ax=axes[2])

        plt.tight_layout()
        plt.show()

    def save_model(self, filepath: str):
        """
        Save the model weights to the specified filepath

        Args:
            filepath: Path to save the model weights
        """
        self.save_weights(filepath)
        print(f"Model weights saved to {filepath}")

    def load_model(self, filepath: str):
        """
        Load model weights from the specified filepath

        Args:
            filepath: Path to load the model weights from
        """
        self.load_weights(filepath)
        print(f"Model weights loaded from {filepath}")

    def predict_class(self, input_data: np.ndarray) -> Tuple[int, float]:
        """
        Predict the class for the given input data

        Args:
            input_data: 3D SSP map of shape (x, y, z, channels) or (batch_size, x, y, z, channels)

        Returns:
            Tuple of (predicted_class, confidence)
        """
        # Ensure input has batch dimension
        if len(input_data.shape) == 4:
            input_data = np.expand_dims(input_data, axis=0)

        # Make prediction
        predictions = self.predict(input_data)
        predicted_class = np.argmax(predictions[0])
        confidence = np.max(predictions[0])

        return predicted_class, confidence


def create_sspnet_model(
    input_shape: Tuple[int, int, int, int] = (96, 96, 96, 1), num_classes: int = 2
) -> SSPNet3DCNN:
    """
    Create and return an SSPNet 3D CNN model

    Args:
        input_shape (tuple): Shape of input data (x, y, z, channels)
        num_classes (int): Number of output classes

    Returns:
        SSPNet3DCNN: Initialized SSPNet 3D CNN model
    """
    model = SSPNet3DCNN(input_shape=input_shape, num_classes=num_classes)

    # Build the model
    model.build(input_shape=(None,) + input_shape)

    # Print parameter counts
    param_counts = model.count_parameters()
    print(f"Model created with parameter counts:")
    print(f"  Convolutional layers: {param_counts['convolutional']} parameters")
    print(f"  Fully Connected layers: {param_counts['fully_connected']} parameters")
    print(f"  Batch Normalization: {param_counts['batch_normalization']} parameters")
    print(f"  Total: {param_counts['total']} parameters")

    return model


if __name__ == "__main__":
    # Example usage
    print("Creating SSPNet 3D CNN model...")
    model = create_sspnet_model(input_shape=(96, 96, 96, 1))

    # Print model summary
    print("\nModel Summary:")
    model.model_summary()

    # Create dummy input data
    dummy_input = np.random.random((1, 96, 96, 96, 1))

    # Test prediction
    print("\nTesting prediction...")
    predictions = model.predict(dummy_input)
    print(f"Prediction shape: {predictions.shape}")
    print(f"Prediction values: {predictions}")

    # Test class prediction
    predicted_class, confidence = model.predict_class(dummy_input[0])
    print(f"Predicted class: {predicted_class}, Confidence: {confidence:.4f}")

    # Test interpretability methods
    print("\nTesting interpretability methods...")
    saliency_map = model.generate_saliency_map(dummy_input[0])
    print(f"Saliency map shape: {saliency_map.shape}")

    grad_cam = model.generate_grad_cam(dummy_input[0])
    print(f"Grad-CAM shape: {grad_cam.shape}")

    print("\nSSPNet 3D CNN model implementation completed successfully!")
