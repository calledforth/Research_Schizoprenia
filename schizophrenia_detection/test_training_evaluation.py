"""
Test script for the training and evaluation components of the SSPNet model
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical

# Import our modules
from config import default_config
from models.sspnet_3d_cnn import create_sspnet_model
from training.trainer import ModelTrainer
from training.evaluator import ModelEvaluator
from models.loss_functions import get_loss_function
from models.metrics import get_metric


def create_dummy_data(num_samples=20, input_shape=(96, 96, 96, 1), num_classes=2):
    """
    Create dummy data for testing

    Args:
        num_samples (int): Number of samples to generate
        input_shape (tuple): Shape of input data
        num_classes (int): Number of classes

    Returns:
        Tuple of (X, y) data
    """
    # Generate random input data
    X = np.random.random((num_samples,) + input_shape).astype(np.float32)

    # Generate random labels (50% each class for balanced dataset)
    y = np.random.randint(0, num_classes, size=num_samples)

    # Convert to one-hot encoding if needed
    if num_classes > 1:
        y = to_categorical(y, num_classes=num_classes)

    return X, y


def create_tf_dataset(X, y, batch_size=4, shuffle=True):
    """
    Create a TensorFlow dataset from numpy arrays

    Args:
        X: Input data
        y: Labels
        batch_size (int): Batch size
        shuffle (bool): Whether to shuffle the data

    Returns:
        TensorFlow dataset
    """
    dataset = tf.data.Dataset.from_tensor_slices((X, y))

    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(X))

    dataset = dataset.batch(batch_size)
    return dataset


def test_loss_functions():
    """Test the loss functions"""
    print("Testing loss functions...")

    # Create dummy data
    y_true = np.random.random((10, 1))
    y_pred = np.random.random((10, 1))

    # Test different loss functions
    loss_names = [
        "binary_crossentropy",
        "focal_loss",
        "binary_focal_loss",
        "dice_loss",
        "tversky_loss",
        "focal_tversky_loss",
    ]

    for loss_name in loss_names:
        try:
            loss_fn = get_loss_function(loss_name)
            loss_value = loss_fn(y_true, y_pred)
            print(f"  {loss_name}: {loss_value.numpy():.4f}")
        except Exception as e:
            print(f"  {loss_name}: Error - {str(e)}")

    print("Loss functions test completed.\n")


def test_metrics():
    """Test the metrics"""
    print("Testing metrics...")

    # Create dummy data
    y_true = np.random.randint(0, 2, size=(10,))
    y_pred = np.random.random((10,))

    # Test different metrics
    metric_names = [
        "accuracy",
        "precision",
        "sensitivity",
        "specificity",
        "f1_score",
        "auc",
    ]

    for metric_name in metric_names:
        try:
            metric = get_metric(metric_name)
            if isinstance(metric, str):
                # For built-in metrics like 'accuracy'
                if metric_name == "accuracy":
                    metric_value = calculate_accuracy(y_true, y_pred)
                    print(f"  {metric_name}: {metric_value:.4f}")
            else:
                # For custom metrics
                metric.update_state(y_true, y_pred)
                metric_value = metric.result().numpy()
                print(f"  {metric_name}: {metric_value:.4f}")
                metric.reset_states()
        except Exception as e:
            print(f"  {metric_name}: Error - {str(e)}")

    print("Metrics test completed.\n")


def test_model_creation():
    """Test model creation"""
    print("Testing model creation...")

    try:
        # Create model
        model = create_sspnet_model(
            input_shape=default_config.model.input_shape,
            num_classes=default_config.model.num_classes,
        )

        # Print model summary
        print("Model created successfully!")
        print(f"Model input shape: {model.input_shape}")
        print(f"Model output shape: {model.output_shape}")

        # Count parameters
        param_counts = model.count_parameters()
        print(f"Total parameters: {param_counts['total']}")

        return model
    except Exception as e:
        print(f"Error creating model: {str(e)}")
        return None

    print("Model creation test completed.\n")


def test_training():
    """Test model training"""
    print("Testing model training...")

    try:
        # Create dummy data
        X_train, y_train = create_dummy_data(num_samples=20)
        X_val, y_val = create_dummy_data(num_samples=10)

        # Create datasets
        train_dataset = create_tf_dataset(X_train, y_train, batch_size=4)
        val_dataset = create_tf_dataset(X_val, y_val, batch_size=4, shuffle=False)

        # Create trainer
        trainer = ModelTrainer(config=default_config)

        # Build model
        model = trainer.build_model()

        # Train for just 2 epochs for testing
        history = trainer.train(train_dataset, val_dataset)

        print("Training completed successfully!")
        print(f"Training history keys: {list(history.history.keys())}")

        return model, history
    except Exception as e:
        print(f"Error during training: {str(e)}")
        return None, None

    print("Training test completed.\n")


def test_evaluation():
    """Test model evaluation"""
    print("Testing model evaluation...")

    try:
        # Create dummy data
        X_test, y_test = create_dummy_data(num_samples=10)

        # Create dataset
        test_dataset = create_tf_dataset(X_test, y_test, batch_size=4, shuffle=False)

        # Create a simple model for testing
        model = create_sspnet_model(
            input_shape=default_config.model.input_shape,
            num_classes=default_config.model.num_classes,
        )

        # Compile the model
        model.compile(
            optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]
        )

        # Create evaluator
        evaluator = ModelEvaluator(model, config=default_config)

        # Evaluate the model
        results = evaluator.evaluate(test_dataset)

        print("Evaluation completed successfully!")
        print(f"Evaluation results keys: {list(results.keys())}")
        print(f"Accuracy: {results['accuracy']:.4f}")
        print(f"Sensitivity: {results['sensitivity']:.4f}")
        print(f"Specificity: {results['specificity']:.4f}")
        print(f"AUC: {results['auc']:.4f}")

        return results
    except Exception as e:
        print(f"Error during evaluation: {str(e)}")
        return None

    print("Evaluation test completed.\n")


def calculate_accuracy(y_true, y_pred, threshold=0.5):
    """Calculate accuracy for testing"""
    y_pred_classes = (y_pred > threshold).astype(int)
    return (y_pred_classes == y_true).mean()


def main():
    """Main test function"""
    print("=" * 60)
    print("TESTING TRAINING AND EVALUATION COMPONENTS")
    print("=" * 60)

    # Set random seed for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)

    # Test loss functions
    test_loss_functions()

    # Test metrics
    test_metrics()

    # Test model creation
    model = test_model_creation()

    if model is not None:
        # Test training
        trained_model, history = test_training()

        if trained_model is not None:
            # Test evaluation
            results = test_evaluation()

    print("=" * 60)
    print("ALL TESTS COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()
