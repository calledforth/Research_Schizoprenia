"""
Test script for the data processing implementation
"""

import os
import sys
import numpy as np
import logging

# Add the project directory to the path
sys.path.append("schizophrenia_detection")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_data_loader():
    """Test the data loader implementation"""
    logger.info("Testing data loader...")

    try:
        from schizophrenia_detection.data_processing.data_loader import (
            load_fmri_data,
            load_meg_data,
            load_labels,
            load_all_subjects,
            create_data_generator,
            create_3d_data_generator,
            split_data,
        )

        # Test loading functions
        data_root = "data/raw/mlsp_2014"

        # Load all subjects
        subjects_data = load_all_subjects(data_root)
        logger.info(f"Loaded {len(subjects_data)} subjects")

        # Check first subject
        first_subject = subjects_data[0]
        logger.info(f"First subject ID: {first_subject['subject_id']}")
        logger.info(f"FNC features shape: {first_subject['fnc_features'].shape}")
        logger.info(f"SBM features shape: {first_subject['sbm_features'].shape}")
        logger.info(f"Label: {first_subject['label']}")

        # Test data splitting
        train_data, val_data, test_data = split_data(subjects_data)
        logger.info(
            f"Data split: {len(train_data)} train, {len(val_data)} val, {len(test_data)} test"
        )

        # Test data generator
        generator = create_data_generator(train_data[:10], batch_size=2)
        features, labels = next(generator)
        logger.info(f"Generator batch features shape: {features.shape}")
        logger.info(f"Generator batch labels shape: {labels.shape}")

        # Test 3D data generator
        generator_3d = create_3d_data_generator(
            train_data[:5], batch_size=2, conversion_method="pca"
        )
        features_3d, labels_3d = next(generator_3d)
        logger.info(f"3D Generator batch features shape: {features_3d.shape}")
        logger.info(f"3D Generator batch labels shape: {labels_3d.shape}")

        logger.info("Data loader test passed!")
        return True

    except Exception as e:
        logger.error(f"Data loader test failed: {str(e)}")
        return False


def test_data_augmentation():
    """Test the data augmentation implementation"""
    logger.info("Testing data augmentation...")

    try:
        from schizophrenia_detection.data_processing.data_augmentation import (
            apply_rotation,
            apply_zoom,
            apply_brightness_adjustment,
            add_gaussian_noise,
            apply_random_flip,
            augment_features,
        )

        # Create dummy features
        features = np.random.randn(100)

        # Test individual augmentations
        rotated = apply_rotation(features, (-5, 5))
        logger.info(f"Rotation test passed: {rotated.shape}")

        zoomed = apply_zoom(features, (0.9, 1.1))
        logger.info(f"Zoom test passed: {zoomed.shape}")

        brightness = apply_brightness_adjustment(features, (0.9, 1.1))
        logger.info(f"Brightness test passed: {brightness.shape}")

        noisy = add_gaussian_noise(features, 0.0, 0.01)
        logger.info(f"Noise test passed: {noisy.shape}")

        flipped = apply_random_flip(features, axis=0)
        logger.info(f"Flip test passed: {flipped.shape}")

        # Test combined augmentation
        augmented = augment_features(features)
        logger.info(f"Combined augmentation test passed: {augmented.shape}")

        logger.info("Data augmentation test passed!")
        return True

    except Exception as e:
        logger.error(f"Data augmentation test failed: {str(e)}")
        return False


def test_feature_to_3d():
    """Test the feature to 3D conversion implementation"""
    logger.info("Testing feature to 3D conversion...")

    try:
        from schizophrenia_detection.data_processing.feature_to_3d import (
            FeatureTo3DConverter,
            convert_features_to_3d,
            batch_convert_features_to_3d,
        )

        # Create dummy features
        features = np.random.randn(200)

        # Test individual conversion
        features_3d = convert_features_to_3d(features, method="pca")
        logger.info(f"Individual conversion test passed: {features_3d.shape}")

        # Test batch conversion
        features_list = [np.random.randn(200) for _ in range(10)]
        features_3d_batch = batch_convert_features_to_3d(features_list, method="pca")
        logger.info(f"Batch conversion test passed: {features_3d_batch.shape}")

        # Test converter class
        converter = FeatureTo3DConverter(method="pca")
        converter.fit(features_list)
        converted = converter.transform(features)
        logger.info(f"Converter class test passed: {converted.shape}")

        logger.info("Feature to 3D conversion test passed!")
        return True

    except Exception as e:
        logger.error(f"Feature to 3D conversion test failed: {str(e)}")
        return False


def test_integration():
    """Test integration with SSPNet model"""
    logger.info("Testing integration with SSPNet model...")

    try:
        from schizophrenia_detection.models.sspnet_3d_cnn import create_sspnet_model
        from schizophrenia_detection.data_processing.data_loader import (
            load_all_subjects,
            create_3d_data_generator,
        )

        # Load a small subset of data
        data_root = "data/raw/mlsp_2014"
        subjects_data = load_all_subjects(data_root)[
            :10
        ]  # Use only 10 subjects for testing

        # Create 3D data generator
        generator = create_3d_data_generator(
            subjects_data, batch_size=2, conversion_method="pca"
        )

        # Get a batch
        features_3d, labels = next(generator)

        # Create SSPNet model
        model = create_sspnet_model(input_shape=(96, 96, 96, 1))

        # Test prediction
        predictions = model.predict(features_3d)
        logger.info(f"Model prediction test passed: {predictions.shape}")

        logger.info("Integration test passed!")
        return True

    except Exception as e:
        logger.error(f"Integration test failed: {str(e)}")
        return False


def main():
    """Run all tests"""
    logger.info("Starting implementation tests...")

    tests = [
        test_data_loader,
        test_data_augmentation,
        test_feature_to_3d,
        test_integration,
    ]

    results = []
    for test in tests:
        results.append(test())
        logger.info("-" * 50)

    # Summary
    passed = sum(results)
    total = len(results)
    logger.info(f"Tests completed: {passed}/{total} passed")

    if passed == total:
        logger.info("All tests passed! Implementation is working correctly.")
    else:
        logger.error(f"{total - passed} tests failed. Please check the implementation.")

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
