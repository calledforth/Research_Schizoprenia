"""
Test script for visualization utilities
"""

import numpy as np
import os
import sys
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from schizophrenia_detection.visualization.result_plots import ResultPlotter
from schizophrenia_detection.visualization.brain_visualization import BrainVisualizer
from schizophrenia_detection.visualization.model_visualization import ModelVisualizer
from schizophrenia_detection.visualization.interactive_plots import InteractivePlotter
from schizophrenia_detection.models.sspnet_3d_cnn import create_sspnet_model


def test_result_plots():
    """Test result plotting functionality"""
    print("Testing result plots...")

    # Create dummy data
    y_true = np.random.randint(0, 2, 100)
    y_pred = np.random.randint(0, 2, 100)
    y_score = np.random.random(100)

    # Create training history
    history = {
        "loss": np.random.random(50) + 0.5,
        "val_loss": np.random.random(50) + 0.5,
        "accuracy": np.random.random(50) * 0.5 + 0.5,
        "val_accuracy": np.random.random(50) * 0.5 + 0.5,
    }

    # Create evaluator results
    evaluator_results = {
        "y_true": y_true,
        "y_pred": y_pred,
        "y_pred_proba": y_score,
        "accuracy": 0.75,
        "sensitivity": 0.80,
        "specificity": 0.70,
        "auc": 0.85,
    }

    # Initialize plotter
    plotter = ResultPlotter()

    # Test training curves
    try:
        plotter.plot_training_curves(
            history, save_path="test_outputs/training_curves.png"
        )
        print("✓ Training curves plot created successfully")
    except Exception as e:
        print(f"✗ Training curves plot failed: {e}")

    # Test confusion matrix
    try:
        plotter.plot_confusion_matrix(
            y_true, y_pred, save_path="test_outputs/confusion_matrix.png"
        )
        print("✓ Confusion matrix plot created successfully")
    except Exception as e:
        print(f"✗ Confusion matrix plot failed: {e}")

    # Test ROC curve
    try:
        plotter.plot_roc_curve(y_true, y_score, save_path="test_outputs/roc_curve.png")
        print("✓ ROC curve plot created successfully")
    except Exception as e:
        print(f"✗ ROC curve plot failed: {e}")

    # Test metrics summary
    try:
        plotter.plot_metrics_summary(
            evaluator_results, save_path="test_outputs/metrics_summary.png"
        )
        print("✓ Metrics summary plot created successfully")
    except Exception as e:
        print(f"✗ Metrics summary plot failed: {e}")

    # Test report generation
    try:
        figure_paths = plotter.generate_report_figure_set(
            evaluator_results, output_dir="test_outputs"
        )
        print("✓ Evaluation report generated successfully")
        print(f"  Generated files: {list(figure_paths.keys())}")
    except Exception as e:
        print(f"✗ Evaluation report generation failed: {e}")


def test_brain_visualization():
    """Test brain visualization functionality"""
    print("\nTesting brain visualization...")

    # Create dummy brain data
    volume_data = np.random.random((96, 96, 96))
    grad_cam_data = np.random.random((48, 48, 48))
    saliency_data = np.random.random((48, 48, 48))
    fnc_matrix = np.random.random((20, 20)) * 2 - 1  # Random connectivity matrix
    np.fill_diagonal(fnc_matrix, 1)  # Set diagonal to 1

    # Initialize visualizer
    visualizer = BrainVisualizer()

    # Test Grad-CAM slices
    try:
        visualizer.plot_grad_cam_slices(
            grad_cam_data, save_path="test_outputs/gradcam_slices.png"
        )
        print("✓ Grad-CAM slices plot created successfully")
    except Exception as e:
        print(f"✗ Grad-CAM slices plot failed: {e}")

    # Test saliency slices
    try:
        visualizer.plot_saliency_slices(
            saliency_data, save_path="test_outputs/saliency_slices.png"
        )
        print("✓ Saliency slices plot created successfully")
    except Exception as e:
        print(f"✗ Saliency slices plot failed: {e}")

    # Test slice montage
    try:
        visualizer.plot_slice_montage(
            volume_data, save_path="test_outputs/slice_montage.png"
        )
        print("✓ Slice montage plot created successfully")
    except Exception as e:
        print(f"✗ Slice montage plot failed: {e}")

    # Test FNC heatmap
    try:
        visualizer.plot_fnc_heatmap(
            fnc_matrix, save_path="test_outputs/fnc_heatmap.png"
        )
        print("✓ FNC heatmap plot created successfully")
    except Exception as e:
        print(f"✗ FNC heatmap plot failed: {e}")

    # Test brain report generation
    try:
        figure_paths = visualizer.generate_brain_figure_set(
            grad_cam_data=grad_cam_data,
            saliency_data=saliency_data,
            fnc_matrices={"group1": fnc_matrix},
            output_dir="test_outputs",
        )
        print("✓ Brain report generated successfully")
        print(f"  Generated files: {list(figure_paths.keys())}")
    except Exception as e:
        print(f"✗ Brain report generation failed: {e}")


def test_model_visualization():
    """Test model visualization functionality"""
    print("\nTesting model visualization...")

    # Create dummy model
    try:
        model = create_sspnet_model(input_shape=(48, 48, 48, 1))
        print("✓ Model created successfully")
    except Exception as e:
        print(f"✗ Model creation failed: {e}")
        return

    # Create dummy input data
    input_data = np.random.random((10, 48, 48, 48, 1))

    # Initialize visualizer
    visualizer = ModelVisualizer()

    # Test model architecture plot
    try:
        visualizer.plot_model_architecture(
            model, save_path="test_outputs/model_architecture.png"
        )
        print("✓ Model architecture plot created successfully")
    except Exception as e:
        print(f"✗ Model architecture plot failed: {e}")

    # Test saliency map generation
    try:
        saliency_paths = visualizer.generate_saliency_maps_batch(
            model, input_data[:3], output_dir="test_outputs"
        )
        print("✓ Saliency maps generated successfully")
        print(f"  Generated files: {list(saliency_paths.keys())}")
    except Exception as e:
        print(f"✗ Saliency map generation failed: {e}")

    # Test Grad-CAM generation
    try:
        gradcam_paths = visualizer.generate_grad_cam_batch(
            model, input_data[:3], output_dir="test_outputs"
        )
        print("✓ Grad-CAM maps generated successfully")
        print(f"  Generated files: {list(gradcam_paths.keys())}")
    except Exception as e:
        print(f"✗ Grad-CAM generation failed: {e}")

    # Test interpretability report
    try:
        report_paths = visualizer.generate_interpretability_report(
            model, input_data[:3], output_dir="test_outputs"
        )
        print("✓ Interpretability report generated successfully")
        print(f"  Generated files: {list(report_paths.keys())}")
    except Exception as e:
        print(f"✗ Interpretability report generation failed: {e}")


def test_interactive_plots():
    """Test interactive plotting functionality"""
    print("\nTesting interactive plots...")

    # Create dummy data
    y_true = np.random.randint(0, 2, 100)
    y_pred = np.random.randint(0, 2, 100)
    y_score = np.random.random(100)
    volume_data = np.random.random((32, 32, 32))
    fnc_matrix = np.random.random((15, 15)) * 2 - 1
    np.fill_diagonal(fnc_matrix, 1)

    # Initialize plotter
    plotter = InteractivePlotter()

    # Test interactive confusion matrix
    try:
        fig = plotter.plot_interactive_confusion_matrix(
            y_true, y_pred, save_path="test_outputs/interactive_confusion_matrix.html"
        )
        print("✓ Interactive confusion matrix created successfully")
    except Exception as e:
        print(f"✗ Interactive confusion matrix failed: {e}")

    # Test interactive ROC curve
    try:
        fig = plotter.plot_interactive_roc_curve(
            y_true, y_score, save_path="test_outputs/interactive_roc_curve.html"
        )
        print("✓ Interactive ROC curve created successfully")
    except Exception as e:
        print(f"✗ Interactive ROC curve failed: {e}")

    # Test interactive volume slicer
    try:
        fig = plotter.plot_interactive_volume_slicer(
            volume_data, save_path="test_outputs/interactive_volume_slicer.html"
        )
        print("✓ Interactive volume slicer created successfully")
    except Exception as e:
        print(f"✗ Interactive volume slicer failed: {e}")

    # Test interactive FNC matrix
    try:
        fig = plotter.plot_interactive_fnc_matrix(
            fnc_matrix, save_path="test_outputs/interactive_fnc_matrix.html"
        )
        print("✓ Interactive FNC matrix created successfully")
    except Exception as e:
        print(f"✗ Interactive FNC matrix failed: {e}")

    # Test interactive dashboard
    try:
        evaluator_results = {
            "y_true": y_true,
            "y_pred": y_pred,
            "y_pred_proba": y_score,
            "accuracy": 0.75,
            "sensitivity": 0.80,
            "specificity": 0.70,
            "auc": 0.85,
        }

        html_paths = plotter.generate_interactive_dashboard(
            evaluator_results=evaluator_results,
            volume_data=volume_data,
            fnc_matrices={"group1": fnc_matrix},
            output_dir="test_outputs",
        )
        print("✓ Interactive dashboard generated successfully")
        print(f"  Generated files: {list(html_paths.keys())}")
    except Exception as e:
        print(f"✗ Interactive dashboard generation failed: {e}")


def main():
    """Run all tests"""
    print("Running visualization tests...")

    # Create output directory
    os.makedirs("test_outputs", exist_ok=True)

    # Run tests
    test_result_plots()
    test_brain_visualization()
    test_model_visualization()
    test_interactive_plots()

    print("\nAll tests completed!")
    print("Check the 'test_outputs' directory for generated files.")


if __name__ == "__main__":
    main()
