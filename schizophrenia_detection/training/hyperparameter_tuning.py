"""
Hyperparameter tuning utilities
"""

import itertools
import numpy as np
from ..models.sspnet_3d_cnn import create_sspnet_model
from ..training.trainer import ModelTrainer
from ..training.evaluator import ModelEvaluator
from ..config import default_config


class HyperparameterTuner:
    """
    Class for tuning hyperparameters of the model
    """

    def __init__(self, config=None):
        """
        Initialize the hyperparameter tuner

        Args:
            config: Configuration object
        """
        self.config = config if config is not None else default_config
        self.tuning_results = []

    def grid_search(self, param_grid, train_data, val_data):
        """
        Perform grid search over hyperparameters

        Args:
            param_grid (dict): Dictionary of hyperparameter grids
            train_data: Training data
            val_data: Validation data

        Returns:
            List of tuning results
        """
        # Generate all combinations of hyperparameters
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        param_combinations = list(itertools.product(*param_values))

        print(
            f"Starting grid search with {len(param_combinations)} parameter combinations"
        )

        for i, param_combination in enumerate(param_combinations):
            print(f"\nTesting combination {i+1}/{len(param_combinations)}")

            # Create parameter dictionary
            params = dict(zip(param_names, param_combination))
            print(f"Parameters: {params}")

            # Update config with new parameters
            self._update_config(params)

            # Train model
            trainer = ModelTrainer(self.config)
            model = trainer.build_model()
            history = trainer.train(train_data, val_data)

            # Evaluate model
            evaluator = ModelEvaluator(model, self.config)
            results = evaluator.evaluate(val_data)

            # Store results
            result = {
                "params": params,
                "val_accuracy": max(history.history["val_accuracy"]),
                "val_loss": min(history.history["val_loss"]),
                "results": results,
            }
            self.tuning_results.append(result)

            print(f"Validation accuracy: {result['val_accuracy']:.4f}")

        # Sort results by validation accuracy
        self.tuning_results.sort(key=lambda x: x["val_accuracy"], reverse=True)

        return self.tuning_results

    def random_search(self, param_distributions, n_iter, train_data, val_data):
        """
        Perform random search over hyperparameters

        Args:
            param_distributions (dict): Dictionary of hyperparameter distributions
            n_iter (int): Number of parameter settings sampled
            train_data: Training data
            val_data: Validation data

        Returns:
            List of tuning results
        """
        print(f"Starting random search with {n_iter} iterations")

        for i in range(n_iter):
            print(f"\nTesting iteration {i+1}/{n_iter}")

            # Sample random parameters
            params = {}
            for param_name, param_dist in param_distributions.items():
                if callable(param_dist):
                    params[param_name] = param_dist()
                elif isinstance(param_dist, list):
                    params[param_name] = np.random.choice(param_dist)
                elif isinstance(param_dist, tuple) and len(param_dist) == 2:
                    # Assume it's a range for continuous parameters
                    params[param_name] = np.random.uniform(param_dist[0], param_dist[1])
                else:
                    params[param_name] = param_dist

            print(f"Parameters: {params}")

            # Update config with new parameters
            self._update_config(params)

            # Train model
            trainer = ModelTrainer(self.config)
            model = trainer.build_model()
            history = trainer.train(train_data, val_data)

            # Evaluate model
            evaluator = ModelEvaluator(model, self.config)
            results = evaluator.evaluate(val_data)

            # Store results
            result = {
                "params": params,
                "val_accuracy": max(history.history["val_accuracy"]),
                "val_loss": min(history.history["val_loss"]),
                "results": results,
            }
            self.tuning_results.append(result)

            print(f"Validation accuracy: {result['val_accuracy']:.4f}")

        # Sort results by validation accuracy
        self.tuning_results.sort(key=lambda x: x["val_accuracy"], reverse=True)

        return self.tuning_results

    def _update_config(self, params):
        """
        Update configuration with new parameters

        Args:
            params (dict): New parameters
        """
        for param_name, param_value in params.items():
            if param_name == "learning_rate":
                self.config.training.learning_rate = param_value
            elif param_name == "batch_size":
                self.config.data.batch_size = param_value
            elif param_name == "dropout_rate":
                self.config.model.dropout_rate = param_value
            elif param_name == "num_spatial_filters":
                self.config.model.num_spatial_filters = param_value
            elif param_name == "num_spectral_filters":
                self.config.model.num_spectral_filters = param_value
            else:
                # Add more parameter mappings as needed
                pass

    def print_results(self, top_n=5):
        """
        Print the top n results from tuning

        Args:
            top_n (int): Number of top results to print
        """
        if not self.tuning_results:
            print("No tuning results available. Please run tuning first.")
            return

        print(f"\nTop {top_n} Hyperparameter Combinations:")
        print("=" * 50)

        for i, result in enumerate(self.tuning_results[:top_n]):
            print(f"Rank {i+1}:")
            print(f"  Validation Accuracy: {result['val_accuracy']:.4f}")
            print(f"  Validation Loss: {result['val_loss']:.4f}")
            print("  Parameters:")
            for param, value in result["params"].items():
                print(f"    {param}: {value}")
            print()

    def save_results(self, path):
        """
        Save tuning results to a file

        Args:
            path (str): Path to save the results
        """
        import json

        # Convert numpy arrays to lists for JSON serialization
        results_copy = []
        for result in self.tuning_results:
            result_copy = {}
            for key, value in result.items():
                if key == "results" and isinstance(value, dict):
                    result_copy[key] = {}
                    for sub_key, sub_value in value.items():
                        if isinstance(sub_value, (np.ndarray, np.float32, np.float64)):
                            result_copy[key][sub_key] = float(sub_value)
                        else:
                            result_copy[key][sub_key] = sub_value
                elif isinstance(value, (np.float32, np.float64)):
                    result_copy[key] = float(value)
                else:
                    result_copy[key] = value
            results_copy.append(result_copy)

        with open(path, "w") as f:
            json.dump(results_copy, f, indent=2)

        print(f"Tuning results saved to {path}")

    def get_best_params(self):
        """
        Get the best parameters from tuning

        Returns:
            Dictionary of best parameters
        """
        if not self.tuning_results:
            print("No tuning results available. Please run tuning first.")
            return None

        return self.tuning_results[0]["params"]


def create_param_grid():
    """
    Create a default parameter grid for grid search

    Returns:
        Dictionary of parameter grids
    """
    return {
        "learning_rate": [0.001, 0.0001, 0.00001],
        "batch_size": [2, 4, 8],
        "dropout_rate": [0.3, 0.5, 0.7],
        "num_spatial_filters": [16, 32, 64],
        "num_spectral_filters": [8, 16, 32],
    }


def create_param_distributions():
    """
    Create default parameter distributions for random search

    Returns:
        Dictionary of parameter distributions
    """
    return {
        "learning_rate": lambda: 10
        ** np.random.uniform(-5, -2),  # Log uniform between 1e-5 and 1e-2
        "batch_size": [2, 4, 8, 16],
        "dropout_rate": (0.2, 0.7),  # Uniform between 0.2 and 0.7
        "num_spatial_filters": [16, 32, 64, 128],
        "num_spectral_filters": [8, 16, 32, 64],
    }


def perform_grid_search(train_data, val_data, param_grid=None, config=None):
    """
    Convenience function to perform grid search

    Args:
        train_data: Training data
        val_data: Validation data
        param_grid (dict): Parameter grid for grid search
        config: Configuration object

    Returns:
        List of tuning results
    """
    if param_grid is None:
        param_grid = create_param_grid()

    tuner = HyperparameterTuner(config)
    results = tuner.grid_search(param_grid, train_data, val_data)
    return results


def perform_random_search(
    train_data, val_data, n_iter=20, param_distributions=None, config=None
):
    """
    Convenience function to perform random search

    Args:
        train_data: Training data
        val_data: Validation data
        n_iter (int): Number of parameter settings sampled
        param_distributions (dict): Parameter distributions for random search
        config: Configuration object

    Returns:
        List of tuning results
    """
    if param_distributions is None:
        param_distributions = create_param_distributions()

    tuner = HyperparameterTuner(config)
    results = tuner.random_search(param_distributions, n_iter, train_data, val_data)
    return results
