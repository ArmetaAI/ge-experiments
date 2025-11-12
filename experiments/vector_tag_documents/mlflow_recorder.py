"""
MLflow Recorder for logging experiment metrics.

This module provides a class to connect to an MLflow instance and log metrics
under dynamically named experiments.
"""

import mlflow
from datetime import datetime
from typing import Dict, Any, Optional
from app.shared.config.settings import get_settings


class MLflowRecorder:
    """
    A recorder class for logging metrics to MLflow.

    Features:
    - Connects to MLflow instance using URI from .env
    - Creates dynamic experiment names with timestamp and custom name
    - Logs metrics dictionaries to the experiment

    Example:
        ```python
        recorder = MLflowRecorder(experiment_name="vector_search_test")

        metrics = {
            "accuracy": 0.95,
            "precision": 0.92,
            "recall": 0.88
        }

        recorder.log_metrics(metrics)
        recorder.end_run()
        ```
    """

    def __init__(self, experiment_name: str):
        """
        Initialize the MLflow recorder.

        Args:
            experiment_name: Custom name for the experiment. Will be combined with
                           timestamp to create a unique experiment name.

        Example:
            If experiment_name is "vector_search" and the current timestamp is
            "2025-11-11_14-30-00", the experiment will be named:
            "vector_experiments_vector_search_2025-11-11_14-30-00"
        """
        self.settings = get_settings()

        # Set MLflow tracking URI from settings
        mlflow.set_tracking_uri(self.settings.MLFLOW_TRACKING_URI)

        # Create dynamic experiment name with timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        prefix = self.settings.MLFLOW_EXPERIMENT_NAME_PREFIX
        self.experiment_name = f"{prefix}_{experiment_name}_{timestamp}"

        # Set or create the experiment
        mlflow.set_experiment(self.experiment_name)

        # Start a new run
        self.run = mlflow.start_run()

        print(f"MLflow Recorder initialized")
        print(f"  Tracking URI: {self.settings.MLFLOW_TRACKING_URI}")
        print(f"  Experiment: {self.experiment_name}")
        print(f"  Run ID: {self.run.info.run_id}")

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """
        Log a dictionary of metrics to the current MLflow run.

        Args:
            metrics: Dictionary of metric names to values. Values should be numeric.
            step: Optional step number for the metrics (useful for tracking progress
                 over iterations/epochs)

        Example:
            ```python
            recorder.log_metrics({
                "train_loss": 0.32,
                "val_loss": 0.45,
                "learning_rate": 0.001
            }, step=10)
            ```
        """
        if not metrics:
            print("Warning: Empty metrics dictionary provided")
            return

        try:
            mlflow.log_metrics(metrics, step=step)
            print(f"Logged {len(metrics)} metrics to MLflow")
            if step is not None:
                print(f"  Step: {step}")
            for key, value in metrics.items():
                print(f"  {key}: {value}")
        except Exception as e:
            print(f"Error logging metrics to MLflow: {e}")
            raise

    def log_param(self, key: str, value: Any):
        """
        Log a single parameter to the current MLflow run.

        Args:
            key: Parameter name
            value: Parameter value

        Example:
            ```python
            recorder.log_param("model_type", "sentence-transformers")
            recorder.log_param("embedding_dim", 384)
            ```
        """
        try:
            mlflow.log_param(key, value)
            print(f"Logged parameter: {key} = {value}")
        except Exception as e:
            print(f"Error logging parameter to MLflow: {e}")
            raise

    def log_params(self, params: Dict[str, Any]):
        """
        Log a dictionary of parameters to the current MLflow run.

        Args:
            params: Dictionary of parameter names to values

        Example:
            ```python
            recorder.log_params({
                "model_type": "sentence-transformers",
                "embedding_dim": 384,
                "batch_size": 32
            })
            ```
        """
        if not params:
            print("Warning: Empty params dictionary provided")
            return

        try:
            mlflow.log_params(params)
            print(f"Logged {len(params)} parameters to MLflow")
            for key, value in params.items():
                print(f"  {key}: {value}")
        except Exception as e:
            print(f"Error logging parameters to MLflow: {e}")
            raise

    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None):
        """
        Log a file or directory as an artifact.

        Args:
            local_path: Path to the file or directory to log
            artifact_path: Optional subdirectory in the artifact storage

        Example:
            ```python
            recorder.log_artifact("results/confusion_matrix.png", "plots")
            ```
        """
        try:
            mlflow.log_artifact(local_path, artifact_path)
            print(f"Logged artifact: {local_path}")
        except Exception as e:
            print(f"Error logging artifact to MLflow: {e}")
            raise

    def set_tag(self, key: str, value: str):
        """
        Set a tag for the current run.

        Args:
            key: Tag name
            value: Tag value

        Example:
            ```python
            recorder.set_tag("dataset", "production")
            recorder.set_tag("model_version", "v1.2.3")
            ```
        """
        try:
            mlflow.set_tag(key, value)
            print(f"Set tag: {key} = {value}")
        except Exception as e:
            print(f"Error setting tag in MLflow: {e}")
            raise

    def end_run(self):
        """
        End the current MLflow run.

        Should be called when you're done logging metrics for this experiment.

        Example:
            ```python
            recorder = MLflowRecorder("my_experiment")
            recorder.log_metrics({"accuracy": 0.95})
            recorder.end_run()  # Always end the run when done
            ```
        """
        try:
            mlflow.end_run()
            print(f"Ended MLflow run: {self.experiment_name}")
        except Exception as e:
            print(f"Error ending MLflow run: {e}")
            raise

    def __enter__(self):
        """Support for context manager (with statement)."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Automatically end run when exiting context manager."""
        self.end_run()
        return False  # Don't suppress exceptions


# Example usage
if __name__ == "__main__":
    # Example 1: Basic usage with manual run management
    print("=== Example 1: Manual run management ===")
    recorder = MLflowRecorder(experiment_name="demo_experiment")

    # Log some parameters
    recorder.log_params({
        "model_type": "sentence-transformers",
        "embedding_model": "paraphrase-multilingual-MiniLM-L12-v2",
        "batch_size": 32
    })

    # Log metrics
    metrics = {
        "accuracy": 0.95,
        "precision": 0.92,
        "recall": 0.88,
        "f1_score": 0.90
    }
    recorder.log_metrics(metrics)

    # Log metrics for multiple steps
    for step in range(5):
        step_metrics = {
            "train_loss": 0.5 - (step * 0.08),
            "val_loss": 0.6 - (step * 0.06)
        }
        recorder.log_metrics(step_metrics, step=step)

    # Set tags
    recorder.set_tag("dataset", "test_dataset")
    recorder.set_tag("environment", "development")

    recorder.end_run()

    # Example 2: Using context manager (recommended)
    print("\n=== Example 2: Context manager usage (recommended) ===")
    with MLflowRecorder(experiment_name="demo_context_manager") as recorder:
        recorder.log_params({"learning_rate": 0.001})
        recorder.log_metrics({"accuracy": 0.97})
        # Run automatically ends when exiting the with block
