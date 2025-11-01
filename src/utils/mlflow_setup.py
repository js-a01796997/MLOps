"""
MLflow setup utilities.

Centralized MLflow configuration and setup.
"""

import os
from typing import Dict, Any, Optional
import mlflow
from mlflow.exceptions import MlflowException
from dotenv import load_dotenv


def setup_mlflow(config: Dict[str, Any], tracking_uri: Optional[str] = None) -> None:
    """
    Setup MLflow tracking with configuration.

    Args:
        config: Configuration dictionary containing MLflow settings.
                Expected keys: 'mlflow.tracking_uri', 'mlflow.experiment_name'
        tracking_uri: Optional explicit tracking URI. If provided, overrides config.
                      If not provided, checks environment variable MLFLOW_TRACKING_URI.
                      If still not found, uses config value.

    Example:
        >>> config = {
        ...     'mlflow': {
        ...         'tracking_uri': 'file:./mlruns',
        ...         'experiment_name': 'my_experiment'
        ...     }
        ... }
        >>> setup_mlflow(config)
    """
    # Load environment variables from .env if it exists
    load_dotenv()

    # Determine tracking URI priority:
    # 1. Explicit parameter
    # 2. Environment variable
    # 3. Config file
    final_tracking_uri = None
    if tracking_uri:
        final_tracking_uri = tracking_uri
    elif os.getenv("MLFLOW_TRACKING_URI"):
        final_tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    elif "mlflow" in config and "tracking_uri" in config["mlflow"]:
        final_tracking_uri = config["mlflow"]["tracking_uri"]
    
    if final_tracking_uri:
        mlflow.set_tracking_uri(final_tracking_uri)
        print(f"✓ MLflow Tracking URI set to: {final_tracking_uri}")
    else:
        print("⚠ Warning: No tracking URI specified, using default")

    # Set experiment name from config
    if "mlflow" in config and "experiment_name" in config["mlflow"]:
        experiment_name = config["mlflow"]["experiment_name"]
        try:
            # This will create the experiment if it doesn't exist
            experiment = mlflow.set_experiment(experiment_name)
            print(f"✓ MLflow Experiment: {experiment_name} (ID: {experiment.experiment_id})")
        except MlflowException as e:
            print(f"⚠ Warning: Could not set MLflow experiment '{experiment_name}': {e}")
            print("  Attempting to create experiment...")
            try:
                experiment_id = mlflow.create_experiment(experiment_name)
                mlflow.set_experiment(experiment_name)
                print(f"✓ Created new experiment: {experiment_name} (ID: {experiment_id})")
            except Exception as create_error:
                print(f"✗ Error creating experiment: {create_error}")
                raise
    else:
        print("⚠ Warning: No experiment name specified in config")

