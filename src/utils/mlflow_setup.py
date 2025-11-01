"""
MLflow setup utilities.

Centralized MLflow configuration and setup.
"""

import os
from typing import Dict, Any, Optional
import mlflow
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
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    elif os.getenv("MLFLOW_TRACKING_URI"):
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
    elif "mlflow" in config and "tracking_uri" in config["mlflow"]:
        mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])

    # Set experiment name from config
    if "mlflow" in config and "experiment_name" in config["mlflow"]:
        mlflow.set_experiment(config["mlflow"]["experiment_name"])

