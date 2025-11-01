"""
Utils module for common utilities.

This module provides common utilities for configuration, paths, and MLflow setup.
Following cookiecutter data science project structure.
"""

from .config import load_config
from .paths import get_project_root, get_data_path, get_models_path

# Lazy import for mlflow_setup to avoid errors if mlflow is not installed
try:
    from .mlflow_setup import setup_mlflow
    __all__ = [
        "load_config",
        "get_project_root",
        "get_data_path",
        "get_models_path",
        "setup_mlflow",
    ]
except ImportError:
    # mlflow not available (e.g., during development without virtualenv)
    __all__ = [
        "load_config",
        "get_project_root",
        "get_data_path",
        "get_models_path",
    ]

