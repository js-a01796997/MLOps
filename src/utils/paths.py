"""
Path management utilities.

Centralized path handling for project directories.
Following cookiecutter data science project structure.
"""

from pathlib import Path
from typing import Union


def get_project_root() -> Path:
    """
    Get the project root directory (MLOps).

    This function assumes it's called from a file within src/,
    so it goes up two levels: src -> project_root

    Returns:
        Path object pointing to the project root directory
    """
    # Assuming this file is in src/utils/, so we go up 2 levels
    return Path(__file__).parent.parent.parent


def get_data_path(subpath: str = "") -> Path:
    """
    Get path to data directory.

    Args:
        subpath: Optional subpath within data directory (e.g., "raw", "processed")

    Returns:
        Path object pointing to the data directory or subdirectory
    """
    project_root = get_project_root()
    if subpath:
        return project_root / "data" / subpath
    return project_root / "data"


def get_models_path() -> Path:
    """
    Get path to models directory.

    Returns:
        Path object pointing to the models directory
    """
    project_root = get_project_root()
    return project_root / "models"


def get_config_path(config_file: str = "models_config.yaml") -> Path:
    """
    Get path to config directory.

    Args:
        config_file: Optional config filename

    Returns:
        Path object pointing to the config file or directory
    """
    project_root = get_project_root()
    if config_file:
        return project_root / "config" / config_file
    return project_root / "config"

