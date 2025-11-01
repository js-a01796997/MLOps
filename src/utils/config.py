"""
Configuration utilities.

Centralized configuration loading from YAML files.
"""

import yaml
from pathlib import Path
from typing import Dict, Any


def load_config(config_path: str = "config/models_config.yaml") -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to the configuration YAML file. Can be relative or absolute.
                    Defaults to "config/models_config.yaml"

    Returns:
        Dictionary containing the configuration

    Raises:
        FileNotFoundError: If the config file doesn't exist
        yaml.YAMLError: If the YAML file is malformed
    """
    config_file = Path(config_path)
    
    # If relative path, resolve from project root
    if not config_file.is_absolute():
        project_root = Path(__file__).parent.parent.parent
        config_file = project_root / config_path
    
    if not config_file.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {config_file}\n"
            f"Expected location: {config_file.absolute()}"
        )
    
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise yaml.YAMLError(
            f"Error parsing YAML file {config_file}: {str(e)}"
        ) from e
    except Exception as e:
        raise IOError(
            f"Error reading configuration file {config_file}: {str(e)}"
        ) from e
    
    if config is None:
        raise ValueError(f"Configuration file {config_file} is empty or contains no valid YAML")
    
    return config

