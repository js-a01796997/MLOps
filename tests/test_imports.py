"""Simple smoke tests - just check if code imports and runs without crashing"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def test_data_cleaning_imports():
    """Test that data_cleaning module imports without errors"""
    try:
        from data_cleaning import load_and_convert_types, clean_count_variables
        assert True
    except Exception as e:
        assert False, f"Failed to import data_cleaning: {e}"


def test_data_split_imports():
    """Test that data_split module imports without errors"""
    try:
        from data_split import convert_column_types, split_data, scale_features
        assert True
    except Exception as e:
        assert False, f"Failed to import data_split: {e}"


def test_train_models_imports():
    """Test that train_models module imports without errors"""
    try:
        import train_models
        assert True
    except Exception as e:
        assert False, f"Failed to import train_models: {e}"


def test_utils_config_imports():
    """Test that utils.config module imports without errors"""
    try:
        from utils.config import load_config
        assert True
    except Exception as e:
        assert False, f"Failed to import utils.config: {e}"


def test_utils_paths_imports():
    """Test that utils.paths module imports without errors"""
    try:
        from utils.paths import get_data_path, get_models_path
        assert True
    except Exception as e:
        assert False, f"Failed to import utils.paths: {e}"
