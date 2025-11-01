"""Simple tests for data cleaning - focusing on missing values"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data_cleaning import load_and_convert_types


def test_missing_values_are_handled():
    """Test that missing values are properly converted to NaN"""
    # Create test data with missing values
    data = {
        'instant': [1, 2, 3],
        'dteday': ['2011-01-01', '2011-01-02', '2011-01-03'],
        'season': [1, 2, 3],
        'yr': [0, 0, 0],
        'mnth': [1, 1, 1],
        'hr': [0, 1, 2],
        'holiday': [0, 0, 0],
        'weekday': [0, 1, 2],
        'workingday': [1, 1, 1],
        'weathersit': [1, 2, 1],
        'temp': [0.24, None, 0.22],  # Missing value
        'atemp': [0.28, 0.27, 0.26],
        'hum': [0.81, 0.80, 0.79],
        'windspeed': [0.0, 0.0, 0.0],
        'casual': [3, 8, 5],
        'registered': [13, 32, 27],
        'cnt': [16, 40, 32],
        'mixed_type_col': [1, 2, 3]
    }
    df = pd.DataFrame(data)

    # Apply the function
    result = load_and_convert_types(df)

    # Check that missing values are converted to NaN
    assert result['temp'].isna().sum() == 1
    assert pd.isna(result.loc[1, 'temp'])


def test_non_numeric_values_converted_to_nan():
    """Test that non-numeric values are converted to NaN"""
    # Create test data with non-numeric values
    data = {
        'instant': [1, 2, 3],
        'dteday': ['2011-01-01', '2011-01-02', '2011-01-03'],
        'season': [1, 'invalid', 3],  # Non-numeric value
        'yr': [0, 0, 0],
        'mnth': [1, 1, 1],
        'hr': [0, 1, 2],
        'holiday': [0, 0, 0],
        'weekday': [0, 1, 2],
        'workingday': [1, 1, 1],
        'weathersit': [1, 2, 1],
        'temp': [0.24, 0.25, 0.22],
        'atemp': [0.28, 0.27, 0.26],
        'hum': [0.81, 0.80, 0.79],
        'windspeed': [0.0, 0.0, 0.0],
        'casual': [3, 8, 5],
        'registered': [13, 32, 27],
        'cnt': [16, 40, 32],
        'mixed_type_col': [1, 2, 3]
    }
    df = pd.DataFrame(data)

    # Apply the function
    result = load_and_convert_types(df)

    # Check that non-numeric value was converted to NaN
    assert result['season'].isna().sum() == 1
    assert pd.isna(result.loc[1, 'season'])


def test_data_types_after_conversion():
    """Test that columns have correct data types after conversion"""
    # Create simple valid test data
    data = {
        'instant': [1, 2, 3],
        'dteday': ['2011-01-01', '2011-01-02', '2011-01-03'],
        'season': [1, 2, 3],
        'yr': [0, 0, 0],
        'mnth': [1, 1, 1],
        'hr': [0, 1, 2],
        'holiday': [0, 0, 0],
        'weekday': [0, 1, 2],
        'workingday': [1, 1, 1],
        'weathersit': [1, 2, 1],
        'temp': [0.24, 0.25, 0.22],
        'atemp': [0.28, 0.27, 0.26],
        'hum': [0.81, 0.80, 0.79],
        'windspeed': [0.0, 0.0, 0.0],
        'casual': [3, 8, 5],
        'registered': [13, 32, 27],
        'cnt': [16, 40, 32],
        'mixed_type_col': [1, 2, 3]
    }
    df = pd.DataFrame(data)

    # Apply the function
    result = load_and_convert_types(df)

    # Check data types
    assert result['temp'].dtype == float
    assert result['atemp'].dtype == float
    assert result['hum'].dtype == float
    assert result['windspeed'].dtype == float
    assert pd.api.types.is_integer_dtype(result['season'])
    assert pd.api.types.is_integer_dtype(result['yr'])
