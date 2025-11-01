"""Simple tests for data cleaning - focusing on missing values"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data_cleaning import load_and_convert_types, clean_count_variables


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


def test_casual_plus_registered_equals_cnt():
    """Test that casual + registered = cnt relationship holds"""
    # Create test data where the relationship is valid
    data = {
        'instant': [1, 2, 3, 4, 5],
        'dteday': ['2011-01-01', '2011-01-02', '2011-01-03', '2011-01-04', '2011-01-05'],
        'season': [1, 1, 1, 1, 1],
        'yr': [0, 0, 0, 0, 0],
        'mnth': [1, 1, 1, 1, 1],
        'hr': [0, 1, 2, 3, 4],
        'holiday': [0, 0, 0, 0, 0],
        'weekday': [0, 1, 2, 3, 4],
        'workingday': [1, 1, 1, 1, 1],
        'weathersit': [1, 1, 1, 1, 1],
        'temp': [0.24, 0.22, 0.23, 0.25, 0.26],
        'atemp': [0.28, 0.27, 0.28, 0.29, 0.30],
        'hum': [0.81, 0.80, 0.79, 0.78, 0.77],
        'windspeed': [0.0, 0.0, 0.0, 0.0, 0.0],
        'casual': [5, 10, 8, 12, 15],
        'registered': [20, 30, 25, 40, 35],
        'cnt': [25, 40, 33, 52, 50],
        'mixed_type_col': [1, 2, 3, 4, 5]
    }
    df = pd.DataFrame(data)

    # Apply the cleaning function
    result = clean_count_variables(df)

    # Check that the relationship holds for all rows
    assert (result['casual'] + result['registered'] == result['cnt']).all(), \
        "casual + registered should equal cnt for all rows"

    # Check that no negative values exist
    assert (result['casual'] >= 0).all(), "casual should have no negative values"
    assert (result['registered'] >= 0).all(), "registered should have no negative values"
    assert (result['cnt'] >= 0).all(), "cnt should have no negative values"

    # Check that all are integers
    assert pd.api.types.is_integer_dtype(result['casual']), "casual should be integer type"
    assert pd.api.types.is_integer_dtype(result['registered']), "registered should be integer type"
    assert pd.api.types.is_integer_dtype(result['cnt']), "cnt should be integer type"


def test_inconsistent_counts_are_fixed():
    """Test that inconsistent count relationships are corrected"""
    # Create test data where casual + registered != cnt (inconsistent)
    data = {
        'instant': [1, 2, 3],
        'dteday': ['2011-01-01', '2011-01-02', '2011-01-03'],
        'season': [1, 1, 1],
        'yr': [0, 0, 0],
        'mnth': [1, 1, 1],
        'hr': [0, 1, 2],
        'holiday': [0, 0, 0],
        'weekday': [0, 1, 2],
        'workingday': [1, 1, 1],
        'weathersit': [1, 1, 1],
        'temp': [0.24, 0.22, 0.23],
        'atemp': [0.28, 0.27, 0.28],
        'hum': [0.81, 0.80, 0.79],
        'windspeed': [0.0, 0.0, 0.0],
        'casual': [5, 10, 8],
        'registered': [20, 30, 25],
        'cnt': [30, 50, 40],  # Inconsistent: should be [25, 40, 33]
        'mixed_type_col': [1, 2, 3]
    }
    df = pd.DataFrame(data)

    # Apply the cleaning function
    result = clean_count_variables(df)

    # After cleaning, the relationship should be consistent
    assert (result['casual'] + result['registered'] == result['cnt']).all(), \
        "Inconsistent counts should be fixed so casual + registered = cnt"


def test_negative_values_are_converted_to_absolute():
    """Test that negative count values are converted to absolute values"""
    # Create test data with negative values
    data = {
        'instant': [1, 2, 3, 4],
        'dteday': ['2011-01-01', '2011-01-02', '2011-01-03', '2011-01-04'],
        'season': [1, 1, 1, 1],
        'yr': [0, 0, 0, 0],
        'mnth': [1, 1, 1, 1],
        'hr': [0, 1, 2, 3],
        'holiday': [0, 0, 0, 0],
        'weekday': [0, 1, 2, 3],
        'workingday': [1, 1, 1, 1],
        'weathersit': [1, 1, 1, 1],
        'temp': [0.24, 0.22, 0.23, 0.25],
        'atemp': [0.28, 0.27, 0.28, 0.29],
        'hum': [0.81, 0.80, 0.79, 0.78],
        'windspeed': [0.0, 0.0, 0.0, 0.0],
        'casual': [5, -10, 8, 12],  # Negative value in row 1
        'registered': [20, 30, 25, 40],
        'cnt': [25, 40, 33, 52],
        'mixed_type_col': [1, 2, 3, 4]
    }
    df = pd.DataFrame(data)

    # Apply the cleaning function
    result = clean_count_variables(df)

    # Check that all values are non-negative (converted to absolute values)
    assert (result['casual'] >= 0).all(), "All casual values should be non-negative"
    assert (result['registered'] >= 0).all(), "All registered values should be non-negative"
    assert (result['cnt'] >= 0).all(), "All cnt values should be non-negative"

    # Check that the negative value -10 was converted to 10
    assert result.loc[1, 'casual'] == 10, "Negative value should be converted to absolute value"
