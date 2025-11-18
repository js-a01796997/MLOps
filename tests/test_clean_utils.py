"""Unit tests for clean_utils module - utility functions for data cleaning"""

import pandas as pd
import numpy as np
import pytest
import sys
from pathlib import Path

pytestmark = pytest.mark.unit

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from clean_utils import clean_weather_var, to_int, fix_hour, detect_outliers


class TestCleanWeatherVar:
    """Tests for clean_weather_var function"""

    def test_out_of_range_values_set_to_nan(self):
        """Test that values > 1 are set to NaN"""
        data = {
            'temp': [0.5, 0.8, 1.5, 2.0, 0.3],  # 1.5 and 2.0 are out of range
            'other': [1, 2, 3, 4, 5]
        }
        df = pd.DataFrame(data)

        clean_weather_var(df, 'temp', 0)

        # Values > 1 should be NaN now
        assert pd.isna(df.iloc[2, 0])
        assert pd.isna(df.iloc[3, 0])
        # Values <= 1 should remain
        assert df.iloc[0, 0] == 0.5
        assert df.iloc[1, 0] == 0.8

    def test_nan_values_filled_by_backfill(self):
        """Test that NaN values are filled using backfill"""
        data = {
            'temp': [0.5, np.nan, np.nan, 0.8, 0.3],
        }
        df = pd.DataFrame(data)

        clean_weather_var(df, 'temp', 0)

        # NaN values should be backfilled with next valid value
        assert df['temp'].iloc[1] == 0.8
        assert df['temp'].iloc[2] == 0.8

    def test_valid_values_unchanged(self):
        """Test that valid values (0-1) remain unchanged"""
        data = {
            'temp': [0.0, 0.5, 1.0, 0.75],
        }
        df = pd.DataFrame(data)
        original = df.copy()

        clean_weather_var(df, 'temp', 0)

        pd.testing.assert_series_equal(df['temp'], original['temp'])

    def test_handles_all_nan(self):
        """Test behavior when all values are NaN"""
        data = {
            'temp': [np.nan, np.nan, np.nan],
        }
        df = pd.DataFrame(data)

        clean_weather_var(df, 'temp', 0)

        # All should still be NaN (backfill has nothing to fill with)
        assert df['temp'].isna().all()


class TestToInt:
    """Tests for to_int function"""

    def test_converts_float_to_int(self):
        """Test that float values are converted to Int64"""
        df = pd.DataFrame({'col': [1.0, 2.0, 3.0]})

        to_int(df, 'col')

        assert df['col'].dtype == 'Int64'

    def test_preserves_integer_values(self):
        """Test that integer values are preserved"""
        df = pd.DataFrame({'col': [1, 2, 3]})
        original_values = df['col'].values.copy()

        to_int(df, 'col')

        np.testing.assert_array_equal(df['col'].values, original_values)

    def test_handles_nan_values(self):
        """Test that NaN values are preserved in Int64 type"""
        df = pd.DataFrame({'col': [1.0, np.nan, 3.0]})

        to_int(df, 'col')

        assert df['col'].dtype == 'Int64'
        assert pd.isna(df['col'].iloc[1])
        assert df['col'].iloc[0] == 1
        assert df['col'].iloc[2] == 3


class TestFixHour:
    """Tests for fix_hour function"""

    def test_24_records_assigned_0_to_23(self):
        """Test that exactly 24 records get hours 0-23"""
        data = {
            'hr': [10, 5, 20, 3, 15, 8, 12, 18, 22, 1, 7, 14, 19, 11, 4, 16, 9, 21, 2, 13, 6, 17, 23, 0],
            'dteday': ['2011-01-01'] * 24
        }
        df = pd.DataFrame(data)

        result = fix_hour(df)

        # Should be assigned 0-23 in order
        expected = list(range(24))
        np.testing.assert_array_equal(result['hr'].values, expected)

    def test_25_records_last_set_to_nan(self):
        """Test that with 25 records, the last one is set to NaN"""
        data = {
            'hr': list(range(25)),
            'dteday': ['2011-01-01'] * 25
        }
        df = pd.DataFrame(data)

        result = fix_hour(df)

        # Last record should be NaN
        assert pd.isna(result['hr'].iloc[-1])
        # Others should be 0-24
        assert result['hr'].iloc[0] == 0
        assert result['hr'].iloc[23] == 23

    def test_less_than_24_records_unchanged(self):
        """Test that < 24 records are not modified"""
        data = {
            'hr': [5, 10, 15],
            'dteday': ['2011-01-01'] * 3
        }
        df = pd.DataFrame(data)
        original = df.copy()

        result = fix_hour(df)

        pd.testing.assert_series_equal(result['hr'], original['hr'])


class TestDetectOutliers:
    """Tests for detect_outliers function using IQR method"""

    def test_no_outliers_in_normal_data(self):
        """Test that normal data has no outliers"""
        np.random.seed(42)
        data = {
            'value': np.random.normal(100, 10, 100)  # Normal distribution
        }
        df = pd.DataFrame(data)

        outliers = detect_outliers(df, 'value', iqr_multiplier=1.5)

        # Should have very few or no outliers in normal data
        assert len(outliers) < 10  # Less than 10% outliers

    def test_detects_obvious_outliers(self):
        """Test that obvious outliers are detected"""
        data = {
            'value': [10, 12, 11, 13, 10, 12, 1000, 15, 14, 11]  # 1000 is obvious outlier
        }
        df = pd.DataFrame(data)

        outliers = detect_outliers(df, 'value', iqr_multiplier=1.5)

        # Should detect the outlier at index 6
        assert 6 in outliers

    def test_iqr_multiplier_affects_sensitivity(self):
        """Test that higher IQR multiplier detects fewer outliers"""
        data = {
            'value': [10, 12, 11, 13, 10, 12, 50, 15, 14, 11, 60]
        }
        df = pd.DataFrame(data)

        outliers_strict = detect_outliers(df, 'value', iqr_multiplier=1.5)
        outliers_lenient = detect_outliers(df, 'value', iqr_multiplier=3.0)

        # Stricter threshold should find more outliers
        assert len(outliers_strict) >= len(outliers_lenient)

    def test_returns_correct_indices(self):
        """Test that returned indices correspond to actual outlier rows"""
        data = {
            'value': [10, 10, 10, 100, 10, 10]  # Index 3 is outlier
        }
        df = pd.DataFrame(data)

        outliers = detect_outliers(df, 'value', iqr_multiplier=1.5)

        # Should return index 3
        assert 3 in outliers
        assert len(outliers) == 1

    def test_handles_uniform_data(self):
        """Test behavior with uniform data (no variance)"""
        data = {
            'value': [10, 10, 10, 10, 10]
        }
        df = pd.DataFrame(data)

        outliers = detect_outliers(df, 'value', iqr_multiplier=1.5)

        # No outliers in uniform data
        assert len(outliers) == 0

    def test_symmetric_outlier_detection(self):
        """Test that both high and low outliers are detected"""
        data = {
            'value': [5, 10, 12, 11, 13, 10, 12, 15, 14, 11, 50, -20]  # Both high (50) and low (-20) outliers
        }
        df = pd.DataFrame(data)

        outliers = detect_outliers(df, 'value', iqr_multiplier=1.5)

        # Should detect both outliers
        assert 10 in outliers  # High outlier at index 10 (value 50)
        assert 11 in outliers  # Low outlier at index 11 (value -20)


class TestCleanUtilsEdgeCases:
    """Edge case tests for clean_utils functions"""

    def test_clean_weather_var_empty_dataframe(self):
        """Test clean_weather_var with empty DataFrame"""
        df = pd.DataFrame({'temp': []})

        # Should not raise error
        clean_weather_var(df, 'temp', 0)
        assert len(df) == 0

    def test_to_int_with_large_numbers(self):
        """Test to_int with large integer values"""
        df = pd.DataFrame({'col': [1e6, 2e6, 3e6]})

        to_int(df, 'col')

        assert df['col'].dtype == 'Int64'
        assert df['col'].iloc[0] == 1000000

    def test_detect_outliers_single_value(self):
        """Test detect_outliers with single value"""
        df = pd.DataFrame({'value': [10]})

        outliers = detect_outliers(df, 'value', iqr_multiplier=1.5)

        # Single value cannot be an outlier
        assert len(outliers) == 0

    def test_fix_hour_empty_group(self):
        """Test fix_hour with empty group"""
        df = pd.DataFrame({'hr': [], 'dteday': []})

        result = fix_hour(df)

        assert len(result) == 0
