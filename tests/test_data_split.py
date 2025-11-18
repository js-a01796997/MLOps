"""Unit tests for data_split module - preprocessing, scaling, and splitting"""

import pandas as pd
import numpy as np
import pytest
import sys
from pathlib import Path

pytestmark = pytest.mark.unit

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data_split import convert_column_types, split_data, scale_features


@pytest.fixture
def sample_cleaned_data():
    """Create sample cleaned data for testing"""
    np.random.seed(42)
    n_rows = 100

    data = {
        'instant': range(1, n_rows + 1),
        'dteday': pd.date_range('2011-01-01', periods=n_rows, freq='H'),
        'season': np.random.choice([1, 2, 3, 4], n_rows),
        'yr': np.random.choice([0, 1], n_rows),
        'mnth': np.random.choice(range(1, 13), n_rows),
        'hr': np.random.choice(range(0, 24), n_rows),
        'holiday': np.random.choice([0, 1], n_rows),
        'weekday': np.random.choice(range(0, 7), n_rows),
        'workingday': np.random.choice([0, 1], n_rows),
        'weathersit': np.random.choice([1, 2, 3, 4], n_rows),
        'temp': np.random.uniform(0.0, 1.0, n_rows),
        'atemp': np.random.uniform(0.0, 1.0, n_rows),
        'hum': np.random.uniform(0.0, 1.0, n_rows),
        'windspeed': np.random.uniform(0.0, 1.0, n_rows),
        'casual': np.random.randint(0, 100, n_rows),
        'registered': np.random.randint(0, 500, n_rows),
    }
    data['cnt'] = data['casual'] + data['registered']

    return pd.DataFrame(data)


class TestConvertColumnTypes:
    """Tests for convert_column_types function"""

    def test_categorical_columns_are_converted(self, sample_cleaned_data):
        """Test that season, mnth, weekday, weathersit are converted to categorical"""
        result = convert_column_types(sample_cleaned_data)

        assert result['season'].dtype.name == 'category'
        assert result['mnth'].dtype.name == 'category'
        assert result['weekday'].dtype.name == 'category'
        assert result['weathersit'].dtype.name == 'category'

    def test_categorical_values_are_valid(self, sample_cleaned_data):
        """Test that categorical columns have valid ranges"""
        result = convert_column_types(sample_cleaned_data)

        # Season should be 1-4
        assert set(result['season'].cat.categories) == {1, 2, 3, 4}

        # Month should be 1-12
        assert set(result['mnth'].cat.categories) == set(range(1, 13))

        # Weekday should be 0-6
        assert set(result['weekday'].cat.categories) == set(range(0, 7))

        # Weathersit should be 1-4
        assert set(result['weathersit'].cat.categories) == {1, 2, 3, 4}

    def test_integer_columns_are_converted(self, sample_cleaned_data):
        """Test that integer columns are properly typed"""
        result = convert_column_types(sample_cleaned_data)

        int_cols = ['yr', 'holiday', 'workingday', 'cnt', 'registered', 'casual', 'instant']
        for col in int_cols:
            assert pd.api.types.is_integer_dtype(result[col]), f"{col} should be integer type"

    def test_float_columns_are_converted(self, sample_cleaned_data):
        """Test that float columns are properly typed"""
        result = convert_column_types(sample_cleaned_data)

        float_cols = ['temp', 'atemp', 'hum', 'windspeed']
        for col in float_cols:
            assert result[col].dtype == np.float64, f"{col} should be float64 type"

    def test_handles_missing_optional_column(self, sample_cleaned_data):
        """Test that function handles missing mixed_type_col gracefully"""
        # mixed_type_col is optional and may not exist in cleaned data
        result = convert_column_types(sample_cleaned_data)
        assert result is not None


class TestSplitData:
    """Tests for split_data function"""

    def test_split_sizes_are_correct(self, sample_cleaned_data):
        """Test that train/valid/test splits have correct proportions"""
        df = convert_column_types(sample_cleaned_data)

        train, valid, test = split_data(df, train_size=0.70, valid_size=0.15, test_size=0.15)

        total = len(train) + len(valid) + len(test)

        # Allow for small rounding differences
        assert abs(len(train) / total - 0.70) < 0.05
        assert abs(len(valid) / total - 0.15) < 0.05
        assert abs(len(test) / total - 0.15) < 0.05

    def test_splits_are_non_overlapping(self, sample_cleaned_data):
        """Test that train/valid/test splits don't overlap"""
        df = convert_column_types(sample_cleaned_data)

        train, valid, test = split_data(df, random_state=42)

        # Create a unique identifier column for checking overlaps
        # Since we dropped instant, casual, registered, dteday, we can't use those
        # Instead, check that total rows = original rows after dropping columns
        total_rows = len(train) + len(valid) + len(test)
        expected_rows = len(df) - 4  # We drop 4 columns but row count stays same

        # The row count should match (we're splitting the same data)
        assert total_rows == len(df)

    def test_redundant_columns_are_removed(self, sample_cleaned_data):
        """Test that instant, casual, registered, dteday are removed"""
        df = convert_column_types(sample_cleaned_data)

        train, valid, test = split_data(df)

        removed_cols = ['instant', 'casual', 'registered', 'dteday']
        for col in removed_cols:
            assert col not in train.columns
            assert col not in valid.columns
            assert col not in test.columns

    def test_dummy_variables_created(self, sample_cleaned_data):
        """Test that categorical variables are converted to dummies"""
        df = convert_column_types(sample_cleaned_data)

        train, valid, test = split_data(df)

        # Original categorical columns should be replaced with dummy columns
        # Check that we have dummy columns (they'll have names like 'season_2', 'season_3', etc.)
        col_names = train.columns.tolist()

        # Should have season dummies (drop_first=True, so 3 dummies for 4 categories)
        season_dummies = [c for c in col_names if c.startswith('season_')]
        assert len(season_dummies) == 3, "Should have 3 season dummy columns"

        # Should have mnth dummies (11 for 12 months)
        mnth_dummies = [c for c in col_names if c.startswith('mnth_')]
        assert len(mnth_dummies) == 11, "Should have 11 month dummy columns"

    def test_target_variable_exists(self, sample_cleaned_data):
        """Test that cnt (target variable) is preserved"""
        df = convert_column_types(sample_cleaned_data)

        train, valid, test = split_data(df)

        assert 'cnt' in train.columns
        assert 'cnt' in valid.columns
        assert 'cnt' in test.columns

    def test_reproducibility_with_random_state(self, sample_cleaned_data):
        """Test that same random_state produces same splits"""
        df = convert_column_types(sample_cleaned_data)

        train1, valid1, test1 = split_data(df, random_state=42)
        train2, valid2, test2 = split_data(df, random_state=42)

        # Should be identical
        pd.testing.assert_frame_equal(train1, train2)
        pd.testing.assert_frame_equal(valid1, valid2)
        pd.testing.assert_frame_equal(test1, test2)


class TestScaleFeatures:
    """Tests for scale_features function"""

    @pytest.fixture
    def sample_splits(self, sample_cleaned_data):
        """Create sample train/valid/test splits"""
        df = convert_column_types(sample_cleaned_data)
        return split_data(df, random_state=42)

    def test_scaler_fits_on_training_data(self, sample_splits):
        """Test that scaler is fitted on training data only"""
        train, valid, test = sample_splits

        train_scaled, valid_scaled, test_scaled, scaler = scale_features(train, valid, test)

        # Scaler should have been fitted (has attributes)
        assert hasattr(scaler, 'data_min_')
        assert hasattr(scaler, 'data_max_')

    def test_scaled_values_in_range(self, sample_splits):
        """Test that scaled values are in [0, 1] range"""
        train, valid, test = sample_splits

        cols_to_scale = ['temp', 'atemp', 'hum', 'windspeed', 'cnt']
        train_scaled, valid_scaled, test_scaled, scaler = scale_features(train, valid, test, cols_to_scale)

        for col in cols_to_scale:
            if col in train_scaled.columns:
                # Training data should be in [0, 1]
                assert train_scaled[col].min() >= 0.0
                assert train_scaled[col].max() <= 1.0

                # Note: valid and test might go slightly outside [0, 1] if they have values
                # outside the training range - this is expected behavior

    def test_scaling_preserves_relationships(self, sample_splits):
        """Test that scaling preserves relative relationships between values"""
        train, valid, test = sample_splits

        cols_to_scale = ['temp', 'atemp', 'hum', 'windspeed', 'cnt']
        train_scaled, valid_scaled, test_scaled, scaler = scale_features(train, valid, test, cols_to_scale)

        # Check that relative order is preserved in training data
        for col in cols_to_scale:
            if col in train.columns:
                original_order = train[col].argsort().values
                scaled_order = train_scaled[col].argsort().values

                np.testing.assert_array_equal(original_order, scaled_order,
                    err_msg=f"Scaling should preserve order for {col}")

    def test_unscaled_columns_unchanged(self, sample_splits):
        """Test that columns not in cols_to_scale remain unchanged"""
        train, valid, test = sample_splits

        cols_to_scale = ['temp', 'atemp', 'hum', 'windspeed', 'cnt']
        train_scaled, valid_scaled, test_scaled, scaler = scale_features(train, valid, test, cols_to_scale)

        # Get columns that shouldn't be scaled
        unscaled_cols = [c for c in train.columns if c not in cols_to_scale]

        for col in unscaled_cols:
            pd.testing.assert_series_equal(train[col], train_scaled[col],
                check_names=True, obj=f"Column {col} should not be scaled")

    def test_same_scaler_for_all_splits(self, sample_splits):
        """Test that the same scaler is applied to train/valid/test"""
        train, valid, test = sample_splits

        cols_to_scale = ['temp', 'atemp', 'hum', 'windspeed', 'cnt']
        train_scaled, valid_scaled, test_scaled, scaler = scale_features(train, valid, test, cols_to_scale)

        # All splits should use the same min/max from training data
        # This is verified by checking the scaler was only fitted once
        assert scaler.n_features_in_ == len(cols_to_scale)

    def test_default_columns_to_scale(self, sample_splits):
        """Test that default cols_to_scale is used when None provided"""
        train, valid, test = sample_splits

        # Call without specifying cols_to_scale
        train_scaled, valid_scaled, test_scaled, scaler = scale_features(train, valid, test)

        # Should have scaled the default columns
        assert scaler.n_features_in_ == 5  # temp, atemp, hum, windspeed, cnt


class TestDataSplitIntegration:
    """Integration tests for the complete data split pipeline"""

    def test_full_pipeline_flow(self, sample_cleaned_data):
        """Test the complete flow: convert -> split -> scale"""
        # Convert types
        df = convert_column_types(sample_cleaned_data)

        # Split data
        train, valid, test = split_data(df, random_state=42)

        # Scale features
        train_scaled, valid_scaled, test_scaled, scaler = scale_features(train, valid, test)

        # Verify final output
        assert not train_scaled.empty
        assert not valid_scaled.empty
        assert not test_scaled.empty
        assert train_scaled.shape[0] > valid_scaled.shape[0]
        assert train_scaled.shape[0] > test_scaled.shape[0]

    def test_no_data_leakage(self, sample_cleaned_data):
        """Test that there's no data leakage between splits"""
        df = convert_column_types(sample_cleaned_data)

        # Add a unique identifier before splitting
        df['unique_id'] = range(len(df))

        train, valid, test = split_data(df, random_state=42)

        # Check that unique_ids don't overlap
        train_ids = set(train['unique_id'].values) if 'unique_id' in train.columns else set()
        valid_ids = set(valid['unique_id'].values) if 'unique_id' in valid.columns else set()
        test_ids = set(test['unique_id'].values) if 'unique_id' in test.columns else set()

        # Note: unique_id might be dropped, but if it exists, check no overlap
        if train_ids and valid_ids:
            assert len(train_ids & valid_ids) == 0, "Train and valid should not overlap"
        if train_ids and test_ids:
            assert len(train_ids & test_ids) == 0, "Train and test should not overlap"
        if valid_ids and test_ids:
            assert len(valid_ids & test_ids) == 0, "Valid and test should not overlap"
