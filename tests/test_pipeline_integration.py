"""Integration tests for end-to-end pipeline flow

These tests validate the complete pipeline from raw data through preprocessing,
splitting, and preparation for model training.
"""

import pandas as pd
import numpy as np
import pytest
import sys
from pathlib import Path
import tempfile
import shutil

pytestmark = pytest.mark.integration

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data_cleaning import load_and_convert_types, clean_count_variables, main as clean_main
from data_split import convert_column_types, split_data, scale_features, main as split_main


@pytest.fixture
def temp_dirs():
    """Create temporary directories for testing"""
    temp_root = tempfile.mkdtemp()
    raw_dir = Path(temp_root) / "raw"
    processed_dir = Path(temp_root) / "processed"
    raw_dir.mkdir(parents=True)
    processed_dir.mkdir(parents=True)

    yield raw_dir, processed_dir

    # Cleanup
    shutil.rmtree(temp_root)


@pytest.fixture
def sample_raw_data():
    """Create sample raw data that mimics the actual dataset"""
    np.random.seed(42)
    n_rows = 200

    dates = pd.date_range('2011-01-01', periods=n_rows, freq='H')
    data = {
        'instant': range(1, n_rows + 1),
        'dteday': dates.strftime('%Y-%m-%d').tolist(),
        'season': np.random.choice([1, 2, 3, 4], n_rows),
        'yr': [0 if d.year == 2011 else 1 for d in dates],
        'mnth': [d.month for d in dates],
        'hr': [d.hour for d in dates],
        'holiday': np.random.choice([0, 1], n_rows, p=[0.95, 0.05]),
        'weekday': [d.weekday() for d in dates],
        'workingday': np.random.choice([0, 1], n_rows, p=[0.3, 0.7]),
        'weathersit': np.random.choice([1, 2, 3, 4], n_rows, p=[0.5, 0.3, 0.15, 0.05]),
        'temp': np.random.uniform(0.0, 1.0, n_rows),
        'atemp': np.random.uniform(0.0, 1.0, n_rows),
        'hum': np.random.uniform(0.0, 1.0, n_rows),
        'windspeed': np.random.uniform(0.0, 0.5, n_rows),
        'casual': np.random.randint(0, 100, n_rows),
        'registered': np.random.randint(0, 500, n_rows),
        'mixed_type_col': np.random.choice([1, 2, 3], n_rows)
    }
    data['cnt'] = data['casual'] + data['registered']

    return pd.DataFrame(data)


class TestDataCleaningPipeline:
    """Integration tests for the data cleaning pipeline"""

    def test_load_and_convert_types_pipeline(self, sample_raw_data):
        """Test that load_and_convert_types correctly processes data"""
        result = load_and_convert_types(sample_raw_data)

        # Check data types
        assert result['dteday'].dtype == 'datetime64[ns]'
        assert pd.api.types.is_integer_dtype(result['season'])
        assert pd.api.types.is_integer_dtype(result['yr'])
        assert result['temp'].dtype == float
        assert result['windspeed'].dtype == float

        # Check no unexpected nulls in key columns
        assert result['season'].notna().sum() > 0
        assert result['yr'].notna().sum() > 0

    def test_clean_count_variables_maintains_relationship(self, sample_raw_data):
        """Test that clean_count_variables maintains casual + registered = cnt"""
        df = load_and_convert_types(sample_raw_data)
        result = clean_count_variables(df)

        # The relationship should hold for all rows
        assert (result['casual'] + result['registered'] == result['cnt']).all()

        # No negative values
        assert (result['casual'] >= 0).all()
        assert (result['registered'] >= 0).all()
        assert (result['cnt'] >= 0).all()

    def test_cleaning_pipeline_removes_invalid_rows(self, sample_raw_data):
        """Test that cleaning pipeline removes rows with issues"""
        # Add some problematic data
        sample_raw_data.loc[0, 'temp'] = 5.0  # Out of range
        sample_raw_data.loc[1, 'casual'] = np.nan

        df = load_and_convert_types(sample_raw_data)
        df = clean_count_variables(df)

        # Should have fewer rows after cleaning
        assert len(df) < len(sample_raw_data)

        # All remaining data should be valid
        assert df['temp'].max() <= 1.0
        assert df['casual'].notna().all()


class TestDataSplitPipeline:
    """Integration tests for the data splitting pipeline"""

    def test_split_pipeline_produces_three_sets(self, sample_raw_data):
        """Test that splitting produces train, valid, and test sets"""
        # Clean the data first
        df = load_and_convert_types(sample_raw_data)
        df = clean_count_variables(df)

        # Convert types for splitting
        df = convert_column_types(df)

        # Split
        train, valid, test = split_data(df, random_state=42)

        # Check all three sets exist and are non-empty
        assert len(train) > 0
        assert len(valid) > 0
        assert len(test) > 0

        # Check proportions are reasonable
        total = len(train) + len(valid) + len(test)
        assert len(train) / total > 0.6  # At least 60% in training
        assert len(test) / total > 0.1   # At least 10% in test

    def test_split_and_scale_pipeline(self, sample_raw_data):
        """Test complete split and scale pipeline"""
        # Clean the data first
        df = load_and_convert_types(sample_raw_data)
        df = clean_count_variables(df)

        # Convert and split
        df = convert_column_types(df)
        train, valid, test = split_data(df, random_state=42)

        # Scale
        train_s, valid_s, test_s, scaler = scale_features(train, valid, test)

        # Check scaling worked
        assert 'cnt' in train_s.columns
        assert train_s['cnt'].min() >= 0
        assert train_s['cnt'].max() <= 1

        # Check scaler has correct attributes
        assert hasattr(scaler, 'data_min_')
        assert hasattr(scaler, 'data_max_')


class TestEndToEndPipeline:
    """End-to-end integration tests simulating complete pipeline execution"""

    def test_complete_pipeline_from_raw_to_splits(self, sample_raw_data, temp_dirs):
        """Test the complete pipeline from raw data to train/valid/test splits"""
        raw_dir, processed_dir = temp_dirs

        # Save raw data
        raw_file = raw_dir / "test_data.csv"
        sample_raw_data.to_csv(raw_file, index=False)

        # Step 1: Clean data
        cleaned_file = processed_dir / "cleaned.csv"
        cleaned_df = clean_main(input_path=raw_file, output_path=cleaned_file)

        assert cleaned_file.exists()
        assert len(cleaned_df) > 0
        assert 'casual' in cleaned_df.columns
        assert 'registered' in cleaned_df.columns
        assert 'cnt' in cleaned_df.columns

        # Verify relationship holds
        assert (cleaned_df['casual'] + cleaned_df['registered'] == cleaned_df['cnt']).all()

        # Step 2: Split and scale data
        train, valid, test = split_main(input_path=cleaned_file, output_dir=processed_dir)

        # Verify files were created
        assert (processed_dir / "train.csv").exists()
        assert (processed_dir / "valid.csv").exists()
        assert (processed_dir / "test.csv").exists()

        # Verify splits
        assert len(train) > len(valid)
        assert len(train) > len(test)
        assert 'cnt' in train.columns

    def test_pipeline_handles_data_with_issues(self, temp_dirs):
        """Test that pipeline handles data with various issues"""
        raw_dir, processed_dir = temp_dirs

        # Create problematic data
        data = {
            'instant': [1, 2, 3, 4, 5],
            'dteday': ['2011-01-01', '2011-01-02', '2011-01-03', '2011-01-04', '2011-01-05'],
            'season': [1, 2, 'invalid', 4, 1],  # Invalid value
            'yr': [0, 0, 0, 0, 0],
            'mnth': [1, 1, 1, 1, 1],
            'hr': [0, 1, 2, 3, 4],
            'holiday': [0, 0, 0, 0, 0],
            'weekday': [0, 1, 2, 3, 4],
            'workingday': [1, 1, 1, 1, 1],
            'weathersit': [1, 2, 1, 2, 1],
            'temp': [0.5, 0.6, 2.0, 0.7, 0.8],  # Out of range value
            'atemp': [0.5, 0.6, 0.7, 0.7, 0.8],
            'hum': [0.8, 0.7, 0.6, 0.7, 0.8],
            'windspeed': [0.1, 0.2, 0.3, 0.2, 0.1],
            'casual': [5, 10, np.nan, 15, 20],  # NaN value
            'registered': [20, 30, 40, 50, 60],
            'cnt': [25, 40, 45, 65, 80],
            'mixed_type_col': [1, 2, 3, 1, 2]
        }
        df = pd.DataFrame(data)

        raw_file = raw_dir / "problematic_data.csv"
        df.to_csv(raw_file, index=False)

        # Clean data - should handle issues gracefully
        cleaned_file = processed_dir / "cleaned.csv"
        cleaned_df = clean_main(input_path=raw_file, output_path=cleaned_file)

        # Should have removed or fixed problematic rows
        assert len(cleaned_df) < len(df)

        # Remaining data should be valid
        assert cleaned_df['temp'].max() <= 1.0
        assert cleaned_df['casual'].notna().all()
        assert (cleaned_df['casual'] + cleaned_df['registered'] == cleaned_df['cnt']).all()

    def test_pipeline_output_ready_for_modeling(self, sample_raw_data, temp_dirs):
        """Test that pipeline output is ready for ML model training"""
        raw_dir, processed_dir = temp_dirs

        # Save and process data
        raw_file = raw_dir / "data.csv"
        sample_raw_data.to_csv(raw_file, index=False)

        cleaned_file = processed_dir / "cleaned.csv"
        clean_main(input_path=raw_file, output_path=cleaned_file)

        train, valid, test = split_main(input_path=cleaned_file, output_dir=processed_dir)

        # Verify output is model-ready
        # 1. No missing values in key columns
        assert train['cnt'].notna().all()

        # 2. Features are numeric
        numeric_cols = train.select_dtypes(include=[np.number]).columns
        assert len(numeric_cols) > 0

        # 3. Target variable exists
        assert 'cnt' in train.columns

        # 4. No redundant columns
        assert 'instant' not in train.columns
        assert 'dteday' not in train.columns
        assert 'casual' not in train.columns
        assert 'registered' not in train.columns

        # 5. Data is scaled (cnt should be in [0, 1])
        assert train['cnt'].min() >= 0
        assert train['cnt'].max() <= 1

        # 6. Same columns in all splits
        assert set(train.columns) == set(valid.columns)
        assert set(train.columns) == set(test.columns)


class TestPipelineDataQuality:
    """Tests to ensure data quality throughout the pipeline"""

    def test_no_infinite_values(self, sample_raw_data):
        """Test that pipeline doesn't introduce infinite values"""
        df = load_and_convert_types(sample_raw_data)
        df = clean_count_variables(df)

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            assert not np.isinf(df[col]).any(), f"Column {col} contains infinite values"

    def test_data_types_consistent(self, sample_raw_data):
        """Test that data types remain consistent through pipeline"""
        df = load_and_convert_types(sample_raw_data)

        # After type conversion
        assert df['temp'].dtype == float
        assert df['season'].dtype.name == 'Int64'

        # After cleaning
        df = clean_count_variables(df)
        assert df['temp'].dtype == float
        assert pd.api.types.is_integer_dtype(df['casual'])

    def test_row_count_decreases_reasonably(self, sample_raw_data):
        """Test that cleaning doesn't remove too much data"""
        original_count = len(sample_raw_data)

        df = load_and_convert_types(sample_raw_data)
        df = clean_count_variables(df)

        final_count = len(df)

        # Should retain at least 80% of data (assuming reasonable quality input)
        retention_rate = final_count / original_count
        assert retention_rate >= 0.8, f"Only retained {retention_rate*100:.1f}% of data"

    def test_column_ranges_valid(self, sample_raw_data):
        """Test that column values are within expected ranges after cleaning"""
        df = load_and_convert_types(sample_raw_data)
        df = clean_count_variables(df)

        # Weather variables should be [0, 1]
        assert df['temp'].min() >= 0 and df['temp'].max() <= 1
        assert df['hum'].min() >= 0 and df['hum'].max() <= 1

        # Count variables should be non-negative
        assert df['casual'].min() >= 0
        assert df['registered'].min() >= 0
        assert df['cnt'].min() >= 0
