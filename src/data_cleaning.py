"""
Data Cleaning Pipeline

Comprehensive data cleaning for bike sharing dataset.
Follows modular structure with separate functions for each cleaning step.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List
from clean_utils import clean_weather_var, to_int, fix_hour, detect_outliers
from utils.paths import get_project_root, get_data_path


def load_and_convert_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Load data and convert types to numeric/date formats.

    Args:
        df: Raw DataFrame

    Returns:
        DataFrame with converted types
    """
    print("\n1. DataFrame information before cleaning:\n")
    print(df.info())

    # Column definitions
    to_int_cols = ['instant', 'season', 'yr', 'mnth', 'hr', 'holiday', 'weekday', 
                   'workingday', 'weathersit', 'casual', 'registered', 'cnt', 'mixed_type_col']
    to_float_cols = ['temp', 'atemp', 'hum', 'windspeed']

    # Find non-numeric values
    cols = to_float_cols + to_int_cols
    all_non_numeric_values = []
    for col in cols:
        non_numeric_values = df[pd.to_numeric(df[col], errors='coerce').isna()][col].unique()
        all_non_numeric_values.extend(non_numeric_values)

    all_non_numeric_values = list(set(all_non_numeric_values))
    print(f'\n2. Non-numeric values found in columns {cols}: {all_non_numeric_values}')

    # Replace non-numeric values with NaN
    for col in cols:
        df[col] = df[col].replace(all_non_numeric_values, np.nan)

    # Convert to numeric types
    df[to_int_cols] = df[to_int_cols].astype(float).astype('Int64')
    df[to_float_cols] = df[to_float_cols].astype(float)

    # Convert date column
    df['dteday'] = pd.to_datetime(df['dteday'].str.strip(), format='%Y-%m-%d', errors='coerce')
    print("\n3. Converted column 'dteday' to date type.")

    return df


def clean_season(df: pd.DataFrame) -> pd.DataFrame:
    """Clean season column by matching valid values from same date."""
    invalid_seasons = df[~df['season'].isin([1, 2, 3, 4])]['season'].unique()

    for index, row in df.iterrows():
        if row['season'] in invalid_seasons:
            valid_season_rows = df[(df['dteday'] == row['dteday']) & 
                                   (df['season'].isin([1, 2, 3, 4]))]
            if not valid_season_rows.empty:
                df.loc[index, 'season'] = valid_season_rows.iloc[0]['season']

    print('\n4. Cleaned noise in season variable')
    return df


def clean_yr(df: pd.DataFrame) -> pd.DataFrame:
    """Clean yr column by inferring from date."""
    df['year_aux'] = df['dteday'].dt.year
    valid_yr = [0, 1]

    for index, row in df.iterrows():
        if pd.notna(row['yr']) and row['yr'] not in valid_yr:
            if pd.notna(row['year_aux']):
                if row['year_aux'] == 2011:
                    df.loc[index, 'yr'] = 0
                elif row['year_aux'] == 2012:
                    df.loc[index, 'yr'] = 1
                else:
                    df.loc[index, 'yr'] = np.nan
        elif pd.isna(row['yr']):
            if pd.notna(row['year_aux']):
                if row['year_aux'] == 2011:
                    df.loc[index, 'yr'] = 0
                elif row['year_aux'] == 2012:
                    df.loc[index, 'yr'] = 1
                else:
                    df.loc[index, 'yr'] = np.nan

    df = df.drop('year_aux', axis=1)
    print('\n5. Cleaned noise in yr variable')
    return df


def clean_mnth(df: pd.DataFrame) -> pd.DataFrame:
    """Clean mnth column by inferring from date."""
    invalid_mnth = df[~df['mnth'].isin(range(1, 13))]['mnth'].unique()

    for index, row in df.iterrows():
        if row['mnth'] in invalid_mnth:
            if pd.notna(row['dteday']):
                df.loc[index, 'mnth'] = row['dteday'].month
            else:
                df.loc[index, 'mnth'] = np.nan

    print('\n6. Cleaned noise in mnth variable')
    return df


def clean_workingday(df: pd.DataFrame) -> pd.DataFrame:
    """Clean workingday column by inferring from date and holiday."""
    invalid_workingdays = df[~df['workingday'].isin([0, 1])]['workingday'].unique()

    for index, row in df.iterrows():
        if row['workingday'] in invalid_workingdays:
            if pd.notna(row['dteday']):
                if row['dteday'].weekday() in [5, 6] or (pd.notna(row['holiday']) and row['holiday'] == 1):
                    df.loc[index, 'workingday'] = 0
                else:
                    df.loc[index, 'workingday'] = 1
            else:
                df.loc[index, 'workingday'] = np.nan

    df['workingday'] = pd.to_numeric(df['workingday'], errors='coerce').astype('Int64')
    print('\n7. Cleaned noise in workingday variable')
    return df


def clean_weekday(df: pd.DataFrame) -> pd.DataFrame:
    """Clean weekday column by inferring from date."""
    invalid_weekdays = df[~df['weekday'].isin(range(7))]['weekday'].unique()

    for index, row in df.iterrows():
        if row['weekday'] in invalid_weekdays:
            if pd.notna(row['dteday']):
                df.loc[index, 'weekday'] = row['dteday'].weekday()
            else:
                df.loc[index, 'weekday'] = np.nan

    df['weekday'] = pd.to_numeric(df['weekday'], errors='coerce').astype('Int64')
    print('\n8. Cleaned noise in weekday variable')
    return df


def clean_weather_variables(df: pd.DataFrame) -> pd.DataFrame:
    """Clean weather variables (temp, atemp, hum, windspeed)."""
    clean_weather_var(df, 'temp', 10)
    print('\n9. Cleaned noise in temp variable')

    clean_weather_var(df, 'atemp', 11)
    print('\n10. Cleaned noise in atemp variable')

    clean_weather_var(df, 'hum', 12)
    print('\n11. Cleaned noise in hum variable')

    clean_weather_var(df, 'windspeed', 13)
    print('\n12. Cleaned noise in windspeed variable')

    return df


def clean_count_variables(df: pd.DataFrame, iqr_multiplier: float = 2.5) -> pd.DataFrame:
    """
    Clean count variables (casual, registered, cnt).
    
    This is the most complex cleaning step involving:
    - Rounding to integers
    - Taking absolute values
    - Outlier detection and replacement
    - Filling missing values using relationship: casual + registered = cnt
    - Fixing inconsistent rows
    - Removing rows with NaN or negative values
    """
    # Round values to integers
    print("="*80)
    print("ROUNDING VALUES TO INTEGERS")
    print("="*80)

    df['casual'] = df['casual'].round().astype('Int64')
    df['registered'] = df['registered'].round().astype('Int64')
    df['cnt'] = df['cnt'].round().astype('Int64')
    print("All values in casual, registered, and cnt have been rounded to integers")

    # Use absolute value to avoid negative values
    print("="*80)
    print("USING THE ABSOLUTE VALUE TO AVOID NEGATIVE VALUES")
    print("="*80)

    df['cnt'] = df['cnt'].abs()
    df['casual'] = df['casual'].abs()
    df['registered'] = df['registered'].abs()
    print("All values in casual, registered, and cnt have been converted to absolute values")

    # Outlier detection
    print("="*80)
    print(f"OUTLIER DETECTION (IQR Multiplier: {iqr_multiplier})")
    print("="*80)

    casual_outliers = detect_outliers(df, 'casual', iqr_multiplier)
    registered_outliers = detect_outliers(df, 'registered', iqr_multiplier)
    cnt_outliers = detect_outliers(df, 'cnt', iqr_multiplier)

    # Identify rows where outliers appear in only one column
    casual_only = set(casual_outliers) - set(registered_outliers) - set(cnt_outliers)
    registered_only = set(registered_outliers) - set(casual_outliers) - set(cnt_outliers)
    cnt_only = set(cnt_outliers) - set(casual_outliers) - set(registered_outliers)

    print("\n" + "="*80)
    print("OUTLIERS IN SINGLE COLUMNS")
    print("="*80)
    print(f"Outliers only in 'casual': {len(casual_only)}")
    print(f"Outliers only in 'registered': {len(registered_only)}")
    print(f"Outliers only in 'cnt': {len(cnt_only)}")

    # Calculate medians
    casual_median = df['casual'].median()
    registered_median = df['registered'].median()
    cnt_median = df['cnt'].median()

    print("\n" + "="*80)
    print("REPLACING SINGLE-COLUMN OUTLIERS WITH MEDIAN")
    print("="*80)
    print(f"Median values:")
    print(f"  casual: {casual_median}")
    print(f"  registered: {registered_median}")
    print(f"  cnt: {cnt_median}")

    # Replace outliers with median
    if len(casual_only) > 0:
        print(f"\nReplacing {len(casual_only)} outliers in 'casual' with median ({casual_median})")
        df.loc[list(casual_only), 'casual'] = casual_median

    if len(registered_only) > 0:
        print(f"Replacing {len(registered_only)} outliers in 'registered' with median ({registered_median})")
        df.loc[list(registered_only), 'registered'] = registered_median

    if len(cnt_only) > 0:
        print(f"Replacing {len(cnt_only)} outliers in 'cnt' with median ({cnt_median})")
        df.loc[list(cnt_only), 'cnt'] = cnt_median

    print(f"\nTotal outliers replaced: {len(casual_only) + len(registered_only) + len(cnt_only)}")

    # Fill missing values using relationship: casual + registered = cnt
    print("="*80)
    print("CALCULATING MISSING VALUES USING casual + registered = cnt")
    print("="*80)

    casual_nan_before = df['casual'].isna().sum()
    registered_nan_before = df['registered'].isna().sum()
    cnt_nan_before = df['cnt'].isna().sum()

    print(f"Missing values before calculation:")
    print(f"  casual: {casual_nan_before}")
    print(f"  registered: {registered_nan_before}")
    print(f"  cnt: {cnt_nan_before}")

    # Fill missing values
    df['cnt'] = df['cnt'].fillna(df['casual'] + df['registered'])
    df['casual'] = df['casual'].fillna(df['cnt'] - df['registered'])
    df['registered'] = df['registered'].fillna(df['cnt'] - df['casual'])

    casual_nan_after = df['casual'].isna().sum()
    registered_nan_after = df['registered'].isna().sum()
    cnt_nan_after = df['cnt'].isna().sum()

    print(f"\nMissing values after calculation:")
    print(f"  casual: {casual_nan_after}")
    print(f"  registered: {registered_nan_after}")
    print(f"  cnt: {cnt_nan_after}")

    total_filled = (casual_nan_before - casual_nan_after) + (registered_nan_before - registered_nan_after) + (cnt_nan_before - cnt_nan_after)
    print(f"\nTotal missing values filled: {total_filled}")

    # Fix inconsistent rows
    print("="*80)
    print("FIXING INCONSISTENT ROWS (casual + registered != cnt)")
    print("="*80)

    inconsistent_mask = (df['casual'] + df['registered']) != df['cnt']
    inconsistent_count = inconsistent_mask.sum()

    print(f"Rows where casual + registered != cnt: {inconsistent_count} ({inconsistent_count/len(df)*100:.2f}%)")

    if inconsistent_count > 0:
        print(f"\nRecalculating the biggest value in each inconsistent row\n")

        recalc_casual = 0
        recalc_registered = 0
        recalc_cnt = 0

        for idx in df[inconsistent_mask].index:
            casual_val = df.loc[idx, 'casual']
            registered_val = df.loc[idx, 'registered']
            cnt_val = df.loc[idx, 'cnt']

            max_val = max(casual_val, registered_val, cnt_val)

            if casual_val == max_val:
                df.loc[idx, 'casual'] = df.loc[idx, 'cnt'] - df.loc[idx, 'registered']
                recalc_casual += 1
            elif registered_val == max_val:
                df.loc[idx, 'registered'] = df.loc[idx, 'cnt'] - df.loc[idx, 'casual']
                recalc_registered += 1
            else:
                df.loc[idx, 'cnt'] = df.loc[idx, 'casual'] + df.loc[idx, 'registered']
                recalc_cnt += 1

        print(f"Recalculation summary:")
        print(f"  casual recalculated: {recalc_casual}")
        print(f"  registered recalculated: {recalc_registered}")
        print(f"  cnt recalculated: {recalc_cnt}")

        still_inconsistent = ((df['casual'] + df['registered']) != df['cnt']).sum()
        print(f"\nVerification after recalculation:")
        print(f"  Inconsistent rows remaining: {still_inconsistent}")

        if still_inconsistent == 0:
            print("\nAll rows now follow the relationship: casual + registered = cnt")
        else:
            print("\nWarning: Some inconsistent rows remain!")

    # Delete rows with NaN values
    rows_before = len(df)
    df = df.dropna(subset=['casual', 'registered', 'cnt'])
    rows_after = len(df)
    rows_deleted = rows_before - rows_after
    print(f"\nRows before deletion: {rows_before}")
    print(f"Rows after deletion: {rows_after}")
    print(f"Rows deleted: {rows_deleted} ({rows_deleted/rows_before*100:.2f}%)")

    # Check and delete negative values
    print("="*80)
    print("CHECKING FOR NEGATIVE VALUES")
    print("="*80)

    negative_casual = (df['casual'] < 0).sum()
    negative_registered = (df['registered'] < 0).sum()
    negative_cnt = (df['cnt'] < 0).sum()

    print(f"Negative values found:")
    print(f"  casual: {negative_casual}")
    print(f"  registered: {negative_registered}")
    print(f"  cnt: {negative_cnt}")

    total_negatives = negative_casual + negative_registered + negative_cnt
    print(f"\nTotal negative values: {total_negatives}")

    if total_negatives > 0:
        negative_mask = (df['casual'] < 0) | (df['registered'] < 0) | (df['cnt'] < 0)
        print(f"\nRows with negative values:")
        print(df[negative_mask][['instant', 'dteday', 'casual', 'registered', 'cnt']])

    # Delete rows with negative values
    print("="*80)
    print("DELETING ROWS WITH NEGATIVE VALUES")
    print("="*80)

    rows_before = len(df)
    negative_mask = (df['casual'] < 0) | (df['registered'] < 0) | (df['cnt'] < 0)
    df = df[~negative_mask]
    rows_after = len(df)
    rows_deleted = rows_before - rows_after

    print(f"Rows before deletion: {rows_before}")
    print(f"Rows after deletion: {rows_after}")
    print(f"Rows deleted: {rows_deleted} ({rows_deleted/rows_before*100:.2f}%)")

    print('\n14. Cleaned noise in variables: casual, registered and cnt')
    return df


def clean_holidays(df: pd.DataFrame) -> pd.DataFrame:
    """Clean holidays column using official US/DC holiday list."""
    holidays = [
        '2011-01-17', '2011-02-21', '2011-04-15', '2011-05-30', '2011-07-04',
        '2011-09-05', '2011-10-10', '2011-11-11', '2011-11-24', '2011-12-26',
        '2012-01-02', '2012-01-16', '2012-02-20', '2012-04-16', '2012-05-28',
        '2012-07-04', '2012-09-03', '2012-10-08', '2012-11-12', '2012-11-22',
        '2012-12-25',
    ]

    holidays = [pd.to_datetime(date) for date in holidays]
    holidays_mask = df['dteday'].isin(holidays)
    df['holiday'] = holidays_mask.astype(int)
    print('\n15. Cleaned noise in holidays variable')
    return df


def clean_hr(df: pd.DataFrame) -> pd.DataFrame:
    """Clean hr (hour) column by removing duplicates and fixing ranges."""
    df = df.drop_duplicates(subset=['dteday', 'season', 'yr', 'mnth', 'hr', 
                                    'holiday', 'weekday', 'workingday'])
    hour_mask_more_than_23 = df["hr"] > 23
    df.loc[hour_mask_more_than_23, 'hr'] = np.nan
    new_df = df.groupby('dteday', group_keys=False)[['hr', 'dteday']].apply(fix_hour)
    df['hr'] = new_df['hr']
    print('\n16. Cleaned noise in hr variable')
    return df


def clean_weathersit(df: pd.DataFrame) -> pd.DataFrame:
    """Clean weathersit column by removing invalid values and forward filling."""
    weathersit_invalid_mask = df['weathersit'] > 4
    df.loc[weathersit_invalid_mask, 'weathersit'] = np.nan
    df['weathersit'] = df['weathersit'].bfill()
    print('\n17. Cleaned noise in weathersit variable')
    return df


def main(input_path: Path = None, output_path: Path = None) -> pd.DataFrame:
    """
    Main function to run the complete data cleaning pipeline.

    Args:
        input_path: Path to input CSV file. If None, uses default.
        output_path: Path to output CSV file. If None, uses default.

    Returns:
        Cleaned DataFrame

    Raises:
        FileNotFoundError: If input file doesn't exist
        pd.errors.EmptyDataError: If input file is empty
        ValueError: If DataFrame is empty after cleaning
    """
    # Get paths
    if input_path is None:
        input_path = get_data_path("raw") / "bike_sharing_modified.csv"
    if output_path is None:
        output_path = get_data_path("processed") / "bike_sharing_cleaned.csv"

    # Validate input path
    if not input_path.exists():
        raise FileNotFoundError(
            f"Input file not found: {input_path}\n"
            f"Please ensure the file exists or provide a valid input_path."
        )

    # Load data
    print(f"Loading data from: {input_path}")
    try:
        bike_sharing = pd.read_csv(input_path)
    except pd.errors.EmptyDataError:
        raise pd.errors.EmptyDataError(f"Input file {input_path} is empty")
    except Exception as e:
        raise IOError(f"Error reading input file {input_path}: {str(e)}") from e

    if bike_sharing.empty:
        raise ValueError(f"Input file {input_path} contains no data")

    # Run cleaning pipeline
    bike_sharing = load_and_convert_types(bike_sharing)
    bike_sharing = clean_season(bike_sharing)
    bike_sharing = clean_yr(bike_sharing)
    bike_sharing = clean_mnth(bike_sharing)
    bike_sharing = clean_workingday(bike_sharing)
    bike_sharing = clean_weekday(bike_sharing)
    bike_sharing = clean_weather_variables(bike_sharing)
    
    # Drop mixed_type_col
    bike_sharing = bike_sharing.drop('mixed_type_col', axis=1)
    print('\n13. Removed column mixed_type_col\n')

    bike_sharing = clean_count_variables(bike_sharing)
    bike_sharing = clean_holidays(bike_sharing)
    bike_sharing = clean_hr(bike_sharing)
    bike_sharing = clean_weathersit(bike_sharing)

    # Convert integer columns
    int_cols = ['instant', 'season', 'yr', 'mnth', 'hr', 'holiday', 'weekday', 'workingday', 'weathersit']
    for col in int_cols:
        to_int(bike_sharing, col)
    print('\n18. Converted integer columns')

    # Final info
    print("\n19. DataFrame information after cleaning:\n")
    print(bike_sharing.info())

    # Validate cleaned data
    if bike_sharing.empty:
        raise ValueError("DataFrame is empty after cleaning. Check input data and cleaning steps.")

    # Save cleaned data
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        bike_sharing.to_csv(output_path, index=False)
        print(f"\n20. Cleaned DataFrame saved to: {output_path}\n")
    except Exception as e:
        raise IOError(f"Error saving cleaned data to {output_path}: {str(e)}") from e

    return bike_sharing


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError, IOError) as e:
        print(f"\n❌ Error: {e}")
        exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
