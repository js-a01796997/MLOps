"""
Data Splitting and Preprocessing Pipeline

Splits cleaned data into train/validation/test sets and applies preprocessing.
"""

from pathlib import Path
import pickle
from typing import Tuple
import pandas as pd
from pandas.api.types import CategoricalDtype
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from utils.paths import get_data_path, get_models_path


def convert_column_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert columns to appropriate data types (categorical, integer, float).

    Args:
        df: DataFrame with mixed types

    Returns:
        DataFrame with converted types
    """
    print("Se cambiará el tipo de dato de las columnas season, mnth, weekday y weathersit a categóricas.")

    cat_cols = ["season", "mnth", "weekday", "weathersit"]
    int_cols = ["yr", "holiday", "workingday", "cnt", "registered", "casual", "instant", "mixed_type_col"]
    float_cols = ["temp", "atemp", "hum", "windspeed"]

    # Convert to numeric correctly
    cols = [c for c in (cat_cols + int_cols + float_cols) if c in df.columns]
    df[cols] = df[cols].apply(pd.to_numeric, errors="coerce")

    # Define categorical dtypes
    season_dtype = CategoricalDtype(categories=[1, 2, 3, 4], ordered=False)
    mnth_dtype = CategoricalDtype(categories=list(range(1, 13)), ordered=False)
    weekday_dtype = CategoricalDtype(categories=list(range(0, 7)), ordered=False)
    weathersit_dtype = CategoricalDtype(categories=[1, 2, 3, 4], ordered=False)

    dtype_map = {
        "season": season_dtype,
        "mnth": mnth_dtype,
        "weekday": weekday_dtype,
        "weathersit": weathersit_dtype,
    }

    # Apply categorical types
    for c, ctype in dtype_map.items():
        if c in df.columns:
            df[c] = df[c].astype(ctype)

    # Apply integer types
    for c in int_cols:
        if c in df.columns:
            df[c] = df[c].astype("Int64")

    # Apply float types
    for c in float_cols:
        if c in df.columns:
            df[c] = df[c].astype("float64")

    print(f"La nueva información del DataFrame: \n\n {df.info()}")
    return df


def split_data(
    df: pd.DataFrame,
    train_size: float = 0.70,
    valid_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 333
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data into train, validation, and test sets.

    Args:
        df: Input DataFrame
        train_size: Proportion of data for training
        valid_size: Proportion of data for validation
        test_size: Proportion of data for testing
        random_state: Random seed for reproducibility

    Returns:
        Tuple of (train_df, valid_df, test_df)
    """
    # Drop redundant columns
    df_new = df.drop(['instant', 'casual', 'registered', 'dteday'], axis=1)
    print("Se eliminaron las columnas instant, casual, registered y dteday por ser redundantes para el modelo")

    # Convert categorical variables to dummy variables
    df_new = pd.get_dummies(df_new, drop_first=True)

    # Split data: first train/test split, then split test into valid/test
    df_train, df_vt = train_test_split(
        df_new,
        train_size=train_size,
        test_size=(valid_size + test_size),
        random_state=random_state
    )
    df_valid, df_test = train_test_split(
        df_vt,
        train_size=(valid_size / (valid_size + test_size)),
        test_size=(test_size / (valid_size + test_size)),
        random_state=random_state
    )

    print(f"\nEl tamaño del set de entrenamiento es: {df_train.shape}")
    print(f"\nEl tamaño del set de validación es: {df_valid.shape}")
    print(f"\nEl tamaño del set de prueba es: {df_test.shape}")

    return df_train, df_valid, df_test


def scale_features(
    df_train: pd.DataFrame,
    df_valid: pd.DataFrame,
    df_test: pd.DataFrame,
    cols_to_scale: list = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, MinMaxScaler]:
    """
    Scale features using MinMaxScaler.

    Args:
        df_train: Training DataFrame
        df_valid: Validation DataFrame
        df_test: Test DataFrame
        cols_to_scale: List of column names to scale. If None, uses default columns.

    Returns:
        Tuple of (scaled_train_df, scaled_valid_df, scaled_test_df, scaler)
    """
    if cols_to_scale is None:
        cols_to_scale = ['temp', 'atemp', 'hum', 'windspeed', 'cnt']

    # Fit scaler on training data
    scaler = MinMaxScaler()
    df_train.loc[:, cols_to_scale] = scaler.fit_transform(df_train[cols_to_scale])
    print("Se escalaron las variables temp, atemp, hum, windspeed y cnt")

    # Ensure columns are float before scaling
    for df_ in (df_valid, df_test):
        for c in cols_to_scale:
            if c in df_.columns:
                df_.loc[:, c] = df_[c].astype("float64")

    # Transform validation and test sets
    df_valid.loc[:, cols_to_scale] = scaler.transform(df_valid[cols_to_scale])
    df_test.loc[:, cols_to_scale] = scaler.transform(df_test[cols_to_scale])

    print("Se escalaron las variables temp, atemp, hum, windspeed y cnt en validación y test.")
    return df_train, df_valid, df_test, scaler


def save_scaler(scaler: MinMaxScaler, scaler_path: Path = None) -> None:
    """
    Save scaler to pickle file.

    Args:
        scaler: Fitted MinMaxScaler
        scaler_path: Path to save scaler. If None, uses default path.
    """
    if scaler_path is None:
        scaler_path = get_models_path() / "minmax_scaler.pickle"

    scaler_path.parent.mkdir(parents=True, exist_ok=True)
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)

    print(f"Scaler guardado en: {scaler_path}")


def save_splits(
    df_train: pd.DataFrame,
    df_valid: pd.DataFrame,
    df_test: pd.DataFrame,
    output_dir: Path = None
) -> None:
    """
    Save train/validation/test splits to CSV files.

    Args:
        df_train: Training DataFrame
        df_valid: Validation DataFrame
        df_test: Test DataFrame
        output_dir: Directory to save files. If None, uses default processed data directory.
    """
    if output_dir is None:
        output_dir = get_data_path("processed")

    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = output_dir / "train.csv"
    valid_path = output_dir / "valid.csv"
    test_path = output_dir / "test.csv"

    df_train.to_csv(train_path, index=False)
    df_valid.to_csv(valid_path, index=False)
    df_test.to_csv(test_path, index=False)

    print("\nSe guardaron los datasets en la carpeta data/processed")


def main(input_path: Path = None, output_dir: Path = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Main function to run the complete data splitting and preprocessing pipeline.

    Args:
        input_path: Path to cleaned input CSV. If None, uses default.
        output_dir: Directory to save output files. If None, uses default.

    Returns:
        Tuple of (train_df, valid_df, test_df)
    """
    # Get paths using utils
    if input_path is None:
        input_path = get_data_path("processed") / "bike_sharing_cleaned.csv"

    if output_dir is None:
        output_dir = get_data_path("processed")

    # Load data
    print(f"Loading data from: {input_path}")
    df = pd.read_csv(input_path)
    print(f"La info del DataFrame: \n\n {df.info()}")

    # Convert column types
    df = convert_column_types(df)

    # Split data
    df_train, df_valid, df_test = split_data(df)

    # Scale features
    df_train, df_valid, df_test, scaler = scale_features(df_train, df_valid, df_test)

    # Save scaler
    save_scaler(scaler)

    # Save splits
    save_splits(df_train, df_valid, df_test, output_dir)

    return df_train, df_valid, df_test


if __name__ == "__main__":
    main()
