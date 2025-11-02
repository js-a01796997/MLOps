"""
Inference pipeline for bike sharing predictions.
Loads models from MLflow and makes predictions on raw data.
"""

import argparse
import warnings
from pathlib import Path
from typing import Optional, Any

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

from utils.config import load_config
from utils.mlflow_setup import setup_mlflow
from data_cleaning import (
    load_and_convert_types,
    clean_season,
    clean_yr,
    clean_mnth,
    clean_workingday,
    clean_weekday,
    clean_weather_variables,
    clean_holidays,
    clean_hr,
    clean_weathersit
)
from clean_utils import to_int

warnings.filterwarnings('ignore')


def list_available_models():
    client = MlflowClient()
    models = []

    for rm in client.search_registered_models():
        model_name = rm.name
        for version in rm.latest_versions:
            models.append({
                'name': model_name,
                'version': version.version,
                'stage': version.current_stage,
                'run_id': version.run_id,
                'creation_time': pd.to_datetime(version.creation_timestamp, unit='ms')
            })

    if models:
        df = pd.DataFrame(models)
        return df.sort_values(['name', 'version'], ascending=[True, False])
    else:
        return pd.DataFrame(columns=['name', 'version', 'stage', 'run_id', 'creation_time'])


def load_model_from_mlflow(model_name: str, stage: str = "Production", version: Optional[int] = None):
    if version:
        model_uri = f"models:/{model_name}/{version}"
        print(f"Loading model: {model_name} version {version}")
    else:
        model_uri = f"models:/{model_name}/{stage}"
        print(f"Loading model: {model_name} stage {stage}")

    try:
        model = mlflow.sklearn.load_model(model_uri)
        print("Model loaded successfully")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Use --list-models to see available models")
        raise


def clean_raw_data(df: pd.DataFrame, include_target: bool = False) -> pd.DataFrame:
    print("Cleaning raw data...")

    df = load_and_convert_types(df)
    df = clean_season(df)
    df = clean_yr(df)
    df = clean_mnth(df)
    df = clean_workingday(df)
    df = clean_weekday(df)
    df = clean_weather_variables(df)

    if 'mixed_type_col' in df.columns:
        df = df.drop('mixed_type_col', axis=1)

    if include_target and all(col in df.columns for col in ['casual', 'registered', 'cnt']):
        from data_cleaning import clean_count_variables
        df = clean_count_variables(df)

    df = clean_holidays(df)
    df = clean_hr(df)
    df = clean_weathersit(df)

    int_cols = ['instant', 'season', 'yr', 'mnth', 'hr', 'holiday', 'weekday', 'workingday', 'weathersit']
    for col in int_cols:
        if col in df.columns:
            to_int(df, col)

    cols_to_drop = ['instant', 'casual', 'registered', 'dteday']
    df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])

    return df


def preprocess_for_model(df: pd.DataFrame) -> pd.DataFrame:
    from sklearn.preprocessing import MinMaxScaler
    from data_split import convert_column_types

    df = convert_column_types(df)

    if 'cnt' in df.columns:
        df = df.drop(columns=['cnt'])

    numeric_cols = ['temp', 'atemp', 'hum', 'windspeed']
    categorical_cols = ['season', 'mnth', 'weekday', 'weathersit']

    scaler = MinMaxScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True, dtype=int)

    print(f"Preprocessing complete. Final shape: {df.shape}")
    return df


def predict_from_csv(model, input_csv: str, output_csv: Optional[str] = None) -> pd.DataFrame:
    print(f"Reading data from: {input_csv}")
    raw_df = pd.read_csv(input_csv)

    cleaned_df = clean_raw_data(raw_df, include_target=False)
    preprocessed_df = preprocess_for_model(cleaned_df)

    print("Making predictions...")
    predictions = model.predict(preprocessed_df)

    result_df = raw_df.copy()
    result_df['predicted_cnt'] = predictions

    if output_csv:
        Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
        result_df.to_csv(output_csv, index=False)
        print(f"Predictions saved to: {output_csv}")

    return result_df


def predict_from_dataframe(model, df: pd.DataFrame) -> np.ndarray:
    cleaned_df = clean_raw_data(df.copy(), include_target=False)
    preprocessed_df = preprocess_for_model(cleaned_df)
    predictions = model.predict(preprocessed_df)
    return predictions


def main():
    parser = argparse.ArgumentParser(description="Inference pipeline for bike sharing predictions")
    parser.add_argument("--config", default="config/models_config.yaml", help="Config file path")
    parser.add_argument("--list-models", action="store_true", help="List all available models in MLflow")
    parser.add_argument("--model-name", type=str, help="Model name to load from MLflow")
    parser.add_argument("--stage", type=str, default="Production", help="Model stage (Production, Staging, None)")
    parser.add_argument("--version", type=int, help="Specific model version (overrides stage)")
    parser.add_argument("--input", type=str, help="Input CSV file with raw data")
    parser.add_argument("--output", type=str, help="Output CSV file for predictions")

    args = parser.parse_args()

    config = load_config(args.config)
    setup_mlflow(config)

    if args.list_models:
        print("\nAvailable models in MLflow:")
        print("="*80)
        models_df = list_available_models()
        if not models_df.empty:
            print(models_df.to_string(index=False))
        else:
            print("No models found in registry")
        return

    if not args.model_name:
        print("Error: --model-name is required")
        print("Use --list-models to see available models")
        return

    if not args.input:
        print("Error: --input is required")
        return

    model = load_model_from_mlflow(args.model_name, stage=args.stage, version=args.version)

    predictions_df = predict_from_csv(model, args.input, args.output)

    print("\nPrediction summary:")
    print(f"  Mean predicted count: {predictions_df['predicted_cnt'].mean():.2f}")
    print(f"  Min predicted count: {predictions_df['predicted_cnt'].min():.2f}")
    print(f"  Max predicted count: {predictions_df['predicted_cnt'].max():.2f}")
    print(f"\nFirst 5 predictions:")
    print(predictions_df[['predicted_cnt']].head().to_string())


if __name__ == "__main__":
    main()
