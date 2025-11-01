"""
Model Comparison Script

Evaluates and compares multiple trained models on the test set.
Logs metrics to MLflow for tracking.
"""

import pickle
from pathlib import Path
from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import mlflow
import mlflow.sklearn

from utils.paths import get_project_root, get_data_path, get_models_path


def load_models(models_dir: Path) -> Dict[str, Any]:
    """
    Load trained models from pickle files.

    Args:
        models_dir: Path to the models directory

    Returns:
        Dictionary mapping model names to model objects
    """
    models = {}
    model_files = {
        'LinearRegression': 'modelo_LR.pickle',
        'Lasso': 'modelo_lasso.pickle',
        'Ridge': 'modelo_ridge.pickle'
    }

    for name, filename in model_files.items():
        model_path = models_dir / filename
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        with open(model_path, 'rb') as f:
            models[name] = pickle.load(f)
    
    return models


def load_test_data(test_path: Path) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load and prepare test data.

    Args:
        test_path: Path to the test CSV file

    Returns:
        Tuple of (X_test, y_test) DataFrames
    """
    if not test_path.exists():
        raise FileNotFoundError(f"Test data file not found: {test_path}")
    
    df_test = pd.read_csv(test_path)
    y_test = df_test['cnt']
    X_test = df_test.drop('cnt', axis=1)

    # Impute missing values with median
    for col in ['yr', 'hr', 'holiday', 'workingday']:
        if col in X_test.columns:
            X_test[col] = X_test[col].fillna(X_test[col].median())

    return X_test, y_test


def evaluate_model(model: Any, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
    """
    Evaluate a single model on test data.

    Args:
        model: Trained model with predict() method
        X_test: Test features
        y_test: Test target values

    Returns:
        Dictionary with RMSE and MAE metrics
    """
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    
    return {'RMSE': rmse, 'MAE': mae}


def compare_models(
    models: Dict[str, Any],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    experiment_name: str = "bs_evaluation"
) -> pd.DataFrame:
    """
    Compare multiple models on test data and log to MLflow.

    Args:
        models: Dictionary mapping model names to model objects
        X_test: Test features
        y_test: Test target values
        experiment_name: MLflow experiment name

    Returns:
        DataFrame with comparison results
    """
    # Setup MLflow
    mlflow.set_experiment(experiment_name)
    
    results = {}
    
    with mlflow.start_run(run_name="compare_on_test", tags={"stage": "test"}):
        for name, model in models.items():
            with mlflow.start_run(run_name=f"test_{name}", nested=True, tags={"model": name}):
                metrics = evaluate_model(model, X_test, y_test)
                
                # Log metrics to MLflow
                mlflow.log_metric("rmse_test", metrics['RMSE'])
                mlflow.log_metric("mae_test", metrics['MAE'])
                
                results[name] = metrics

        # Convert results to DataFrame for better visualization
        results_df = pd.DataFrame(results).T
        print("\n" + "="*80)
        print("MODEL COMPARISON RESULTS (Test Set)")
        print("="*80)
        print(results_df)
        print("="*80 + "\n")
        
        return results_df


def main():
    """
    Main function to run model comparison.

    Raises:
        FileNotFoundError: If models or test data files don't exist
        ValueError: If no models are loaded or test data is empty
    """
    # Get paths using utils
    project_root = get_project_root()
    models_dir = get_models_path()
    test_path = get_data_path("processed") / "test.csv"

    # Validate paths
    if not models_dir.exists():
        raise FileNotFoundError(f"Models directory not found: {models_dir}")
    if not test_path.exists():
        raise FileNotFoundError(
            f"Test data file not found: {test_path}\n"
            f"Please ensure test.csv exists in data/processed/"
        )

    # Load models and test data
    print("Loading models...")
    try:
        models = load_models(models_dir)
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"Error loading models: {e}\n"
            f"Please ensure model pickle files exist in {models_dir}"
        ) from e

    if not models:
        raise ValueError(f"No models found in {models_dir}")

    print(f"Loaded {len(models)} models: {list(models.keys())}")

    print("Loading test data...")
    try:
        X_test, y_test = load_test_data(test_path)
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"Error loading test data: {e}\n"
            f"Please ensure test.csv exists and is valid"
        ) from e

    if X_test.empty or y_test.empty:
        raise ValueError(f"Test data is empty in {test_path}")

    print(f"Test set shape: {X_test.shape}")

    # Compare models
    print("Evaluating models on test set...")
    try:
        results_df = compare_models(models, X_test, y_test)
    except Exception as e:
        raise RuntimeError(f"Error during model comparison: {str(e)}") from e
    
    return results_df


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        print(f"\n❌ Error: {e}")
        exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)