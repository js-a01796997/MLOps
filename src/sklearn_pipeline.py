"""
Scikit-learn Pipeline for Automated ML Workflow
Combines preprocessing, training, and evaluation in a unified pipeline

This module integrates with the existing preprocessing from data_split.py
to create end-to-end scikit-learn pipelines.
"""

import sys
import pickle
import argparse
import importlib
import traceback
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, Union
import warnings

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.base import BaseEstimator, TransformerMixin
import mlflow
import mlflow.sklearn

from utils.config import load_config
from utils.mlflow_setup import setup_mlflow
from data_split import convert_column_types, split_data

warnings.filterwarnings('ignore')


class FeatureSelector(BaseEstimator, TransformerMixin):
    """Custom transformer to select and drop specific columns"""

    def __init__(self, columns_to_drop=None):
        self.columns_to_drop = columns_to_drop or ['instant', 'casual', 'registered', 'dteday']

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()
        cols_to_drop = [col for col in self.columns_to_drop if col in X_copy.columns]
        if cols_to_drop:
            X_copy = X_copy.drop(columns=cols_to_drop)
        return X_copy


class DataFrameConverter(BaseEstimator, TransformerMixin):
    """Ensures output is a DataFrame with proper column names"""

    def __init__(self):
        self.feature_names_ = None

    def fit(self, X, y=None):
        if hasattr(X, 'columns'):
            self.feature_names_ = X.columns.tolist()
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            return X
        elif self.feature_names_ is not None:
            return pd.DataFrame(X, columns=self.feature_names_)
        return X


def create_preprocessing_pipeline(
    numeric_features: list,
    categorical_features: list,
    scale_target: bool = False
) -> ColumnTransformer:
    """
    Create preprocessing pipeline matching data_split.py preprocessing

    Uses MinMaxScaler for numeric features (matching existing preprocessing)
    and OneHotEncoder with drop='first' for categorical features

    Args:
        numeric_features: List of numeric feature names (e.g., temp, atemp, hum, windspeed)
        categorical_features: List of categorical feature names (e.g., season, mnth, weekday)
        scale_target: Whether to include target column in scaling (default: False for training)

    Returns:
        ColumnTransformer with preprocessing steps
    """
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', MinMaxScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='passthrough'
    )

    return preprocessor


def create_full_pipeline(
    model,
    numeric_features: list,
    categorical_features: list,
    columns_to_drop: list = None
) -> Pipeline:
    """
    Create complete pipeline with preprocessing and model

    Args:
        model: Scikit-learn estimator (e.g., LinearRegression, Ridge, etc.)
        numeric_features: List of numeric feature names
        categorical_features: List of categorical feature names
        columns_to_drop: List of columns to drop before preprocessing

    Returns:
        Complete Pipeline object
    """
    preprocessor = create_preprocessing_pipeline(numeric_features, categorical_features)

    pipeline = Pipeline(steps=[
        ('feature_selector', FeatureSelector(columns_to_drop=columns_to_drop)),
        ('preprocessor', preprocessor),
        ('model', model)
    ])

    return pipeline


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate regression metrics

    Args:
        y_true: True target values
        y_pred: Predicted target values

    Returns:
        Dictionary of metrics
    """
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100 if mask.any() else np.inf

    return {
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'mape': mape
    }


def evaluate_pipeline(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate pipeline on train, validation, and test sets

    Args:
        pipeline: Fitted pipeline
        X_train, y_train: Training data
        X_valid, y_valid: Validation data
        X_test, y_test: Test data

    Returns:
        Dictionary with metrics for each dataset
    """
    y_train_pred = pipeline.predict(X_train)
    y_valid_pred = pipeline.predict(X_valid)
    y_test_pred = pipeline.predict(X_test)

    train_metrics = calculate_metrics(y_train, y_train_pred)
    valid_metrics = calculate_metrics(y_valid, y_valid_pred)
    test_metrics = calculate_metrics(y_test, y_test_pred)

    return {
        'train': train_metrics,
        'valid': valid_metrics,
        'test': test_metrics
    }


def train_pipeline_with_search(
    base_pipeline: Pipeline,
    param_grid: Dict[str, Any],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv_folds: int = 5,
    search_method: str = 'grid',
    n_iter: int = 10,
    scoring: str = 'neg_mean_squared_error'
) -> Tuple[Pipeline, Dict[str, Any]]:
    """
    Train pipeline with hyperparameter search

    Args:
        base_pipeline: Base pipeline to optimize
        param_grid: Parameter grid for search
        X_train, y_train: Training data
        cv_folds: Number of CV folds
        search_method: 'grid' or 'random'
        n_iter: Number of iterations for RandomizedSearchCV
        scoring: Scoring metric

    Returns:
        Tuple of (best_pipeline, best_params)
    """
    adjusted_param_grid = {
        f'model__{key}': value for key, value in param_grid.items()
    }

    if isinstance(base_pipeline, Pipeline):
        search_param_grid = {f"model__{k}": v for k, v in param_grid.items()}
    else:
        search_param_grid = param_grid


    if search_method.lower() == 'random':
        search = RandomizedSearchCV(
            estimator=base_pipeline,
            param_distributions=search_param_grid,
            n_iter=n_iter,
            cv=cv_folds,
            scoring=scoring,
            n_jobs=-1,
            verbose=1,
            random_state=42
        )
        print(f"Running RandomizedSearchCV with {n_iter} iterations and {cv_folds}-fold CV...")
    else:
        search = GridSearchCV(
            estimator=base_pipeline,
            param_grid=adjusted_param_grid,
            cv=cv_folds,
            scoring=scoring,
            n_jobs=-1,
            verbose=1
        )
        print(f"Running GridSearchCV with {cv_folds}-fold CV...")

    search.fit(X_train, y_train)

    best_pipeline = search.best_estimator_
    best_params = search.best_params_

    best_params_clean = {
        key.replace('model__', ''): value for key, value in best_params.items()
    }

    return best_pipeline, best_params_clean


def load_and_prepare_data(
    config: Dict[str, Any],
    use_existing_splits: bool = False
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Load data and prepare for pipeline

    Args:
        config: Configuration dictionary
        use_existing_splits: If True, load existing processed splits.
                            If False, load cleaned data and split manually.

    Returns:
        Tuple of (X_train, y_train, X_valid, y_valid, X_test, y_test)
    """
    target_col = config['data']['target_column']

    if use_existing_splits:
        print("Loading existing preprocessed data splits...")
        train_df = pd.read_csv(config['data']['train'])
        valid_df = pd.read_csv(config['data']['valid'])
        test_df = pd.read_csv(config['data']['test'])

        X_train = train_df.drop(columns=[target_col])
        y_train = train_df[target_col]

        X_valid = valid_df.drop(columns=[target_col])
        y_valid = valid_df[target_col]

        X_test = test_df.drop(columns=[target_col])
        y_test = test_df[target_col]

        print("Note: Data is already preprocessed (scaled and encoded)")
    else:
        cleaned_data_path = Path("data/processed/bike_sharing_cleaned.csv")

        if not cleaned_data_path.exists():
            raise FileNotFoundError(
                f"Cleaned data not found at {cleaned_data_path}. "
                "Please run data_cleaning.py first."
            )

        print(f"Loading cleaned data from: {cleaned_data_path}")
        df = pd.read_csv(cleaned_data_path)

        df = convert_column_types(df)

        cols_to_drop = ['instant', 'casual', 'registered', 'dteday']
        df_clean = df.drop(columns=[col for col in cols_to_drop if col in df.columns])

        df_train, df_vt = train_test_split(
            df_clean, train_size=0.70, test_size=0.30, random_state=333
        )

        df_valid, df_test = train_test_split(
            df_vt, train_size=0.5, test_size=0.5, random_state=333
        )

        X_train = df_train.drop(columns=[target_col])
        y_train = df_train[target_col]

        X_valid = df_valid.drop(columns=[target_col])
        y_valid = df_valid[target_col]

        X_test = df_test.drop(columns=[target_col])
        y_test = df_test[target_col]

    print(f"Train shape: {X_train.shape}")
    print(f"Valid shape: {X_valid.shape}")
    print(f"Test shape: {X_test.shape}")

    return X_train, y_train, X_valid, y_valid, X_test, y_test


def get_feature_lists(X: pd.DataFrame, use_existing_splits: bool = False) -> Tuple[list, list]:
    """
    Identify numeric and categorical features from DataFrame

    Matches the preprocessing from data_split.py:
    - Numeric features (to scale): temp, atemp, hum, windspeed
    - Categorical features (to encode): season, mnth, weekday, weathersit, yr, hr, holiday, workingday

    Args:
        X: Input DataFrame
        use_existing_splits: If True, data is already preprocessed (no features lists needed)

    Returns:
        Tuple of (numeric_features, categorical_features)
    """
    if use_existing_splits:
        return [], []

    numeric_features = ['temp', 'atemp', 'hum', 'windspeed']
    categorical_features = ['season', 'mnth', 'weekday', 'weathersit', 'yr', 'hr', 'holiday', 'workingday']

    numeric_features = [col for col in numeric_features if col in X.columns]
    categorical_features = [col for col in categorical_features if col in X.columns]

    return numeric_features, categorical_features


def save_pipeline(pipeline: Pipeline, filepath: Union[str, Path]) -> None:
    """
    Save pipeline to pickle file

    Args:
        pipeline: Trained pipeline
        filepath: Path to save the pipeline
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, 'wb') as f:
        pickle.dump(pipeline, f)

    print(f"Pipeline saved to: {filepath}")


def load_pipeline(filepath: Union[str, Path]) -> Pipeline:
    """
    Load pipeline from pickle file

    Args:
        filepath: Path to the saved pipeline

    Returns:
        Loaded pipeline
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"Pipeline file not found: {filepath}")

    with open(filepath, 'rb') as f:
        pipeline = pickle.load(f)

    print(f"Pipeline loaded from: {filepath}")
    return pipeline


def train_model_with_pipeline(
    model_name: str,
    model_config: Dict[str, Any],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    mlflow_config: Dict[str, Any],
    use_existing_splits: bool = False
) -> Dict[str, Any]:
    """
    Train a model using scikit-learn pipeline with MLflow tracking

    Args:
        model_name: Name of the model
        model_config: Model configuration
        X_train, y_train: Training data
        X_valid, y_valid: Validation data
        X_test, y_test: Test data
        mlflow_config: MLflow configuration
        use_existing_splits: If True, data is already preprocessed (no preprocessing pipeline needed)

    Returns:
        Dictionary with results
    """
    print(f"\n{'='*80}")
    print(f"Training: {model_name}")
    print(f"{'='*80}")

    module = importlib.import_module(model_config['module'])
    model_class = getattr(module, model_config['class'])
    base_model = model_class()

    if use_existing_splits:
        print("Using preprocessed data - no preprocessing pipeline needed")
        base_pipeline = base_model
    else:
        numeric_features, categorical_features = get_feature_lists(X_train)
        print(f"Numeric features: {numeric_features}")
        print(f"Categorical features: {categorical_features}")

        base_pipeline = create_full_pipeline(
            model=base_model,
            numeric_features=numeric_features,
            categorical_features=categorical_features
        )

    with mlflow.start_run(run_name=f"{model_name}_pipeline", nested=True):
        mlflow.log_param("model_type", model_config['class'])
        mlflow.log_param("cv_folds", model_config['cv_folds'])
        mlflow.log_param("search_method", model_config.get('search_method', 'grid'))
        mlflow.log_dict(model_config['param_grid'], "param_grid.json")

        best_pipeline, best_params = train_pipeline_with_search(
            base_pipeline=base_pipeline,
            param_grid=model_config['param_grid'],
            X_train=X_train,
            y_train=y_train,
            cv_folds=model_config['cv_folds'],
            search_method=model_config.get('search_method', 'grid'),
            n_iter=model_config.get('n_iter', 10)
        )

        print(f"\nBest parameters: {best_params}")

        for param_name, param_value in best_params.items():
            mlflow.log_param(f"best_{param_name}", param_value)

        metrics_dict = evaluate_pipeline(
            pipeline=best_pipeline,
            X_train=X_train,
            y_train=y_train,
            X_valid=X_valid,
            y_valid=y_valid,
            X_test=X_test,
            y_test=y_test
        )

        for split_name, metrics in metrics_dict.items():
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(f"{split_name}_{metric_name}", metric_value)

        print(f"\nMetrics:")
        for split_name in ['train', 'valid', 'test']:
            m = metrics_dict[split_name]
            print(f"  {split_name.capitalize():5s} - RMSE: {m['rmse']:.4f}, MAE: {m['mae']:.4f}, R2: {m['r2']:.4f}")

        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)
        pipeline_path = models_dir / f"pipeline_{model_name}.pickle"
        save_pipeline(best_pipeline, pipeline_path)

        mlflow.sklearn.log_model(
            sk_model=best_pipeline,
            artifact_path="pipeline",
            registered_model_name=f"bike_sharing_pipeline_{model_name}"
        )

        mlflow.log_artifact(str(pipeline_path), "pickled_pipeline")

        run_id = mlflow.active_run().info.run_id

        return {
            'model_name': model_name,
            'run_id': run_id,
            'best_params': best_params,
            'train_metrics': metrics_dict['train'],
            'valid_metrics': metrics_dict['valid'],
            'test_metrics': metrics_dict['test'],
            'pipeline_path': str(pipeline_path)
        }


def train_all_models_with_pipeline(
    config_path: str = "config/models_config.yaml",
    use_existing_splits: bool = True
):
    """
    Train all enabled models using scikit-learn pipelines

    Args:
        config_path: Path to configuration file
        use_existing_splits: If True, use existing preprocessed data.
                            If False, load cleaned data and apply preprocessing in pipeline.

    Returns:
        Tuple of (results, comparison_df)
    """
    config = load_config(config_path)
    setup_mlflow(config)

    print(f"MLflow Tracking URI: {mlflow.get_tracking_uri()}")
    print(f"Experiment: {config['mlflow']['experiment_name']}")

    print("\nLoading data...")
    X_train, y_train, X_valid, y_valid, X_test, y_test = load_and_prepare_data(
        config, use_existing_splits=use_existing_splits
    )

    results = []

    with mlflow.start_run(run_name="pipeline_training"):
        mlflow.log_param("total_models", sum(1 for m in config['models'].values() if m.get('enabled', False)))
        mlflow.log_param("train_size", len(X_train))
        mlflow.log_param("valid_size", len(X_valid))
        mlflow.log_param("test_size", len(X_test))

        for model_name, model_config in config['models'].items():
            if not model_config.get('enabled', False):
                print(f"\nSkipping {model_name} (disabled in config)")
                continue

            try:
                result = train_model_with_pipeline(
                    model_name=model_name,
                    model_config=model_config,
                    X_train=X_train,
                    y_train=y_train,
                    X_valid=X_valid,
                    y_valid=y_valid,
                    X_test=X_test,
                    y_test=y_test,
                    mlflow_config=config['mlflow'],
                    use_existing_splits=use_existing_splits
                )
                results.append(result)

            except Exception as e:
                print(f"\nError training {model_name}: {str(e)}")
                traceback.print_exc()
                continue

    print(f"\n{'='*80}")
    print("TRAINING SUMMARY")
    print(f"{'='*80}")

    if results:
        comparison_data = []
        for r in results:
            comparison_data.append({
                'Model': r['model_name'],
                'Train_RMSE': r['train_metrics']['rmse'],
                'Valid_RMSE': r['valid_metrics']['rmse'],
                'Test_RMSE': r['test_metrics']['rmse'],
                'Train_MAE': r['train_metrics']['mae'],
                'Valid_MAE': r['valid_metrics']['mae'],
                'Test_MAE': r['test_metrics']['mae'],
                'Train_R2': r['train_metrics']['r2'],
                'Valid_R2': r['valid_metrics']['r2'],
                'Test_R2': r['test_metrics']['r2']
            })

        comparison_df = pd.DataFrame(comparison_data)
        print("\nModel Comparison (Test Set):")
        print(comparison_df[['Model', 'Test_RMSE', 'Test_MAE', 'Test_R2']].to_string(index=False))

        best_idx = comparison_df['Test_RMSE'].idxmin()
        best_model = comparison_df.loc[best_idx, 'Model']
        best_rmse = comparison_df.loc[best_idx, 'Test_RMSE']

        print(f"\nBest Model: {best_model} (Test RMSE: {best_rmse:.4f})")

        comparison_path = "models/pipeline_model_comparison.csv"
        comparison_df.to_csv(comparison_path, index=False)
        print(f"\nComparison saved to: {comparison_path}")

        return results, comparison_df
    else:
        print("\nNo models were successfully trained!")
        return [], None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train models with scikit-learn pipelines")
    parser.add_argument(
        "--config",
        default="config/models_config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--use-raw-data",
        action="store_true",
        help="Use raw cleaned data and apply preprocessing in pipeline (default: use existing preprocessed splits)"
    )

    args = parser.parse_args()

    print("="*80)
    print("Scikit-learn Pipeline Training")
    print("="*80)
    print(f"Config: {args.config}")
    print(f"Mode: {'Raw data with preprocessing pipeline' if args.use_raw_data else 'Existing preprocessed splits'}\n")

    use_existing_splits = not args.use_raw_data

    results, comparison = train_all_models_with_pipeline(
        config_path=args.config,
        use_existing_splits=use_existing_splits
    )

    print("\nTraining complete!")
    print(f"View results: mlflow ui --port 5000")
