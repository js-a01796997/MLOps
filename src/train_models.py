"""
MLflow-based Unified Model Training Script
Trains multiple scikit-learn models with experiment tracking and model registry
"""

import os
import random
import inspect
import importlib
import pickle
from pathlib import Path
from typing import Dict, Any, Tuple
import warnings
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from utils.config import load_config
from utils.mlflow_setup import setup_mlflow

# Fijar semilla global para reproducibilidad
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)

warnings.filterwarnings('ignore')


def load_data(config: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load train, validation, and test datasets"""
    train_df = pd.read_csv(config['data']['train'])
    valid_df = pd.read_csv(config['data']['valid'])
    test_df = pd.read_csv(config['data']['test'])

    return train_df, valid_df, test_df


def prepare_features_target(df: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, pd.Series]:
    """Split features and target"""
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calculate regression metrics"""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    # MAPE (Mean Absolute Percentage Error)
    # Avoid division by zero
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100 if mask.any() else np.inf

    return {
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'mape': mape
    }


def create_prediction_plots(y_train, y_train_pred, y_test, y_test_pred, model_name):
    """Create visualization plots for model predictions"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'{model_name} - Model Performance', fontsize=16)

    # Plot 1: Actual vs Predicted (Train)
    axes[0, 0].scatter(y_train, y_train_pred, alpha=0.5, s=10)
    axes[0, 0].plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', lw=2)
    axes[0, 0].set_xlabel('Actual')
    axes[0, 0].set_ylabel('Predicted')
    axes[0, 0].set_title('Train Set: Actual vs Predicted')
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Actual vs Predicted (Test)
    axes[0, 1].scatter(y_test, y_test_pred, alpha=0.5, s=10)
    axes[0, 1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    axes[0, 1].set_xlabel('Actual')
    axes[0, 1].set_ylabel('Predicted')
    axes[0, 1].set_title('Test Set: Actual vs Predicted')
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Residuals (Train)
    residuals_train = y_train - y_train_pred
    axes[1, 0].scatter(y_train_pred, residuals_train, alpha=0.5, s=10)
    axes[1, 0].axhline(y=0, color='r', linestyle='--', lw=2)
    axes[1, 0].set_xlabel('Predicted')
    axes[1, 0].set_ylabel('Residuals')
    axes[1, 0].set_title('Train Set: Residual Plot')
    axes[1, 0].grid(True, alpha=0.3)

    # Plot 4: Residuals (Test)
    residuals_test = y_test - y_test_pred
    axes[1, 1].scatter(y_test_pred, residuals_test, alpha=0.5, s=10)
    axes[1, 1].axhline(y=0, color='r', linestyle='--', lw=2)
    axes[1, 1].set_xlabel('Predicted')
    axes[1, 1].set_ylabel('Residuals')
    axes[1, 1].set_title('Test Set: Residual Plot')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def load_model_class(module_name: str, class_name: str):
    """Dynamically load model class from sklearn"""
    module = importlib.import_module(module_name)
    model_class = getattr(module, class_name)
    return model_class


def train_model(
    model_name: str,
    model_config: Dict[str, Any],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    mlflow_config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Train a single model with GridSearchCV and log to MLflow

    Returns:
        Dictionary with model results and metrics
    """
    print(f"\n{'='*80}")
    print(f"Training: {model_name}")
    print(f"{'='*80}")

    # Load model class
    model_class = load_model_class(model_config['module'], model_config['class'])

    # Create base model
    base_model = model_class()

    # Fijar random_state del modelo si el estimador lo soporta
    params = base_model.get_params()
    if "random_state" in params:
        base_model.set_params(random_state=SEED)

    # Start MLflow run
    with mlflow.start_run(run_name=model_name, nested=True):
        # Set tags for better organization
        mlflow.set_tag("model_type", model_config['class'])
        mlflow.set_tag("model_family", "regression")
        mlflow.set_tag("dataset", "bike_sharing")
        mlflow.set_tag("framework", model_config['module'].split('.')[0])
        mlflow.set_tag("search_method", model_config.get('search_method', 'grid'))

        # Set run description
        run_description = f"Training {model_config['class']} model for bike sharing demand prediction using {model_config.get('search_method', 'grid')} search with {model_config['cv_folds']}-fold cross-validation"
        mlflow.set_tag("mlflow.note.content", run_description)

        # Log configuration
        mlflow.log_param("model_type", model_config['class'])
        mlflow.log_param("cv_folds", model_config['cv_folds'])
        mlflow.log_param("module", model_config['module'])

        # Log dataset information
        mlflow.log_param("n_features", X_train.shape[1])
        mlflow.log_param("n_train_samples", X_train.shape[0])
        mlflow.log_param("n_valid_samples", X_valid.shape[0])
        mlflow.log_param("n_test_samples", X_test.shape[0])

        # Log feature names
        mlflow.log_param("target_variable", "cnt")
        mlflow.log_text("\n".join(X_train.columns.tolist()), "features.txt")

        mlflow.log_dict(model_config['param_grid'], "param_grid.json")

        # Handle missing values by imputing with median
        print("Checking for missing values...")
        nan_counts = X_train.isna().sum()
        total_nans = nan_counts.sum()

        if total_nans > 0:
            print(f"Found {total_nans} missing values. Imputing with median...")
            for col in X_train.columns:
                if X_train[col].isna().sum() > 0:
                    median_val = X_train[col].median()
                    X_train[col] = X_train[col].fillna(median_val)
                    X_valid[col] = X_valid[col].fillna(median_val)
                    X_test[col] = X_test[col].fillna(median_val)
                    print(f"  {col}: imputed {nan_counts[col]} values with {median_val}")

        # Choose search method
        search_method = model_config.get('search_method', 'grid').lower()

        if search_method == 'random':
            n_iter = model_config.get('n_iter', 10)
            search = RandomizedSearchCV(
                estimator=base_model,
                param_distributions=model_config['param_grid'],
                n_iter=n_iter,
                cv=model_config['cv_folds'],
                scoring='neg_mean_squared_error',
                n_jobs=-1,
                verbose=0,
                random_state=SEED
            )
            print(f"Running RandomizedSearchCV with {n_iter} iterations and {model_config['cv_folds']}-fold CV...")
            mlflow.log_param("search_method", "random")
            mlflow.log_param("n_iter", n_iter)
        else:
            search = GridSearchCV(
                estimator=base_model,
                param_grid=model_config['param_grid'],
                cv=model_config['cv_folds'],
                scoring='neg_mean_squared_error',
                n_jobs=-1,
                verbose=0
            )
            print(f"Running GridSearchCV with {model_config['cv_folds']}-fold CV...")
            mlflow.log_param("search_method", "grid")

        search.fit(X_train, y_train)

        # Best model
        best_model = search.best_estimator_
        best_params = search.best_params_
        best_score = search.best_score_

        print(f"\nBest parameters: {best_params}")
        print(f"Best CV score (neg_MSE): {best_score:.4f}")
        print(f"Best CV RMSE: {np.sqrt(-best_score):.4f}")

        # Log search info
        mlflow.log_metric("cv_best_rmse", np.sqrt(-best_score))
        mlflow.log_metric("cv_best_score", best_score)

        # Log best parameters found by search
        for param_name, param_value in best_params.items():
            mlflow.log_param(param_name, param_value)

        # Log all final model hyperparameters (including defaults)
        all_params = best_model.get_params()
        for param_name, param_value in all_params.items():
            if param_name not in best_params:
                # Only log simple types to avoid clutter
                if isinstance(param_value, (int, float, str, bool, type(None))):
                    mlflow.log_param(f"default_{param_name}", param_value)

        # Predictions on all sets
        y_train_pred = best_model.predict(X_train)
        y_valid_pred = best_model.predict(X_valid)
        y_test_pred = best_model.predict(X_test)

        # Calculate metrics
        train_metrics = calculate_metrics(y_train, y_train_pred)
        valid_metrics = calculate_metrics(y_valid, y_valid_pred)
        test_metrics = calculate_metrics(y_test, y_test_pred)

        # Log metrics
        for metric_name in ['rmse', 'mae', 'r2', 'mape']:
            mlflow.log_metric(f"train_{metric_name}", train_metrics[metric_name])
            mlflow.log_metric(f"valid_{metric_name}", valid_metrics[metric_name])
            mlflow.log_metric(f"test_{metric_name}", test_metrics[metric_name])

        # Print metrics
        print(f"\nMetrics:")
        print(f"  Train - RMSE: {train_metrics['rmse']:.4f}, MAE: {train_metrics['mae']:.4f}, R2: {train_metrics['r2']:.4f}")
        print(f"  Valid - RMSE: {valid_metrics['rmse']:.4f}, MAE: {valid_metrics['mae']:.4f}, R2: {valid_metrics['r2']:.4f}")
        print(f"  Test  - RMSE: {test_metrics['rmse']:.4f}, MAE: {test_metrics['mae']:.4f}, R2: {test_metrics['r2']:.4f}")

        # Create and log prediction plots
        try:
            fig = create_prediction_plots(y_train, y_train_pred, y_test, y_test_pred, model_name)
            plot_path = f"plots/{model_name}_predictions.png"
            Path("plots").mkdir(exist_ok=True)
            fig.savefig(plot_path, dpi=100, bbox_inches='tight')
            mlflow.log_artifact(plot_path, "plots")
            plt.close(fig)
        except Exception as e:
            print(f"Warning: Could not create plots: {e}")

        # Log feature importance if available
        if hasattr(best_model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': X_train.columns,
                'importance': best_model.feature_importances_
            }).sort_values('importance', ascending=False)

            # Log top 10 features as parameters
            for idx, row in feature_importance.head(10).iterrows():
                mlflow.log_param(f"top_feature_{idx+1}", f"{row['feature']} ({row['importance']:.4f})")

            # Save feature importance as artifact
            importance_path = "feature_importance.csv"
            feature_importance.to_csv(importance_path, index=False)
            mlflow.log_artifact(importance_path)
            print(f"\nTop 5 important features:")
            print(feature_importance.head(5).to_string(index=False))

        # Log model with MLflow
        model_info = mlflow.sklearn.log_model(
            sk_model=best_model,
            artifact_path="model",
            registered_model_name=f"bike_sharing_{model_name}"
        )

        # Add description and tags to the registered model version
        from mlflow.tracking import MlflowClient
        client = MlflowClient()

        # Get the registered model version
        model_name_full = f"bike_sharing_{model_name}"
        try:
            # Update model description
            model_description = f"{model_config['class']} model for bike sharing demand prediction. Trained with {model_config.get('search_method', 'grid')} search. Test RMSE: {test_metrics['rmse']:.4f}, R2: {test_metrics['r2']:.4f}"
            client.update_registered_model(
                name=model_name_full,
                description=model_description
            )

            # Get latest version number
            latest_versions = client.get_latest_versions(model_name_full, stages=["None"])
            if latest_versions:
                version = latest_versions[0].version

                # Set tags on the model version
                client.set_model_version_tag(model_name_full, version, "model_type", model_config['class'])
                client.set_model_version_tag(model_name_full, version, "test_rmse", f"{test_metrics['rmse']:.4f}")
                client.set_model_version_tag(model_name_full, version, "test_r2", f"{test_metrics['r2']:.4f}")
                client.set_model_version_tag(model_name_full, version, "framework", model_config['module'])
        except Exception as e:
            print(f"Warning: Could not update model metadata: {e}")

        # Save model locally (for compatibility with existing workflow)
        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)
        model_path = models_dir / f"modelo_{model_name}.pickle"

        with open(model_path, 'wb') as f:
            pickle.dump(best_model, f)

        mlflow.log_artifact(str(model_path), "pickled_model")
        print(f"\nModel saved to: {model_path}")

        # Get run ID for later reference
        run_id = mlflow.active_run().info.run_id

        return {
            'model_name': model_name,
            'run_id': run_id,
            'best_params': best_params,
            'train_metrics': train_metrics,
            'valid_metrics': valid_metrics,
            'test_metrics': test_metrics,
            'model_path': str(model_path)
        }


def train_all_models(config_path: str = "config/models_config.yaml"):
    """
    Main function to train all enabled models
    """
    # Load configuration
    config = load_config(config_path)

    # Setup MLflow using centralized utility
    setup_mlflow(config)

    print(f"MLflow Tracking URI: {mlflow.get_tracking_uri()}")
    print(f"Experiment: {config['mlflow']['experiment_name']}")

    # Load data
    print("\nLoading data...")
    train_df, valid_df, test_df = load_data(config)

    target_col = config['data']['target_column']
    X_train, y_train = prepare_features_target(train_df, target_col)
    X_valid, y_valid = prepare_features_target(valid_df, target_col)
    X_test, y_test = prepare_features_target(test_df, target_col)

    print(f"Train shape: {X_train.shape}")
    print(f"Valid shape: {X_valid.shape}")
    print(f"Test shape: {X_test.shape}")

    # Train models
    results = []

    # Parent run for all models
    with mlflow.start_run(run_name="training_pipeline"):
        mlflow.log_param("total_models", sum(1 for m in config['models'].values() if m.get('enabled', False)))
        mlflow.log_param("train_size", len(X_train))
        mlflow.log_param("valid_size", len(X_valid))
        mlflow.log_param("test_size", len(X_test))

        for model_name, model_config in config['models'].items():
            if not model_config.get('enabled', False):
                print(f"\nSkipping {model_name} (disabled in config)")
                continue

            try:
                result = train_model(
                    model_name=model_name,
                    model_config=model_config,
                    X_train=X_train,
                    y_train=y_train,
                    X_valid=X_valid,
                    y_valid=y_valid,
                    X_test=X_test,
                    y_test=y_test,
                    mlflow_config=config['mlflow']
                )
                results.append(result)

            except Exception as e:
                print(f"\nError training {model_name}: {str(e)}")
                import traceback
                traceback.print_exc()
                continue

    # Summary
    print(f"\n{'='*80}")
    print("TRAINING SUMMARY")
    print(f"{'='*80}")

    if results:
        # Create comparison DataFrame
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

        # Best model by test RMSE
        best_idx = comparison_df['Test_RMSE'].idxmin()
        best_model = comparison_df.loc[best_idx, 'Model']
        best_rmse = comparison_df.loc[best_idx, 'Test_RMSE']

        print(f"\nBest Model: {best_model} (Test RMSE: {best_rmse:.4f})")

        # Save comparison
        comparison_path = "models/model_comparison.csv"
        comparison_df.to_csv(comparison_path, index=False)
        print(f"\nComparison saved to: {comparison_path}")

        return results, comparison_df
    else:
        print("\nNo models were successfully trained!")
        return [], None


if __name__ == "__main__":
    import sys

    # Allow custom config path
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config/models_config.yaml"

    print("="*80)
    print("MLflow Training Pipeline")
    print("="*80)
    print(f"Config: {config_path}\n")

    results, comparison = train_all_models(config_path)

    print("\nTraining complete!")
    print(f"View results: mlflow ui --port 5000")
