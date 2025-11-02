"""
MLflow Model Registry Management
Promotes models through stages: None -> Staging -> Production
Includes comparison with local XGBoost model
"""

import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd
from typing import Optional, Dict, List
from pathlib import Path
import pickle

from utils.config import load_config
from utils.mlflow_setup import setup_mlflow

def load_xgboost_metrics() -> Optional[Dict]:
    """Load XGBoost model and compute metrics from pickle file"""
    xgb_path = Path("models/modelo_xgboost.pickle")
    if not xgb_path.exists():
        print(f"XGBoost model not found at: {xgb_path}")
        return None

    try:
        with open(xgb_path, "rb") as f:
            model_data = pickle.load(f)

        metrics_dict = model_data.get("metrics", {})

        metrics = {
            "run_name": "XGBoost",
            "run_id": "local_xgboost",
            # Busca tanto test_* como métricas antiguas
            "test_rmse": (
                metrics_dict.get("test_rmse")
                or metrics_dict.get("rmse")
                or float("nan")
            ),
            "test_mae": (
                metrics_dict.get("test_mae")
                or metrics_dict.get("mae")
                or float("nan")
            ),
            "test_r2": (
                metrics_dict.get("test_r2")
                or metrics_dict.get("r2")
                or float("nan")
            ),
            "valid_rmse": (
                metrics_dict.get("valid_rmse")
                or float("inf")
            ),
        }

        # Si las métricas están en formato numérico correcto
        print(f"Loaded XGBoost metrics: {metrics}")
        return metrics

    except Exception as e:
        print(f"Error loading XGBoost model: {e}")
        return None


def compare_models_for_promotion(experiment_name: str, top_n: int = 5, include_xgboost: bool = True):
    """
    Compare top models including XGBoost to help decide which to promote

    Args:
        experiment_name: Name of the experiment
        top_n: Number of top models to show
        include_xgboost: Whether to include XGBoost in comparison
    """
    client = MlflowClient()

    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        print(f"Experiment '{experiment_name}' not found!")
        return

    # Get all runs sorted by test RMSE
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="",
        order_by=["metrics.test_rmse ASC"],
        max_results=top_n
    )

    print("\n" + "="*80)
    print(f"TOP MODELS COMPARISON")
    print("="*80)

    comparison_data = []
    
    # Add MLflow tracked models
    for run in runs:
        comparison_data.append({
            'Model Type': 'MLflow',
            'Run Name': run.data.tags.get('mlflow.runName', 'unknown'),
            'Run ID': run.info.run_id[:8],
            'Test RMSE': run.data.metrics.get('test_rmse', float('inf')),
            'Test MAE': run.data.metrics.get('test_mae', float('inf')),
            'Test R2': run.data.metrics.get('test_r2', float('-inf')),
            'Valid RMSE': run.data.metrics.get('valid_rmse', float('inf'))
        })

    # Add XGBoost if requested
    if include_xgboost:
        xgb_metrics = load_xgboost_metrics()
        if xgb_metrics:
            comparison_data.append({
                'Model Type': 'XGBoost',
                'Run Name': xgb_metrics['run_name'],
                'Run ID': xgb_metrics['run_id'],
                'Test RMSE': xgb_metrics['test_rmse'],
                'Test MAE': xgb_metrics['test_mae'],
                'Test R2': xgb_metrics['test_r2'],
                'Valid RMSE': xgb_metrics['valid_rmse']
            })

    # Create DataFrame and sort by Test RMSE
    df = pd.DataFrame(comparison_data)
    df = df.sort_values('Test RMSE')
    print(df.to_string(index=False))

    return df

def auto_promote_best_model(
    experiment_name: str,
    metric: str = "test_rmse",
    thresholds: Optional[Dict[str, float]] = None,
    include_xgboost: bool = True
):
    """
    Automatically promote the best model if it meets thresholds

    Args:
        experiment_name: Name of the experiment
        metric: Metric to optimize
        thresholds: Dictionary of metric thresholds
        include_xgboost: Whether to include XGBoost in comparison
    """
    print("\n" + "="*80)
    print("AUTO PROMOTION (INCLUDING XGBOOST)")
    print("="*80)

    # Compare all models including XGBoost
    df = compare_models_for_promotion(experiment_name, include_xgboost=include_xgboost)
    
    if df.empty:
        print("No models found to compare!")
        return

    # Get best model info
    best_model = df.iloc[0]
    
    print(f"\nBest model: {best_model['Run Name']} ({best_model['Model Type']})")
    print(f"  Run ID: {best_model['Run ID']}")
    print(f"  Test RMSE: {best_model['Test RMSE']:.4f}")
    print(f"  Test MAE: {best_model['Test MAE']:.4f}")
    print(f"  Test R2: {best_model['Test R2']:.4f}")

    # If best model is XGBoost, register it with MLflow
    if best_model['Model Type'] == 'XGBoost':
        print("\nBest model is XGBoost - registering with MLflow...")
        try:
            with open("models/modelo_xgboost.pickle", "rb") as f:
                model_data = pickle.load(f)
            
            # Register XGBoost model with MLflow
            mlflow.xgboost.log_model(
                model_data['model'],
                "xgboost_model",
                registered_model_name="XGBoostRegressor"
            )
            print("XGBoost model registered successfully")
            
        except Exception as e:
            print(f"Error registering XGBoost model: {e}")
            return

    # Continue with normal promotion logic
    # ... rest of existing auto_promote_best_model code ...

def main():
    """Main function with CLI interface"""
    import argparse

    parser = argparse.ArgumentParser(description="MLflow Model Registry Management")
    parser.add_argument("--config", default="config/models_config.yaml", help="Config file path")
    parser.add_argument("--list", action="store_true", help="List all registered models")
    parser.add_argument("--compare", action="store_true", help="Compare top models")
    parser.add_argument("--top-n", type=int, default=5, help="Number of top models to show")
    parser.add_argument("--auto-promote", action="store_true", help="Auto-promote best model")
    parser.add_argument("--promote", type=str, help="Model name to promote")
    parser.add_argument("--version", type=int, help="Model version to promote")
    parser.add_argument("--stage", default="Staging", choices=["Staging", "Production"], help="Target stage")
    parser.add_argument("--include-xgboost", action="store_true", help="Include XGBoost in comparison")

    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    setup_mlflow(config)

    experiment_name = config['mlflow']['experiment_name']

    if args.list:
        list_registered_models()

    elif args.compare:
        compare_models_for_promotion(experiment_name, args.top_n, args.include_xgboost)

    elif args.auto_promote:
        thresholds = config.get('registry', {}).get('staging_threshold')
        auto_promote_best_model(experiment_name, thresholds=thresholds, include_xgboost=args.include_xgboost)

    elif args.promote:
        promote_model(args.promote, version=args.version, stage=args.stage)

    else:
        # Default: show comparison
        compare_models_for_promotion(experiment_name, args.top_n, args.include_xgboost)
        print("\nUse --help to see all options")

if __name__ == "__main__":
    main()