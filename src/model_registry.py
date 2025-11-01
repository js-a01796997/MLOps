"""
MLflow Model Registry Management
Promotes models through stages: None -> Staging -> Production
"""

import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd
from typing import Optional, Dict, List

from utils.config import load_config
from utils.mlflow_setup import setup_mlflow


def get_best_run(experiment_name: str, metric: str = "test_rmse") -> Optional[Dict]:
    """
    Find the best run based on a metric

    Args:
        experiment_name: Name of the MLflow experiment
        metric: Metric to optimize (lower is better for RMSE/MAE)

    Returns:
        Dictionary with run information
    """
    client = MlflowClient()

    # Get experiment
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        print(f"Experiment '{experiment_name}' not found!")
        return None

    # Search runs
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=[f"metrics.{metric} ASC"],  # Lower is better
        max_results=1
    )

    if not runs:
        print("No runs found!")
        return None

    best_run = runs[0]

    return {
        'run_id': best_run.info.run_id,
        'run_name': best_run.data.tags.get('mlflow.runName', 'unknown'),
        'metrics': best_run.data.metrics,
        'params': best_run.data.params
    }


def list_registered_models():
    """List all registered models with their versions"""
    client = MlflowClient()

    print("\n" + "="*80)
    print("REGISTERED MODELS")
    print("="*80)

    try:
        models = client.search_registered_models()

        if not models:
            print("No registered models found.")
            return

        for model in models:
            print(f"\nModel: {model.name}")
            print(f"  Latest versions:")

            # Get all versions
            versions = client.search_model_versions(f"name='{model.name}'")

            # Group by stage
            stages = {}
            for version in versions:
                stage = version.current_stage
                if stage not in stages:
                    stages[stage] = []
                stages[stage].append(version)

            # Print by stage
            for stage in ['Production', 'Staging', 'Archived', 'None']:
                if stage in stages:
                    for v in stages[stage]:
                        print(f"    Version {v.version} - {stage}")
                        print(f"      Run ID: {v.run_id}")
                        if v.description:
                            print(f"      Description: {v.description}")

    except Exception as e:
        print(f"Error listing models: {str(e)}")


def promote_model(
    model_name: str,
    version: Optional[int] = None,
    stage: str = "Staging",
    archive_existing: bool = True
):
    """
    Promote a model version to a stage

    Args:
        model_name: Registered model name
        version: Model version (if None, uses latest)
        stage: Target stage ('Staging' or 'Production')
        archive_existing: Whether to archive existing models in that stage
    """
    client = MlflowClient()

    print(f"\nPromoting model: {model_name}")

    try:
        # Get model versions
        if version is None:
            versions = client.search_model_versions(f"name='{model_name}'")
            if not versions:
                print(f"No versions found for model '{model_name}'")
                return
            # Sort by version number and get latest
            version = max([int(v.version) for v in versions])

        print(f"  Version: {version}")
        print(f"  Target stage: {stage}")

        # Archive existing models in target stage
        if archive_existing:
            existing = client.get_latest_versions(model_name, stages=[stage])
            for model_version in existing:
                print(f"  Archiving existing version {model_version.version} in {stage}")
                client.transition_model_version_stage(
                    name=model_name,
                    version=model_version.version,
                    stage="Archived"
                )

        # Transition to new stage
        client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage=stage
        )

        print(f"Successfully promoted {model_name} v{version} to {stage}")

    except Exception as e:
        print(f"Error promoting model: {str(e)}")


def compare_models_for_promotion(experiment_name: str, top_n: int = 5):
    """
    Compare top models to help decide which to promote

    Args:
        experiment_name: Name of the experiment
        top_n: Number of top models to show
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
    print(f"TOP {len(runs)} MODELS BY TEST RMSE")
    print("="*80)

    comparison_data = []
    for run in runs:
        comparison_data.append({
            'Run Name': run.data.tags.get('mlflow.runName', 'unknown'),
            'Run ID': run.info.run_id[:8],
            'Test RMSE': run.data.metrics.get('test_rmse', float('inf')),
            'Test MAE': run.data.metrics.get('test_mae', float('inf')),
            'Test R2': run.data.metrics.get('test_r2', float('-inf')),
            'Valid RMSE': run.data.metrics.get('valid_rmse', float('inf'))
        })

    df = pd.DataFrame(comparison_data)
    print(df.to_string(index=False))

    return df


def auto_promote_best_model(
    experiment_name: str,
    metric: str = "test_rmse",
    thresholds: Optional[Dict[str, float]] = None
):
    """
    Automatically promote the best model if it meets thresholds

    Args:
        experiment_name: Name of the experiment
        metric: Metric to optimize
        thresholds: Dictionary of metric thresholds (e.g., {'rmse': 100, 'r2': 0.5})
    """
    print("\n" + "="*80)
    print("AUTO PROMOTION")
    print("="*80)

    best_run = get_best_run(experiment_name, metric)

    if best_run is None:
        return

    print(f"\nBest run: {best_run['run_name']}")
    print(f"  Run ID: {best_run['run_id']}")
    print(f"  Test RMSE: {best_run['metrics'].get('test_rmse', 'N/A'):.4f}")
    print(f"  Test MAE: {best_run['metrics'].get('test_mae', 'N/A'):.4f}")
    print(f"  Test R2: {best_run['metrics'].get('test_r2', 'N/A'):.4f}")

    # Check thresholds
    if thresholds:
        meets_criteria = True
        print("\nChecking thresholds:")

        if 'rmse' in thresholds:
            test_rmse = best_run['metrics'].get('test_rmse', float('inf'))
            meets = test_rmse < thresholds['rmse']
            print(f"  RMSE < {thresholds['rmse']}: {meets} ({test_rmse:.4f})")
            meets_criteria = meets_criteria and meets

        if 'r2' in thresholds:
            test_r2 = best_run['metrics'].get('test_r2', float('-inf'))
            meets = test_r2 > thresholds['r2']
            print(f"  R2 > {thresholds['r2']}: {meets} ({test_r2:.4f})")
            meets_criteria = meets_criteria and meets

        if not meets_criteria:
            print("\nModel does not meet thresholds. Not promoting.")
            return

    # Find registered model name for this run
    client = MlflowClient()
    run = client.get_run(best_run['run_id'])

    # Get model artifacts
    artifacts = client.list_artifacts(best_run['run_id'], path="model")
    if not artifacts:
        print("\nNo model artifacts found for this run")
        return

    # Search for registered models with this run_id
    all_models = client.search_registered_models()
    model_name = None

    for model in all_models:
        versions = client.search_model_versions(f"name='{model.name}'")
        for version in versions:
            if version.run_id == best_run['run_id']:
                model_name = model.name
                model_version = version.version
                break
        if model_name:
            break

    if model_name:
        print(f"\nPromoting {model_name} version {model_version} to Staging...")
        promote_model(model_name, version=int(model_version), stage="Staging")
    else:
        print("\nModel not found in registry. Make sure auto_register is enabled.")


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

    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    setup_mlflow(config)

    experiment_name = config['mlflow']['experiment_name']

    if args.list:
        list_registered_models()

    elif args.compare:
        compare_models_for_promotion(experiment_name, args.top_n)

    elif args.auto_promote:
        thresholds = config.get('registry', {}).get('staging_threshold')
        auto_promote_best_model(experiment_name, thresholds=thresholds)

    elif args.promote:
        promote_model(args.promote, version=args.version, stage=args.stage)

    else:
        # Default: show comparison
        compare_models_for_promotion(experiment_name, args.top_n)
        print("\nUse --help to see all options")


if __name__ == "__main__":
    main()
