#!/usr/bin/env python3
"""
Test script to verify MLflow remote connection.
Run this to check if MLflow is properly configured before training models.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from utils.config import load_config
from utils.mlflow_setup import setup_mlflow
import mlflow


def test_mlflow_connection():
    """Test MLflow connection and experiment setup"""
    print("=" * 60)
    print("Testing MLflow Connection")
    print("=" * 60)
    
    try:
        # Load configuration
        config = load_config("config/models_config.yaml")
        
        # Setup MLflow
        setup_mlflow(config)
        
        # Verify connection
        tracking_uri = mlflow.get_tracking_uri()
        print(f"\n✓ Tracking URI: {tracking_uri}")
        
        # Try to get current experiment
        experiment_name = config['mlflow']['experiment_name']
        experiment = mlflow.get_experiment_by_name(experiment_name)
        
        if experiment:
            print(f"✓ Experiment '{experiment_name}' found (ID: {experiment.experiment_id})")
        else:
            print(f"⚠ Experiment '{experiment_name}' not found, but this is OK - it will be created on first run")
        
        # Try to create a test run
        print("\n" + "-" * 60)
        print("Creating test run...")
        with mlflow.start_run(run_name="connection_test"):
            mlflow.log_param("test_param", "test_value")
            mlflow.log_metric("test_metric", 1.0)
            run_id = mlflow.active_run().info.run_id
            print(f"✓ Successfully created test run: {run_id}")
            print(f"✓ View it at: {tracking_uri.rstrip('/')}/#/experiments/{experiment.experiment_id}/runs/{run_id}")
        
        print("\n" + "=" * 60)
        print("✓ MLflow connection test PASSED!")
        print("=" * 60)
        return True
        
    except Exception as e:
        print("\n" + "=" * 60)
        print("✗ MLflow connection test FAILED!")
        print("=" * 60)
        print(f"Error: {e}")
        print("\nTroubleshooting:")
        print("1. Check if the MLflow server is accessible:")
        print(f"   curl {config.get('mlflow', {}).get('tracking_uri', 'N/A')}")
        print("2. If authentication is required, set environment variables:")
        print("   export MLFLOW_TRACKING_USERNAME=your_username")
        print("   export MLFLOW_TRACKING_PASSWORD=your_password")
        print("3. Check your network connection")
        return False


if __name__ == "__main__":
    success = test_mlflow_connection()
    sys.exit(0 if success else 1)

