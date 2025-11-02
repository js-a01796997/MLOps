"""
Complete MLOps Pipeline Orchestrator
Runs the full pipeline: data cleaning -> splitting -> training -> evaluation -> registry
"""

import os
import sys
import subprocess
import argparse
import logging
from pathlib import Path
import yaml


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def print_step(step_name: str, step_num: int, total_steps: int):
    """Print a formatted step header"""
    print("\n" + "="*80)
    print(f"STEP {step_num}/{total_steps}: {step_name}")
    print("="*80 + "\n")


def run_command(command: list, description: str):
    """Run a shell command and handle errors"""
    logger.info(f"Running: {description}")
    logger.info(f"Command: {' '.join(command)}")

    result = subprocess.run(command, capture_output=False, text=True)

    if result.returncode != 0:
        logger.error(f"Error in step: {description}")
        sys.exit(1)

    logger.info(f"{description} completed successfully")
    return result


def check_file_exists(file_path: str, description: str):
    """Check if a file exists"""
    if not Path(file_path).exists():
        logger.error(f"Required file not found: {file_path} ({description})")
        return False
    logger.info(f"Found: {file_path}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Run the complete MLOps pipeline")
    parser.add_argument("--config", default="config/models_config.yaml", help="Configuration file")
    parser.add_argument("--skip-cleaning", action="store_true", help="Skip data cleaning step")
    parser.add_argument("--skip-splitting", action="store_true", help="Skip data splitting step")
    parser.add_argument("--skip-training", action="store_true", help="Skip model training step")
    parser.add_argument("--skip-registry", action="store_true", help="Skip model registry step")
    parser.add_argument("--skip-xgboost", action="store_true", help="Skip XGBoost training and comparison")
    parser.add_argument("--models", nargs="+", help="Specific models to train")
    
    args = parser.parse_args()

    print("="*80)
    print("MLOps PIPELINE - BIKE SHARING DEMAND PREDICTION")
    print("="*80)

    if not check_file_exists(args.config, "Configuration file"):
        sys.exit(1)

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    total_steps = 4 - sum([args.skip_cleaning, args.skip_splitting, args.skip_training, args.skip_registry])
    current_step = 0

    # Step 1: Data Cleaning
    if not args.skip_cleaning:
        current_step += 1
        print_step("Data Cleaning", current_step, total_steps)

        if not check_file_exists("data/raw/bike_sharing_modified.csv", "Raw modified data"):
            logger.warning("Using original data instead")
            if not check_file_exists("data/raw/bike_sharing_original.csv", "Raw original data"):
                logger.error("No raw data found. Please add data to data/raw/")
                sys.exit(1)

        run_command(
            ["python", "src/data_cleaning.py"],
            "Data cleaning"
        )

        if not check_file_exists("data/processed/bike_sharing_cleaned.csv", "Cleaned data"):
            sys.exit(1)

    # Step 2: Data Splitting
    if not args.skip_splitting:
        current_step += 1
        print_step("Data Splitting & Preprocessing", current_step, total_steps)

        if not check_file_exists("data/processed/bike_sharing_cleaned.csv", "Cleaned data"):
            logger.error("Cleaned data not found. Run data cleaning first.")
            sys.exit(1)

        run_command(
            ["python", "src/data_split.py"],
            "Data splitting and preprocessing"
        )

        for split in ["train.csv", "valid.csv", "test.csv"]:
            if not check_file_exists(f"data/processed/{split}", f"{split.split('.')[0]} set"):
                sys.exit(1)

    # Step 3: Model Training
    if not args.skip_training:
        current_step += 1
        print_step("Model Training with MLflow", current_step, total_steps)

        for split in ["train.csv", "valid.csv", "test.csv"]:
            if not check_file_exists(f"data/processed/{split}", f"{split.split('.')[0]} set"):
                logger.error("Data splits not found. Run data splitting first.")
                sys.exit(1)

        # Train main models
        if args.models:
            logger.info(f"Training only: {', '.join(args.models)}")
            temp_config = config.copy()
            for model_name in temp_config['models']:
                if model_name in args.models:
                    temp_config['models'][model_name]['enabled'] = True
                else:
                    temp_config['models'][model_name]['enabled'] = False

            temp_config_path = "config/temp_models_config.yaml"
            with open(temp_config_path, 'w') as f:
                yaml.dump(temp_config, f)

            run_command(
                ["python", "src/train_models.py", temp_config_path],
                "Model training"
            )

            os.remove(temp_config_path)
        else:
            run_command(
                ["python", "src/train_models.py", args.config],
                "Model training"
            )

        # Train XGBoost if requested
        if not args.skip_xgboost:
            print("\n" + "-"*40)
            print("Training XGBoost Model")
            print("-"*40 + "\n")
            
            run_command(
                ["python", "src/xgbregressor.py"],
                "XGBoost training"
            )

    # Step 4: Model Registry
    if not args.skip_registry:
        current_step += 1
        print_step("Model Registry & Promotion", current_step, total_steps)

        # Include XGBoost in comparison by default unless skipped
        if not args.skip_xgboost and Path("models/modelo_xgboost.pickle").exists():
            run_command(
                ["python", "src/model_registry.py", "--config", args.config, "--compare", "--include-xgboost"],
                "Model comparison and registry (including XGBoost)"
            )
        else:
            run_command(
                ["python", "src/model_registry.py", "--config", args.config, "--compare"],
                "Model comparison and registry"
            )

        print("\nWould you like to auto-promote the best model to Staging?")
        response = input("Enter 'yes' to auto-promote: ").strip().lower()

        if response == 'yes':
            cmd = ["python", "src/model_registry.py", "--config", args.config, "--auto-promote"]
            if not args.skip_xgboost:
                cmd.append("--include-xgboost")

            run_command(cmd, "Auto-promotion to Staging")


    print("\n" + "="*80)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("="*80)

    print("\nNext steps:")
    print("  1. View experiment tracking: mlflow ui --port 5000")
    print("  2. Check model comparison: cat models/model_comparison.csv")
    print("  3. List registered models: python src/model_registry.py --list")
    print("  4. Promote model to production: python src/model_registry.py --promote <model_name> --stage Production")

    print("\n" + "="*80)


if __name__ == "__main__":
    main()
