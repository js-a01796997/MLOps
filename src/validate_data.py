import pandas as pd
from pathlib import Path
import sys

def validate_cleaned_data():
    cleaned_path = Path('data/processed/bike_sharing_cleaned.csv')
    if not cleaned_path.exists():
        print(f"Error: {cleaned_path} not found")
        return False

    df = pd.read_csv(cleaned_path)
    print(f"Cleaned data shape: {df.shape}")

    if df.shape[0] == 0:
        print("Error: Cleaned data is empty")
        return False

    if df.shape[1] == 0:
        print("Error: Cleaned data has no columns")
        return False

    print("Cleaned data validation passed")
    return True

def validate_split_data():
    files = ['train.csv', 'test.csv', 'valid.csv']

    for file in files:
        file_path = Path(f'data/processed/{file}')
        if not file_path.exists():
            print(f"Error: {file} not found")
            return False

        df = pd.read_csv(file_path)
        print(f"{file} shape: {df.shape}")

        if df.shape[0] == 0:
            print(f"Error: {file} is empty")
            return False

        if df.shape[1] == 0:
            print(f"Error: {file} has no columns")
            return False

    print("Split data validation passed")
    return True

if __name__ == "__main__":
    success = True

    if not validate_cleaned_data():
        success = False

    if not validate_split_data():
        success = False

    if success:
        print("All data integrity checks passed")
        sys.exit(0)
    else:
        print("Data validation failed")
        sys.exit(1)
