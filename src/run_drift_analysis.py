import os
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp, chi2_contingency
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# =====================================================
# LOAD MODEL
# =====================================================
def load_model(model_path="models/modelo_xgboost.pickle"):
    print("📌 Loading model…")
    with open(model_path, "rb") as f:
        payload = pickle.load(f)

    pipeline = payload["pipeline"]
    feature_cols = payload["feature_columns"]

    print(f"✔ Model loaded. Feature count: {len(feature_cols)}")
    return pipeline, feature_cols


# =====================================================
# METRICS
# =====================================================
def compute_performance(model, X, y):
    preds = model.predict(X)
    rmse = float(np.sqrt(mean_squared_error(y, preds)))
    mae = float(mean_absolute_error(y, preds))
    r2 = float(r2_score(y, preds))
    return rmse, mae, r2


# =====================================================
# DRIFT TESTS
# =====================================================
def compute_drift(series_base, series_new):
    """Detect drift automatically according to data type."""
    try:
        # Numeric drift (Kolmogorov-Smirnov)
        if np.issubdtype(series_base.dtype, np.number):
            stat, p = ks_2samp(series_base.dropna(), series_new.dropna())
            return p < 0.05, p

        # Categorical drift (Chi²)
        base_vals = series_base.astype(str)
        drift_vals = series_new.astype(str)

        table = pd.crosstab(base_vals, drift_vals)
        chi2, p, _, _ = chi2_contingency(table)
        return p < 0.05, p

    except Exception:
        return False, 1.0  # if something goes wrong, assume no drift


# =====================================================
# PREPROCESS: DUMMIES TO MATCH MODEL FEATURES
# =====================================================
def align_features(df_raw, feature_cols):
    """Create dummies and insert missing feature columns."""
    df = df_raw.copy()

    df = pd.get_dummies(df, drop_first=True)

    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0

    df = df[feature_cols]
    return df


# =====================================================
# SAFE PLOTS
# =====================================================
def plot_feature(df_base, df_drift, column, output_folder):
    """Plots numericals as histograms and categoricals as barplots."""
    os.makedirs(output_folder, exist_ok=True)
    plt.figure(figsize=(6, 4))

    base_col = df_base[column]
    drift_col = df_drift[column]

    # NUMERIC FEATURE → HISTOGRAM
    if np.issubdtype(base_col.dtype, np.number):
        plt.hist(base_col, bins=30, alpha=0.5, label="Base")
        plt.hist(drift_col, bins=30, alpha=0.5, label="Drift")
        plt.title(f"Histogram: {column}")
        plt.legend()

    else:
        # CATEGORY → BARPLOT
        base_counts = base_col.astype(str).value_counts()
        drift_counts = drift_col.astype(str).value_counts()

        labels = sorted(list(set(base_counts.index) | set(drift_counts.index)))
        base_vals = [base_counts.get(lbl, 0) for lbl in labels]
        drift_vals = [drift_counts.get(lbl, 0) for lbl in labels]

        x = np.arange(len(labels))
        width = 0.4

        plt.bar(x - width / 2, base_vals, width, label="Base")
        plt.bar(x + width / 2, drift_vals, width, label="Drift")
        plt.xticks(x, labels, rotation=45)
        plt.title(f"Category Frequency: {column}")
        plt.legend()

    plt.tight_layout()
    plt.savefig(f"{output_folder}/{column}.png")
    plt.close()


# =====================================================
# MAIN
# =====================================================
def run():

    # -----------------------------------------------
    # LOAD MODEL
    # -----------------------------------------------
    model, feature_cols = load_model()

    # -----------------------------------------------
    # LOAD BASE DATASET
    # -----------------------------------------------
    base_path = "data/processed/dataset_clean.csv"
    df_base_raw = pd.read_csv(base_path)

    print("📌 Loading base dataset…")

    y_base = df_base_raw["cnt"]
    X_base = align_features(df_base_raw.drop(columns=["cnt"]), feature_cols)

    # Base performance
    base_rmse, base_mae, base_r2 = compute_performance(model, X_base, y_base)

    print(f"📊 BASE METRICS — RMSE={base_rmse:.4f}  MAE={base_mae:.4f}  R²={base_r2:.4f}\n")

    # -----------------------------------------------
    # DRIFT DATASETS
    # -----------------------------------------------
    drift_files = [
        "data/processed/dataset_season_drift.csv",
        "data/processed/dataset_wind_drift.csv",
        "data/processed/dataset_temp_drift.csv"
    ]

    summary = []

    for drift_file in drift_files:

        print(f"\n🔥 Checking drift for: {drift_file}")
        df_drift_raw = pd.read_csv(drift_file)

        # Metrics
        y_drift = df_drift_raw["cnt"]
        X_drift = align_features(df_drift_raw.drop(columns=["cnt"]), feature_cols)

        rmse, mae, r2 = compute_performance(model, X_drift, y_drift)
        print(f"  → RMSE={rmse:.4f} | MAE={mae:.4f} | R²={r2:.4f}")

        # Drift analysis per feature
        results = []
        for col in df_drift_raw.columns:
            if col == "cnt":
                continue

            if col not in df_base_raw.columns:
                continue

            drift_flag, p_value = compute_drift(df_base_raw[col], df_drift_raw[col])
            results.append([col, p_value, drift_flag])

            # Plots
            plot_folder = f"plots/{Path(drift_file).stem}"
            plot_feature(df_base_raw, df_drift_raw, col, plot_folder)

        df_results = pd.DataFrame(results, columns=["feature", "p_value", "drift_detected"])
        save_path = drift_file + "_drift_results.csv"
        df_results.to_csv(save_path, index=False)

        print(f"✔ Drift results saved: {save_path}")

        summary.append([drift_file, rmse, mae, r2])


    # -----------------------------------------------
    # SUMMARY TABLE
    # -----------------------------------------------
    df_summary = pd.DataFrame(summary, columns=["drift_file", "RMSE", "MAE", "R2"])
    df_summary.to_csv("drift_summary.csv", index=False)

    print("\n🎉 ANALYSIS COMPLETED — No errors.")
    print("Summary saved to drift_summary.csv")


if __name__ == "__main__":
    run()

    
