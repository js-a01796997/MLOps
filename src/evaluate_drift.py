import json
from pathlib import Path
import pandas as pd
import numpy as np
from joblib import load
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from evidently import Report
from evidently.presets import DataDriftPreset


SEED = 42
np.random.seed(SEED)

TARGET_COL = "cnt"  

def load_model():
    model_path = Path("models/modelo_xgboost.pickle")  
    model = load(model_path)
    return model, model_path

def compute_metrics(y_true, y_pred):
    """
    Compute RMSE, MAE and R2 with compatibility across scikit-learn versions.
    """
    try:
        # Prefer call with squared=False si está disponible
        rmse = float(mean_squared_error(y_true, y_pred, squared=False))
    except TypeError:
        # Fallback: calcular sqrt(MSE) si 'squared' no es aceptado
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return {"rmse": rmse, "mae": mae, "r2": r2}

def main():
    # 1. Cargar datasets
    ref_df = pd.read_csv("data/processed/valid.csv")
    cur_df = pd.read_csv("data/monitoring/drifted.csv")

    # 2. Separar X, y
    X_ref = ref_df.drop(columns=[TARGET_COL])
    y_ref = ref_df[TARGET_COL]

    X_cur = cur_df.drop(columns=[TARGET_COL])
    y_cur = cur_df[TARGET_COL]

    # 3. Cargar modelo
    model, model_path = load_model()

    # 4. Predicciones
    y_ref_pred = model.predict(X_ref)
    y_cur_pred = model.predict(X_cur)

    # 5. Métricas
    ref_metrics = compute_metrics(y_ref, y_ref_pred)
    cur_metrics = compute_metrics(y_cur, y_cur_pred)

    print("Métricas en referencia:", ref_metrics)
    print("Métricas en drifted:   ", cur_metrics)

    # 6. Calcular degradación relativa
    degradation = {
        "rmse_delta": cur_metrics["rmse"] - ref_metrics["rmse"],
        "rmse_pct_increase": (cur_metrics["rmse"] / ref_metrics["rmse"] - 1) * 100
                             if ref_metrics["rmse"] > 0 else np.nan
    }

    # 7. Generar reporte Evidently
    report = Report(metrics=[DataDriftPreset()])

    # OJO: capturamos el resultado de run()
    eval_result = report.run(
        current_data=cur_df,      # Evidently espera current primero
        reference_data=ref_df
    )

    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)
    drift_report_path = reports_dir / "data_drift_report.html"

    # save_html se llama sobre eval_result, no sobre report
    eval_result.save_html(str(drift_report_path))

    # 8. Guardar métricas en JSON
    drift_metrics_path = reports_dir / "drift_metrics.json"
    output = {
        "model_path": str(model_path),
        "reference_metrics": ref_metrics,
        "drifted_metrics": cur_metrics,
        "degradation": degradation
    }

    with open(drift_metrics_path, "w") as f:
        json.dump(output, f, indent=4)

    print(f"Reporte Evidently guardado en: {drift_report_path}")
    print(f"Métricas de drift guardadas en: {drift_metrics_path}")

if __name__ == "__main__":
    main()
