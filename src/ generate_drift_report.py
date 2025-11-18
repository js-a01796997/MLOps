
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# ============================
# CONFIGURACIÓN BÁSICA
# ============================

# Archivo resumen generado por run_drift_analysis.py
SUMMARY_PATH = Path("drift_summary.csv")

# Dataset base (sin drift)
BASE_DATA_PATH = Path("data/processed/dataset_clean.csv")

# Datasets con drift (rutas reales que ya tienes)
DRIFT_CONFIG = [
    {
        "path": Path("data/processed/dataset_season_drift.csv"),
        "label": "Drift estacional (season)",
        "feature": "season"
    },
    {
        "path": Path("data/processed/dataset_wind_drift.csv"),
        "label": "Drift en velocidad del viento (windspeed)",
        "feature": "windspeed"
    },
    {
        "path": Path("data/processed/dataset_temp_drift.csv"),
        "label": "Drift en temperatura (temp)",
        "feature": "temp"
    },
]

# Métricas base (las que ya viste en consola con tu XGBoost)
BASE_RMSE = 265.6147
BASE_MAE = 191.7652
BASE_R2 = -1.0867


# ============================
# FUNCIONES AUXILIARES
# ============================

def wrap_text(text, width=90):
    """Envuelve texto largo para mostrarlo bonito en la figura."""
    return "\n".join(textwrap.wrap(text, width=width))


def load_summary(path: Path) -> pd.DataFrame:
    """
    Carga drift_summary.csv y valida columnas mínimas.
    Se asume que tiene: dataset, rmse, mae, r2
    """
    if not path.exists():
        raise FileNotFoundError(f"No se encontró {path}. Asegúrate de correr antes run_drift_analysis.py")

    df = pd.read_csv(path)

    expected_cols = {"dataset", "rmse", "mae", "r2"}
    if not expected_cols.issubset(set(df.columns)):
        raise ValueError(
            f"Se esperaba que {path} tuviera las columnas {expected_cols}, "
            f"pero tiene: {list(df.columns)}. Ajusta el script o el CSV."
        )

    # Extraemos el nombre simple del archivo para usarlo en gráficos
    df["dataset_name"] = df["dataset"].apply(lambda x: Path(x).name)
    return df


def add_delta_vs_base(summary_df: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega columnas de diferencia contra la métrica base conocida.
    """
    df = summary_df.copy()
    df["delta_rmse"] = df["rmse"] - BASE_RMSE
    df["delta_mae"] = df["mae"] - BASE_MAE
    df["delta_r2"] = df["r2"] - BASE_R2
    return df


def find_worst_drift(summary_df: pd.DataFrame) -> pd.Series:
    """
    Define el "peor drift" como el que más incrementa el RMSE.
    """
    idx = summary_df["delta_rmse"].abs().idxmax()
    return summary_df.loc[idx]


# ============================
# PÁGINAS DEL PDF
# ============================

def add_title_page(pdf: PdfPages):
    """Página 1: Título y descripción general."""
    fig, ax = plt.subplots(figsize=(11.69, 8.27))  # A4 horizontal
    ax.axis("off")

    title = "Simulación de Data Drift y Pérdida de Performance\nModelo XGBoost – Bike Sharing"
    desc = """
    Este reporte documenta la simulación de data drift sobre el modelo de predicción de demanda de bicicletas 
    entrenado con XGBoost. A partir de un dataset base (sin drift) se generaron diferentes escenarios de cambio 
    de distribución en las variables de entrada, y se evaluó el impacto en las métricas de desempeño del modelo 
    (RMSE, MAE y R²).

    El objetivo es demostrar cómo pequeños cambios en la distribución de los datos de entrada pueden degradar 
    el desempeño del modelo en producción y justificar la necesidad de monitoreo continuo, umbrales de alerta 
    y estrategias de retraining o ajuste del pipeline de features.
    """

    ax.text(0.5, 0.8, title, ha="center", va="center", fontsize=18, fontweight="bold")
    ax.text(0.5, 0.4, wrap_text(desc, 100), ha="center", va="center", fontsize=11)

    pdf.savefig(fig)
    plt.close(fig)


def add_base_metrics_page(pdf: PdfPages):
    """Página 2: Métricas base del modelo sin drift."""
    fig, ax = plt.subplots(figsize=(11.69, 8.27))
    ax.axis("off")

    title = "Línea Base del Modelo (sin Data Drift)"
    text = f"""
    El modelo de referencia es un XGBoost entrenado sobre el dataset de entrenamiento limpio y procesado 
    (sin drift). Las métricas de evaluación sobre el conjunto base (dataset_clean) son:

      • RMSE base  = {BASE_RMSE:.4f}
      • MAE base   = {BASE_MAE:.4f}
      • R² base    = {BASE_R2:.4f}

    Estas métricas representan el “punto de referencia” contra el cual se comparan los escenarios con drift. 
    Cualquier incremento significativo en RMSE/MAE o caída en R² indica degradación de performance.
    """

    ax.text(0.5, 0.8, title, ha="center", va="center", fontsize=16, fontweight="bold")
    ax.text(0.05, 0.55, wrap_text(text, 110), ha="left", va="top", fontsize=11)

    pdf.savefig(fig)
    plt.close(fig)


def add_summary_table_page(pdf: PdfPages, summary_df: pd.DataFrame):
    """Página 3: tabla resumen con las métricas de cada drift."""
    fig, ax = plt.subplots(figsize=(11.69, 8.27))
    ax.axis("off")

    ax.set_title("Resumen de Data Drift por Dataset", fontsize=16, fontweight="bold", pad=20)

    cols_to_show = ["dataset_name", "rmse", "mae", "r2", "delta_rmse", "delta_mae", "delta_r2"]
    table_df = summary_df[cols_to_show].copy()
    table_df["rmse"] = table_df["rmse"].round(4)
    table_df["mae"] = table_df["mae"].round(4)
    table_df["r2"] = table_df["r2"].round(4)
    table_df["delta_rmse"] = table_df["delta_rmse"].round(4)
    table_df["delta_mae"] = table_df["delta_mae"].round(4)
    table_df["delta_r2"] = table_df["delta_r2"].round(4)

    table = ax.table(
        cellText=table_df.values,
        colLabels=table_df.columns,
        loc="center",
        cellLoc="center"
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.5)

    pdf.savefig(fig)
    plt.close(fig)


def add_metrics_barplots_page(pdf: PdfPages, summary_df: pd.DataFrame):
    """Página 4: gráficas de barras comparando métricas vs base."""
    fig, axes = plt.subplots(1, 3, figsize=(11.69, 8.27))
    fig.suptitle("Comparación de Métricas vs Línea Base", fontsize=16, fontweight="bold")

    x = np.arange(len(summary_df))
    labels = summary_df["dataset_name"].tolist()

    # RMSE
    axes[0].bar(x, summary_df["rmse"], color="#1f77b4")
    axes[0].axhline(BASE_RMSE, color="red", linestyle="--", label="Base")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=30, ha="right")
    axes[0].set_title("RMSE")
    axes[0].legend()

    # MAE
    axes[1].bar(x, summary_df["mae"], color="#ff7f0e")
    axes[1].axhline(BASE_MAE, color="red", linestyle="--", label="Base")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=30, ha="right")
    axes[1].set_title("MAE")
    axes[1].legend()

    # R²
    axes[2].bar(x, summary_df["r2"], color="#2ca02c")
    axes[2].axhline(BASE_R2, color="red", linestyle="--", label="Base")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels, rotation=30, ha="right")
    axes[2].set_title("R²")
    axes[2].legend()

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    pdf.savefig(fig)
    plt.close(fig)


def add_distribution_pages(pdf: PdfPages):
    """
    Páginas 5–7: para cada drift, comparación visual de la distribución
    de la feature afectada entre base y drift.
    """
    base_df = pd.read_csv(BASE_DATA_PATH)

    for cfg in DRIFT_CONFIG:
        drift_path = cfg["path"]
        label = cfg["label"]
        feature = cfg["feature"]

        if not drift_path.exists():
            print(f"⚠️ No se encontró {drift_path}, se omite página de {label}")
            continue

        drift_df = pd.read_csv(drift_path)

        if feature not in base_df.columns or feature not in drift_df.columns:
            print(f"⚠️ La columna {feature} no existe en alguno de los datasets, se omite.")
            continue

        fig, axes = plt.subplots(1, 2, figsize=(11.69, 8.27))
        fig.suptitle(f"Distribución de {feature} — {label}", fontsize=16, fontweight="bold")

        # Histograma
        axes[0].hist(base_df[feature].dropna(), bins=30, alpha=0.6, label="Base")
        axes[0].hist(drift_df[feature].dropna(), bins=30, alpha=0.6, label="Drift")
        axes[0].set_title("Histograma")
        axes[0].set_xlabel(feature)
        axes[0].set_ylabel("Frecuencia")
        axes[0].legend()

        # Boxplot
        axes[1].boxplot(
            [base_df[feature].dropna(), drift_df[feature].dropna()],
            labels=["Base", "Drift"]
        )
        axes[1].set_title("Boxplot comparativo")
        axes[1].set_ylabel(feature)

        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        pdf.savefig(fig)
        plt.close(fig)


def add_conclusion_page(pdf: PdfPages, summary_df: pd.DataFrame):
    """Última página: resumen ejecutivo y acciones recomendadas."""
    worst = find_worst_drift(summary_df)

    fig, ax = plt.subplots(figsize=(11.69, 8.27))
    ax.axis("off")

    title = "Conclusiones y Recomendaciones"

    text = f"""
    • El modelo XGBoost presenta una línea base con RMSE ~ {BASE_RMSE:.2f} y MAE ~ {BASE_MAE:.2f}. 
      Estos valores sirven como referencia para evaluar la degradación por drift.

    • De los escenarios simulados, el peor caso (por incremento absoluto en RMSE) fue:
        - Dataset: {worst['dataset_name']}
        - ΔRMSE: {worst['delta_rmse']:.4f}
        - ΔMAE:  {worst['delta_mae']:.4f}
        - ΔR²:   {worst['delta_r2']:.4f}

    • Aunque los cambios en este experimento no son extremos, ilustran cómo una variación en la 
      distribución de features clave (por ejemplo temperatura, viento o estacionalidad) puede modificar 
      ligeramente las métricas de error. En producción, cambios más grandes podrían implicar fallas 
      importantes en la calidad de las predicciones.

    • Umbrales propuestos (ejemplo):
        - Alerta amarilla: |ΔRMSE| > 5  o  |ΔMAE| > 5
        - Alerta roja:     |ΔRMSE| > 10 o  |ΔMAE| > 10  o caída de R² mayor a 0.05

    • Acción recomendada cuando se detecta drift:
        - Revisar el pipeline de features para entender qué variables cambiaron.
        - Verificar si el rango de los datos sigue estando dentro de lo visto en entrenamiento.
        - Si el drift persiste, entrenar de nuevo el modelo con datos recientes (retraining) y 
          actualizar el modelo en el registro (MLflow / DVC).

    En resumen, la simulación demuestra la importancia de monitorear tanto la distribución de los datos 
    como las métricas de performance del modelo para activar acciones de mantenimiento proactivo.
    """

    ax.text(0.5, 0.85, title, ha="center", va="center", fontsize=16, fontweight="bold")
    ax.text(0.05, 0.70, wrap_text(text, 110), ha="left", va="top", fontsize=11)

    pdf.savefig(fig)
    plt.close(fig)


# ============================
# MAIN
# ============================

def main():
    print("📌 Cargando resumen de drift…")
    summary = load_summary(SUMMARY_PATH)
    summary = add_delta_vs_base(summary)

    output_pdf = Path("drift_report.pdf")
    print(f"📝 Generando reporte PDF en: {output_pdf}")

    with PdfPages(output_pdf) as pdf:
        add_title_page(pdf)
        add_base_metrics_page(pdf)
        add_summary_table_page(pdf, summary)
        add_metrics_barplots_page(pdf, summary)
        add_distribution_pages(pdf)
        add_conclusion_page(pdf, summary)

    print("✅ Reporte generado correctamente:", output_pdf)


if __name__ == "__main__":
    main()