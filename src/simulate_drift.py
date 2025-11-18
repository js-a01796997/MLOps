import pandas as pd
from pathlib import Path
import numpy as np

SEED = 42
np.random.seed(SEED)

def main():
    # 1. Cargar dataset de referencia
    ref_path = Path("data/processed/valid.csv")
    df_ref = pd.read_csv(ref_path)

    # Copia para modificarla (drifted)
    df_drifted = df_ref.copy()

    # 2. Simular un cambio de clima (más calor y humedad)
    if "temp" in df_drifted.columns:
        # Aumentar temperatura promedio en +0.1
        df_drifted["temp"] = (df_drifted["temp"] + 0.1).clip(0, 1)

    if "atemp" in df_drifted.columns:
        df_drifted["atemp"] = (df_drifted["atemp"] + 0.1).clip(0, 1)

    if "hum" in df_drifted.columns:
        # Aumentar humedad
        df_drifted["hum"] = (df_drifted["hum"] + 0.15).clip(0, 1)

    # 3. Simular cambio en patrón temporal (más actividad nocturna)
    if "hr" in df_drifted.columns:
        # Ejemplo: rotar horas +3
        df_drifted["hr"] = (df_drifted["hr"] + 3) % 24

    # 4. Guardar dataset con drift
    out_dir = Path("data/monitoring")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "drifted.csv"
    df_drifted.to_csv(out_path, index=False)

    print(f"Referencia: {ref_path}  ->  Drifted: {out_path}")

if __name__ == "__main__":
    main()
