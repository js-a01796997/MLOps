from fastapi import FastAPI
import os
import mlflow
from api_v1 import api_v1
from api_v2 import api_v2

# ========= MLflow Configuration =========
# Configurar explícitamente el tracking URI desde la variable de entorno
# Esta configuración se aplica antes de importar los routers para asegurar
# que MLflow esté configurado cuando api_v2.py use mlflow.pyfunc.load_model()
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
if MLFLOW_TRACKING_URI:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    print(f"✓ MLflow Tracking URI configurado: {MLFLOW_TRACKING_URI}")
else:
    print("⚠ Advertencia: MLFLOW_TRACKING_URI no está definida. "
          "MLflow usará el tracking URI por defecto (file:./mlruns). "
          "Para usar un servidor remoto, exporta: export MLFLOW_TRACKING_URI=<uri>")

app = FastAPI(
    title="Bike Sharing Prediction API (Local and MLflow registry)",
    description="API que usa MLflow Model Registry como fuente de modelos o con modelos locales",
    version="2.0.0",
)

# Modelos Locales
app.include_router(api_v1)
# Modelos de MLFlow
app.include_router(api_v2)

@app.get("/", summary="Root", include_in_schema=False)
async def root():
    return {"status": "ok", "message": "Bike Sharing Prediction API"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)