from fastapi import FastAPI
from api_v1 import api_v1
from api_v2 import api_v2

# NOTAS:
# Necesita la variable de entorno para conocer donde esta MLFlow
# export MLFLOW_TRACKING_URI=https://mlflow.labs.jsdevart.com/

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