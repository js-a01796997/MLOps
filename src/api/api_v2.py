from fastapi import FastAPI, APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime, timezone
import numpy as np
import mlflow
from mlflow.tracking import MlflowClient

api_v2 = APIRouter(prefix="/v2", tags=["v2 (MLflow Registry)"])

# ========= Caché simple en memoria =========

class ModelMetadata(BaseModel):
    model_id: str           # modelo de mlflow {model_id}:{version|stage}
    mlflow_uri: str         # models:/name/stage o models:/name/version
    loaded_at: datetime
    model_class: str
    n_features: Optional[int] = None
    feature_names: Optional[List[str]] = None

# ==== Schemas para la respuesta de MLFlow ====

class ModelVersionInfo(BaseModel):
    version: str
    current_stage: str
    creation_timestamp: Optional[datetime] = None
    last_updated_timestamp: Optional[datetime] = None
    run_id: Optional[str] = None


class RegisteredModelInfo(BaseModel):
    name: str
    creation_timestamp: Optional[datetime] = None
    last_updated_timestamp: Optional[datetime] = None
    versions: List[ModelVersionInfo]


# caché de modelos cargados para no hacer tantas peticiones a MLflow en cada request
model_cache: Dict[str, Any] = {}
model_cache_metadata: Dict[str, ModelMetadata] = {}


# ========= Schemas =========

class PredictRequest(BaseModel):
    features: Optional[List[List[float]]] = None


class PredictResponse(BaseModel):
    predictions: List[float]


# ========= Helpers =========

def parse_model_id(model_id: str) -> str:
    """
    Convención:
      model_id = "<mlflow_name>:<stage_or_version>"

    Ejemplos:

      "bike_sharing_xgboost:Production" -> "models:/bike_sharing_xgboost/Production"
      "bike_sharing_xgboost:8"          -> "models:/bike_sharing_xgboost/8"
    """
    parts = model_id.split(":")
    if len(parts) != 2:
        raise HTTPException(
            status_code=400,
            detail="model_id debe tener formato '<mlflow_name>:<stage_o_version>' (ej. 'bike_sharing_xgboost:8')",
        )
    name, stage_or_version = parts
    return f"models:/{name}/{stage_or_version}"


def get_model_from_registry(model_id: str):
    """
    Obtiene el modelo desde caché o desde MLflow si no está en caché.
    """
    global model_cache, model_cache_metadata

    # Si está en caché, lo usamos
    if model_id in model_cache:
        return model_cache[model_id], model_cache_metadata[model_id]

    # Si no está, lo cargamos desde MLflow
    mlflow_uri = parse_model_id(model_id)

    try:
        loaded_model = mlflow.pyfunc.load_model(mlflow_uri)
    except Exception as e:
        raise HTTPException(
            status_code=404,
            detail=f"No se pudo cargar el modelo desde MLflow para model_id='{model_id}' "
                   f"(uri='{mlflow_uri}'): {e}",
        )

    model_class = type(loaded_model).__name__

    # Intentamos detectar número de features desde el modelo "raw"
    base_model = loaded_model.get_raw_model()
    n_features = None
    feature_names = None
    if base_model is not None:
        n_features = getattr(base_model, "n_features_in_", None)
        feature_names = getattr(base_model, "feature_names_in_", None)
        model_class = type(base_model).__name__
        if feature_names is not None:
            feature_names = [str(f) for f in feature_names]

    meta = ModelMetadata(
        model_id=model_id,
        mlflow_uri=mlflow_uri,
        loaded_at=datetime.utcnow(),
        model_class=model_class,
        n_features=int(n_features) if n_features is not None else None,
        feature_names=feature_names,
    )

    # Guardamos en caché
    model_cache[model_id] = loaded_model
    model_cache_metadata[model_id] = meta

    return loaded_model, meta


# ========= Endpoints =========

@api_v2.get("/models/{model_id}/info", response_model=ModelMetadata, summary="Obtener información de un modelo (desde caché o MLflow)")
async def model_info(model_id: str):
    _, meta = get_model_from_registry(model_id)
    return meta


@api_v2.post(
    "/models/{model_id}/predict",
    response_model=PredictResponse,
    summary="Obtener predicciones usando un modelo registrado en MLflow",
)
async def predict(model_id: str, req: PredictRequest):
    model, meta = get_model_from_registry(model_id)

    if req.features is None:
        raise HTTPException(
            status_code=400,
            detail="Debes enviar 'features' con una o varias observaciones",
        )

    expected_n_features = meta.n_features  # puede ser None

    # Normalización y validación
    if len(req.features) == 0:
        raise HTTPException(
            status_code=422,
            detail="'features' no puede estar vacío.",
        )

    row_lengths = {len(row) for row in req.features}
    if len(row_lengths) != 1:
        raise HTTPException(
            status_code=422,
            detail=f"Todas las filas de 'features' deben tener la misma longitud. "
                    f"Se encontraron longitudes: {sorted(row_lengths)}",
        )

    num_features_row = row_lengths.pop()
    if expected_n_features is not None and num_features_row != expected_n_features:
        raise HTTPException(
            status_code=422,
            detail=f"Cada fila de 'features' tiene {num_features_row} features, "
                    f"pero el modelo '{model_id}' espera {expected_n_features} features.",
        )

    X = np.array(req.features, dtype=float)

    try:
        preds = model.predict(X)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al predecir con el modelo '{model_id}': {e}")

    preds_list = [float(p) for p in preds]
    return PredictResponse(predictions=preds_list)

@api_v2.get("/models", response_model=List[RegisteredModelInfo], summary="Listar modelos registrados en MLflow")
async def list_registered_models():
    """
    Lista todos los modelos registrados en MLflow Model Registry,
    junto con sus versiones y etapas (None, Staging, Production, Archived, etc.).
    """
    try:
        client = MlflowClient()
        registered_models = client.search_registered_models()
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"No se pudieron obtener los modelos desde MLflow: {e}",
        )

    result: List[RegisteredModelInfo] = []

    for rm in registered_models:
        # Timestamps vienen en ms desde epoch
        def ts_to_dt(ts_ms):
            if ts_ms is None:
                return None
            return datetime.fromtimestamp(ts_ms / 1000.0, timezone.utc)

        versions: List[ModelVersionInfo] = []
        for mv in rm.latest_versions:
            versions.append(
                ModelVersionInfo(
                    version=mv.version,
                    current_stage=mv.current_stage,
                    creation_timestamp=ts_to_dt(getattr(mv, "creation_timestamp", None)),
                    last_updated_timestamp=ts_to_dt(getattr(mv, "last_updated_timestamp", None)),
                    run_id=getattr(mv, "run_id", None),
                )
            )

        result.append(
            RegisteredModelInfo(
                name=rm.name,
                creation_timestamp=ts_to_dt(getattr(rm, "creation_timestamp", None)),
                last_updated_timestamp=ts_to_dt(getattr(rm, "last_updated_timestamp", None)),
                versions=versions,
            )
        )

    return result

@api_v2.get("/health", summary="healt check v2")
async def health_v1():
    return {"status": "ok", "api_version": "v2"}

if __name__ == "__main__":
    # Probar modulo individual
    import uvicorn

    app = FastAPI(
        title="Bike Sharing Prediction API (MLflow registry)",
        description="API que usa MLflow Model Registry como fuente de modelos",
        version="2.0.0",
    )
    
    app.include_router(api_v2)
    
    @app.get("/", summary="Root", include_in_schema=False)
    async def root():
        return {"status": "ok", "message": "API con MLflow registry"}

    uvicorn.run(app, host="0.0.0.0", port=8000)
