from fastapi import FastAPI, APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime, timezone
import pickle
import numpy as np

api_v1 = APIRouter(prefix="/v1", tags=["v1 (Local Models)"])

# ========= Model registry (en memoria) =========

class ModelMetadata(BaseModel):
    model_id: str
    filename: str
    uploaded_at: datetime
    model_class: str
    n_features: Optional[int] = None
    feature_names: Optional[List[str]] = None

models: Dict[str, Any] = {}
models_metadata: Dict[str, ModelMetadata] = {}

# ========= Schemas Entrada/Salida =========

class PredictRequest(BaseModel):
    features: Optional[List[List[float]]] = None


class PredictResponse(BaseModel):
    predictions: List[float]


# ========= Helpers =========

def get_model_or_404(model_id: str):
    if model_id not in models:
        raise HTTPException(
            status_code=404,
            detail=f"Modelo '{model_id}' no encontrado. Sube primero el modelo con /v1/models/{model_id}/upload",
        )
    return models[model_id], models_metadata[model_id]


# ========= Endpoints v1 =========

@api_v1.post("/models/{model_id}/upload", summary="Subir modelo en pickle para un model_id específico")
async def upload_model(model_id: str, file: UploadFile = File(...)):
    global models, models_metadata

    if not file.filename.endswith(".pkl") and not file.filename.endswith(".pickle"):
        raise HTTPException(status_code=400, detail="El archivo debe ser .pkl o .pickle")

    try:
        contents = await file.read()
        loaded_model = pickle.loads(contents)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"No se pudo cargar el modelo: {e}")

    if not hasattr(loaded_model, "predict"):
        raise HTTPException(status_code=400, detail="El objeto cargado no tiene método 'predict'")

    # Guardar en el registry
    models[model_id] = loaded_model

    # Extraer metadata
    n_features = getattr(loaded_model, "n_features_in_", None)
    feature_names = getattr(loaded_model, "feature_names_in_", None)
    if feature_names is not None:
        feature_names = [str(f) for f in feature_names]

    meta = ModelMetadata(
        model_id=model_id,
        filename=file.filename,
        uploaded_at=datetime.now(timezone.utc),
        model_class=type(loaded_model).__name__,
        n_features=int(n_features) if n_features is not None else None,
        feature_names=feature_names,
    )
    models_metadata[model_id] = meta

    return {
        "message": f"Modelo '{model_id}' cargado correctamente",
        "metadata": meta,
    }


@api_v1.get("/models/{model_id}/info", response_model=ModelMetadata, summary="Obtener metadata de un modelo")
async def model_info(model_id: str):
    if model_id not in models_metadata:
        raise HTTPException(status_code=404, detail=f"No hay metadata para el modelo '{model_id}'.")

    return models_metadata[model_id]

@api_v1.post(
    "/models/{model_id}/predict",
    response_model=PredictResponse,
    summary="Obtener predicciones de un modelo específico",
)
async def predict(model_id: str, req: PredictRequest):
    global models, models_metadata

    model, meta = get_model_or_404(model_id)

    # Validación básica de que haya features
    if req.features is None:
        raise HTTPException(
            status_code=400,
            detail="Debes enviar 'features' con una o varias observaciones",
        )

    expected_n_features = meta.n_features or getattr(model, "n_features_in_", None)

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


# healthcheck para v1
@api_v1.get("/health", summary="health check v1")
async def health_v1():
    return {"status": "ok", "api_version": "v1"}

if __name__ == "__main__":
    # Probar modulo individual
    import uvicorn

    app = FastAPI(
        title="Bike Sharing Prediction API",
        description="API versionada para cargar múltiples modelos y hacer predicciones",
        version="1.1.0",
    )

    app.include_router(api_v1)
    
    @app.get("/", summary="Root", include_in_schema=False)
    async def root():
        return {"status": "ok", "message": "Bike Sharing Prediction API"}

    uvicorn.run(app, host="0.0.0.0", port=8000)
