from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime, timezone
import pickle
import numpy as np

app = FastAPI(
    title="Bike Sharing Prediction API",
    description="API para cargar un modelo (pickle) y hacer predicciones /predict",
    version="1.0.0",
)

model = None  # modelo en memoria

class ModelMetadata(BaseModel):
    filename: str
    uploaded_at: datetime
    model_class: str
    n_features: Optional[int] = None
    feature_names: Optional[List[str]] = None

model_metadata: Optional[ModelMetadata] = None

# ======== Esquemas de entrada/salida ========

class PredictRequest(BaseModel):
    """
    Para enviar datos al endpoint /predict.

    Ejemplos:
    - Acepta múltiples observaciones:
      { "features": [[v1,...,vN],[v1,...,vN], ...] }
    """
    features: Optional[List[List[float]]] = None


class PredictResponse(BaseModel):
    predictions: List[float]


# ======== Endpoints ========

@app.post("/upload_model", summary="Subir modelo entrenado en formato pickle")
async def upload_model(file: UploadFile = File(...)):
    """
    Carga un modelo en memoria a partir de un archivo pickle.
    También extrae metadata útil (n_features, feature_names, etc.).
    """
    global model, model_metadata

    if not file.filename.endswith(".pkl") and not file.filename.endswith(".pickle"):
        raise HTTPException(status_code=400, detail="El archivo debe ser .pkl o .pickle")

    try:
        contents = await file.read()
        loaded_model = pickle.loads(contents)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"No se pudo cargar el modelo: {e}")

    if not hasattr(loaded_model, "predict"):
        raise HTTPException(status_code=400, detail="El objeto cargado no tiene método 'predict'")

    # Guardamos el modelo en memoria
    model = loaded_model

    # Intentamos detectar metadata del modelo
    n_features = getattr(loaded_model, "n_features_in_", None)
    feature_names = getattr(loaded_model, "feature_names_in_", None)
    if feature_names is not None:
        # convertimos a lista
        feature_names = [str(f) for f in feature_names]

    model_metadata = ModelMetadata(
        filename=file.filename,
        uploaded_at=datetime.now(timezone.utc),
        model_class=type(loaded_model).__name__,
        n_features=int(n_features) if n_features is not None else None,
        feature_names=feature_names,
    )

    return {
        "message": "Modelo cargado correctamente",
        "metadata": model_metadata,
    }


@app.get("/model_info", response_model=ModelMetadata, summary="Obtener metadata del modelo cargado")
async def model_info():
    """
    Devuelve información del modelo actualmente cargado:
    - nombre del archivo
    - fecha de carga
    - clase del modelo
    - número de features esperadas
    - nombres de las features (si están disponibles)
    """
    if model_metadata is None:
        raise HTTPException(status_code=404, detail="No hay modelo cargado actualmente.")

    return model_metadata


@app.post("/predict", response_model=PredictResponse, summary="Obtener predicciones del modelo")
async def predict(req: PredictRequest):
    """
    Realiza predicciones usando el modelo cargado.
    Valida tamaño de los vectores contra n_features_in_ (cuando está disponible).
    """
    global model, model_metadata

    if model is None:
        raise HTTPException(status_code=400, detail="No hay un modelo cargado. Usa /upload_model primero.")

    # Validación básica de que haya features
    if req.features is None:
        raise HTTPException(
            status_code=400,
            detail="Debes enviar 'features' con una o varias observaciones",
        )

    # Obtenemos el número de features esperado (si está disponible)
    expected_n_features = None
    if model_metadata is not None and model_metadata.n_features is not None:
        expected_n_features = model_metadata.n_features
    else:
        # fallback: intentar leer directamente del modelo
        expected_n_features = getattr(model, "n_features_in_", None)

    # Normalizamos a matriz X y validamos dimensiones
    if len(req.features) == 0:
        raise HTTPException(
            status_code=422,
            detail="'features' no puede estar vacío.",
        )

    # Todas las filas deben tener la misma longitud
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
                    f"pero el modelo espera {expected_n_features} features.",
        )

    X = np.array(req.features, dtype=float)

    # Hacemos la predicción
    try:
        preds = model.predict(X)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al predecir: {e}")

    preds_list = [float(p) for p in preds]
    return PredictResponse(predictions=preds_list)


@app.get("/", summary="health check")
async def root():
    return {"status": "ok", "message": "Bike Sharing Prediction API"}
