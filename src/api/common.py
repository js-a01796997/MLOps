from pydantic import BaseModel
from typing import List, Optional
from fastapi import HTTPException

# ========= Common Schemas =========

class PredictRequest(BaseModel):
    features: Optional[List[List[float]]] = None


class PredictResponse(BaseModel):
    predictions: List[float]

# ========= Common Utils =========

def validate_predict_input(req: PredictRequest, model_id: str, expected_n_features: int):
    # Validación básica de que haya features
    if req.features is None:
        raise HTTPException(
            status_code=400,
            detail="Debes enviar 'features' con una o varias observaciones",
        )

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
