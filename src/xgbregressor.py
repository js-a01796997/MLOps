import logging
from pathlib import Path
from typing import Dict, Tuple, Optional, Any, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import pickle
from xgboost import XGBRegressor

# =============================================
# CONFIGURACIÓN DE LOGGING
# =============================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# =============================================
# CLASE PRINCIPAL
# =============================================
class XGBoostBikePredictor:
    """Clase para entrenar y evaluar un modelo XGBoost para predicción de bicicletas."""
    
    def __init__(
        self,
        target_col: str = "cnt",
        test_size: float = 0.2,
        random_state: int = 42,
        model_params: Optional[Dict] = None
    ):
        self.target_col = target_col
        self.test_size = test_size
        self.random_state = random_state
        self.model_params = model_params or {
            "n_estimators": 1000,
            "learning_rate": 0.05,
            "max_depth": 6,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": random_state,
            "n_jobs": -1,
            "verbosity": 1,
            "eval_metric": "rmse"
        }
        self.imputer = SimpleImputer(strategy="median")
        self.feature_columns = None
        self.best_model = None
        self.pipeline = None
        self.output_dir = Path("models")
        self.X_val = None
        self.y_val = None
        
    # =============================================
    # MÉTODOS DE PREPARACIÓN DE DATOS
    # =============================================
    def load_data(self, filepath: Union[str, Path]) -> pd.DataFrame:
        logger.info(f"Loading data from {filepath}")
        return pd.read_csv(filepath)
    
    def prepare_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        logger.info("Preparing features and target")
        X = df.drop(columns=[self.target_col]).copy()
        y = df[self.target_col].copy()
        
        bool_cols = X.select_dtypes(include="bool").columns
        X[bool_cols] = X[bool_cols].astype(int)
        
        self.feature_columns = X.columns.tolist()
        return X, y
    
    def preprocess_data(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple:
        logger.info("Applying preprocessing (SimpleImputer)")
        X_train_imp = self.imputer.fit_transform(X_train)
        X_test_imp = self.imputer.transform(X_test)
        return X_train_imp, X_test_imp

    # =============================================
    # ENTRENAMIENTO Y EVALUACIÓN
    # =============================================
    def build_pipeline(self, xgb_params: Optional[Dict[str, Any]] = None):
        logger.info("Building sklearn Pipeline")
        xgb_params = xgb_params or self.model_params
        model = XGBRegressor(**xgb_params)
        self.pipeline = Pipeline([
            ("imputer", self.imputer),
            ("model", model)
        ])
    
    def grid_search(self, X_train: np.ndarray, y_train: np.ndarray, param_grid: Dict) -> None:
        logger.info("Starting GridSearchCV")
        xgb_reg = XGBRegressor(objective="reg:squarederror", random_state=self.random_state)
        
        grid_search = GridSearchCV(
            estimator=xgb_reg,
            param_grid=param_grid,
            cv=5,
            scoring="neg_mean_squared_error",
            verbose=2,
            n_jobs=-1
        )
        
        grid_search.fit(X_train, y_train)
        logger.info(f"Best parameters: {grid_search.best_params_}")
        logger.info(f"Best RMSE (CV): {np.sqrt(-grid_search.best_score_):.4f}")
        self.best_model = grid_search.best_estimator_
    
    def train(self, X_train, y_train, X_val=None, y_val=None):
        if self.pipeline is None:
            raise RuntimeError("Pipeline not built. Call build_pipeline() first.")
        
        self.X_val = X_val
        self.y_val = y_val
        logger.info("Training model pipeline")
        self.pipeline.fit(X_train, y_train)
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        if self.pipeline is None:
            raise RuntimeError("Pipeline not built. Call build_pipeline() first.")
        
        y_pred = self.pipeline.predict(X)
        metrics = {
            "rmse": float(np.sqrt(mean_squared_error(y, y_pred))),
            "mae": float(mean_absolute_error(y, y_pred)),
            "r2": float(r2_score(y, y_pred))
        }
        logger.info(f"Evaluation metrics: {metrics}")
        return metrics
    
    # =============================================
    # GUARDADO DEL MODELO Y MÉTRICAS
    # =============================================
    def save(self, filename: Optional[Path] = None, test_metrics: Optional[Dict[str, float]] = None):
        filename = filename or (self.output_dir / "modelo_xgboost.pickle")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        val_metrics = self.evaluate(self.X_val, self.y_val) if self.X_val is not None else {}
        test_metrics = test_metrics or {}

        payload = {
            "pipeline": self.pipeline,
            "feature_columns": self.feature_columns,
            "metrics": {
                "valid_rmse": float(val_metrics.get("rmse", np.nan)),
                "valid_mae": float(val_metrics.get("mae", np.nan)),
                "valid_r2": float(val_metrics.get("r2", np.nan)),
                "test_rmse": float(test_metrics.get("rmse", np.nan)),
                "test_mae": float(test_metrics.get("mae", np.nan)),
                "test_r2": float(test_metrics.get("r2", np.nan)),
            }
        }

        with open(filename, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info(f"✅ Saved pipeline, metrics and metadata to {filename}")


# =============================================
# MAIN
# =============================================
def main():
    train_path = Path("data/processed/train.csv")
    valid_path = Path("data/processed/valid.csv")
    test_path = Path("data/processed/test.csv")
    model_path = Path("models/modelo_xgboost.pickle")
    
    predictor = XGBoostBikePredictor()
    
    # Cargar datos
    train_df = predictor.load_data(train_path)
    valid_df = predictor.load_data(valid_path)
    test_df = predictor.load_data(test_path)
    
    # Preparar datos
    X_train, y_train = predictor.prepare_data(train_df)
    X_valid, y_valid = predictor.prepare_data(valid_df)
    X_test, y_test = predictor.prepare_data(test_df)
    
    # Preprocesar
    X_train_imp, X_valid_imp = predictor.preprocess_data(X_train, X_valid)
    _, X_test_imp = predictor.preprocess_data(X_train, X_test)
    
    # Grid search
    param_grid = {
        "max_depth": [5, 7, 9],
        "learning_rate": [0.01, 0.1, 0.3],
        "n_estimators": [100],
        "min_child_weight": [1, 3, 5],
        "gamma": [0, 0.1, 0.2],
        "subsample": [0.8, 0.9, 1.0],
        "colsample_bytree": [0.6, 0.7, 0.8]
    }
    
    logger.info("Starting Grid Search for best parameters...")
    predictor.grid_search(X_train_imp, y_train, param_grid)
    
    best_params = predictor.best_model.get_params()
    logger.info(f"Using best parameters found: {best_params}")
    predictor.build_pipeline(xgb_params=best_params)
    
    # Entrenar
    predictor.train(X_train_imp, y_train, X_valid_imp, y_valid)
    
    # Evaluar test
    test_metrics = predictor.evaluate(X_test_imp, y_test)
    logger.info(f"📊 Test set metrics: {test_metrics}")
    
    # Guardar modelo con métricas de test incluidas
    predictor.save(model_path, test_metrics=test_metrics)


if __name__ == "__main__":
    main()
