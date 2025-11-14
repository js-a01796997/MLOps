from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
import pickle
import logging

class SarimaxRegressor(BaseEstimator, RegressorMixin):
    """
    Wrapper sobre scikit-learn para SARIMAX.
    
    Parámetros:
    -----------
    order : tuple(int,int,int)
        (p, d, q)
    seasonal_order : tuple(int,int,int,int)
        (P, D, Q, s)
    trend : str or None
        e.g. 'n', 'c', 't', 'ct' (igual que en SARIMAX)
    enforce_stationarity : bool
    enforce_invertibility : bool
    
    Notas:
    ------
    - Este wrapper asume una serie univariada endógena.
    - Guarda internamente el modelo ajustado (`self.result_`).
    """

    def __init__(
        self,
        order=(1,1,1),
        seasonal_order=(1,1,1,7),
        trend=None,
        enforce_stationarity=True,
        enforce_invertibility=True
    ):
        self.order = order
        self.seasonal_order = seasonal_order
        self.trend = trend
        self.enforce_stationarity = enforce_stationarity
        self.enforce_invertibility = enforce_invertibility

        # atributos que se llenan en fit
        self.model_ = None
        self.result_ = None
        self._y_index = None  # para recordar el índice temporal original

    def fit(self, y):
        """
        Ajusta el modelo SARIMAX.

        Parámetros:
        -----------
        y : pd.Series
            Serie objetivo (por ejemplo df['cnt'])

        Retorna:
        --------
        self
        """

        self._y_index = y.index  # recordar fechas (o posiciones)

        # Guardar referencia para usar luego en predict
        self.endog_ = y

        self.model_ = SARIMAX(
            endog=y,
            order=self.order,
            seasonal_order=self.seasonal_order,
            trend=self.trend,
            enforce_stationarity=self.enforce_stationarity,
            enforce_invertibility=self.enforce_invertibility,
        )

        self.result_ = self.model_.fit(disp=False)
        return self

    def predict(self, steps=None, return_conf_int=False, alpha=0.05):
        """
        Predice valores.

        Casos soportados:
        -----------------
        1) In-sample:
           - Llamas predict() SIN steps
           - Te da el ajuste sobre el rango visto en fit (one-step-ahead in-sample)

        2) Forecast futuro:
           - Llamas predict(steps=hacia_adelante)
           - Debes pasar X con las exógenas futuras (mismas columnas que durante fit)
             o None si el modelo no usa exógenas.

        Parámetros:
        -----------
        steps : int o None
            Cuántos pasos hacia adelante quieres predecir.
            Si es None → predicción in-sample.
        return_conf_int : bool
            Si True, regresa (yhat, conf_int)
        alpha : float
            Nivel de significancia para el intervalo de confianza.
            alpha=0.05 → 95% de confianza.

        Retorna:
        --------
        yhat : pd.Series o np.ndarray
        (yhat, conf_int) si return_conf_int=True
        """

        if self.result_ is None:
            raise RuntimeError("Debes llamar fit() antes de predict().")

        # Caso 1: predicción in-sample (sin steps)
        if steps is None:
            # get_prediction sobre el mismo rango entrenado
            pred_res = self.result_.get_prediction()
            mean = pred_res.predicted_mean

            if return_conf_int:
                ci = pred_res.conf_int(alpha=alpha)
                return mean, ci
            return mean

        # Caso 2: forecast out-of-sample (steps adelante)
        else:
            pred_res = self.result_.get_forecast(steps=steps)

            mean = pred_res.predicted_mean
            if return_conf_int:
                ci = pred_res.conf_int(alpha=alpha)
                return mean, ci
            return mean

    def get_params(self, deep=True):
        """
        Necesario para que GridSearchCV/Pipeline puedan inspeccionar hiperparámetros.
        """
        return {
            "order": self.order,
            "seasonal_order": self.seasonal_order,
            "trend": self.trend,
            "enforce_stationarity": self.enforce_stationarity,
            "enforce_invertibility": self.enforce_invertibility,
        }

    def set_params(self, **params):
        """
        Necesario para que GridSearchCV pueda setear hiperparámetros.
        """
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def save(self, path):
        """Guarda el modelo entrenado completo (incluye configuración y resultado)."""
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path):
        """Carga un modelo SarimaxRegressor guardado."""
        with open(path, 'rb') as f:
            return pickle.load(f)

    def save_statsmodels(self, path):
        """Guarda solo el resultado SARIMAX interno."""
        if self.result_ is not None:
            self.result_.save(path)
        else:
            raise RuntimeError("El modelo no ha sido ajustado todavía.")

    @classmethod
    def load_statsmodels(cls, path):
        """Carga solo el resultado interno de statsmodels."""
        from statsmodels.tsa.statespace.sarimax import SARIMAXResults
        result = SARIMAXResults.load(path)
        wrapper = cls()
        wrapper.result_ = result
        return wrapper



# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def main():
    # Config
    TRAIN_END = "2012-06-30"
    VAL_START, VAL_END = "2012-07-01", "2012-09-30"
    TEST_START, TEST_END = "2012-10-01", "2012-12-31"
    FREQ = "D"

    CSV_PATH = "data/processed/bike_sharing_cleaned.csv"
    baseDf = pd.read_csv(CSV_PATH)

    df = baseDf[['dteday', 'cnt']]
    df.columns = ['ds', 'y']

    # Evitar problemas con duplicados
    # Opción 1
    df = df.drop_duplicates(subset="ds", keep="first")
    df = df.dropna(subset=['ds'])

    df["ds"] = pd.to_datetime(df["ds"])
    # Target
    df["y"] = df["y"].astype(float)

    df = df.sort_values("ds").set_index("ds").asfreq(FREQ)

    # Particionado
    train      = df.loc[:TRAIN_END].copy()
    validation = df.loc[VAL_START:VAL_END].copy()
    test       = df.loc[TEST_START:TEST_END].copy()

    y_train_final = pd.concat([train['y'], validation['y']])
    y_test = test['y']

    sarimax_final = SarimaxRegressor(order=(1,1,1), seasonal_order=(1,1,1,7))
    sarimax_final.fit(y_train_final)

    y_forecast, y_ci = sarimax_final.predict(steps=len(y_test), return_conf_int=True)

    # --- Métricas ---
    mae = mean_absolute_error(y_test, y_forecast)
    rmse = np.sqrt(mean_squared_error(y_test, y_forecast))

    print("SARIMAX Results:")
    print(f"MAE:  {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")


if __name__ == "__main__":
    main()
