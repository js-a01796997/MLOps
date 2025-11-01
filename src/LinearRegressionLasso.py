"""
DEPRECATED: This script is deprecated and will be removed in a future version.

This script has been replaced by the unified training script `train_models.py` which:
- Supports multiple models from configuration
- Has better error handling
- Uses centralized utilities (utils/)
- Provides more comprehensive metrics
- Supports both GridSearchCV and RandomizedSearchCV

Please use `train_models.py` instead:
    python src/train_models.py config/models_config.yaml

To enable Lasso regression in train_models.py, ensure 'lasso_regression' is enabled in the config file.
"""

import warnings
warnings.warn(
    "LinearRegressionLasso.py is deprecated. Use train_models.py instead. "
    "See the module docstring for details.",
    DeprecationWarning,
    stacklevel=2
)

import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
import pickle
from pathlib import Path
import mlflow
import mlflow.sklearn
import os
from dotenv import load_dotenv

#Se obtiene el directorio actual del script
directorio_actual = Path(__file__).parent
#Se obtiene la ruta al directorio mlops
path_mlops = directorio_actual.parent
#ruta a los datos de entrenamiento y prueba
path_df_train = f'{path_mlops}/data/processed/train.csv'
path_df_valid = f'{path_mlops}/data/processed/valid.csv'


df_train = pd.read_csv(path_df_train)
df_valid = pd.read_csv(path_df_valid)

# Carga variables de entorno desde .env (si existe) y establece el tracking URI
load_dotenv()
if os.getenv("MLFLOW_TRACKING_URI"):
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))

mlflow.set_experiment("bs_models")
mlflow.sklearn.autolog(log_models=True)

# Separa características y objetivo en df_train
X_train = df_train.drop(columns=['cnt'], axis=1)
y_train = df_train.pop('cnt')

# Definimos la malla de parámetros para el Grid Search para Lasso
param_grid_lasso = {
    'alpha': [0.001, 0.01, 0.1, 1, 10, 100],
    'fit_intercept': [True, False],
    'copy_X': [True, False],
    'max_iter': [1000, 5000, 10000]
}

#Se inicializa el modelo Lasso
modelo_lasso = Lasso(random_state = 200)

#Se inicializa GridSearchCV para Lasso
grid_search_lasso = GridSearchCV(estimator=modelo_lasso, param_grid=param_grid_lasso,
                                 scoring={'neg_root_mean_squared_error': 'neg_root_mean_squared_error',
                                          'neg_mean_absolute_error': 'neg_mean_absolute_error'},
                                 refit='neg_root_mean_squared_error', cv=5, verbose=1, n_jobs=-1)


# Imputamos los valores faltantes con la mediana antes de ajustar el Grid Search
for col in ['yr', 'hr', 'holiday', 'workingday']:
    if col in X_train.columns:
        X_train[col] = X_train[col].fillna(X_train[col].median())

with mlflow.start_run(run_name="Lasso", tags={"stage": "train", "model": "lasso", "baseline": "false"}):
    # Se ajusta GS a los datos de entrenamiento
    grid_search_lasso.fit(X_train, y_train)

    # Obtenemos los mejores parámetros y el mejor puntaje
    print("Mejores parámetros encontrados para Lasso: ", grid_search_lasso.best_params_)
    print("Mejor RMSE encontrado para Lasso: ", -grid_search_lasso.best_score_)

    # Evalúa el modelo en el conjunto de validación
    # Separa características y objetivo en df_valid
    y_valid = df_valid['cnt']
    X_valid = df_valid.drop('cnt', axis=1)


    # Imputamos valores faltantes en el conjunto de validación
    for col in ['yr', 'hr', 'holiday', 'workingday']:
        if col in X_valid.columns:
            X_valid[col] = X_valid[col].fillna(X_valid[col].median())


    # Hacemos predicción en el conjunto de validación.
    yhat_valid = grid_search_lasso.predict(X_valid)

    # Calculamos e imprimimos RMSE y MAE en el conjunto de validación
    rmse_valid = np.sqrt(mean_squared_error(y_valid, yhat_valid))
    mae_valid = mean_absolute_error(y_valid, yhat_valid)

    print(f"RMSE on validation set: {rmse_valid:.2f}")
    print(f"MAE on validation set: {mae_valid:.2f}")

    # Log de métricas de validación
    mlflow.log_metric("rmse_valid", rmse_valid)
    mlflow.log_metric("mae_valid", mae_valid)

    filename = f'{path_mlops}/models/modelo_lasso.pickle'
    pickle.dump(grid_search_lasso.best_estimator_, open(filename, 'wb'))

    # Guardamos el modelo entrenado usando pickle
    print(f"Modelo guardado en: {filename}")

