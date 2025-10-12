import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge
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

# Definimos la malla de parámetros para el Grid Search para Ridge
param_grid_ridge = {
    'alpha': [0.001, 0.01, 0.1, 1, 10, 100],
    'fit_intercept': [True, False],
    'copy_X': [True, False],
    'max_iter': [1000, 5000, 10000]
}

#Se inicializa el modelo Ridge
modelo_ridge = Ridge(random_state=200)

#Se inicializa GridSearchCV para Ridge
grid_search_ridge = GridSearchCV(estimator=modelo_ridge, param_grid=param_grid_ridge,
                                 scoring={'neg_root_mean_squared_error': 'neg_root_mean_squared_error',
                                          'neg_mean_absolute_error': 'neg_mean_absolute_error'},
                                 refit='neg_root_mean_squared_error', cv=5, verbose=1, n_jobs=-1)

# Imputamos los valores faltantes con la mediana antes de ajustar el Grid Search
for col in ['yr', 'hr', 'holiday', 'workingday']:
    if col in X_train.columns:
        X_train[col] = X_train[col].fillna(X_train[col].median())

with mlflow.start_run(run_name="Ridge", tags={"stage": "train", "model": "ridge", "baseline": "false"}):
    # Se ajusta GS a los datos de entrenamiento
    grid_search_ridge.fit(X_train, y_train)

    # Obtenemos los mejores parámetros y el mejor puntaje
    print("Best parameters for Ridge found: ", grid_search_ridge.best_params_)
    print("Best RMSE for Ridge found: ", -grid_search_ridge.best_score_)


    # Evalúa el modelo en el conjunto de validación
    # Separa características y objetivo en df_valid
    y_valid = df_valid['cnt']
    X_valid = df_valid.drop('cnt', axis=1)

    # Imputamos valores faltantes en el conjunto de validación
    for col in ['yr', 'hr', 'holiday', 'workingday']:
        if col in X_valid.columns:
            X_valid[col] = X_valid[col].fillna(X_valid[col].median())

    # Hacemos predicción en el conjunto de validación.
    yhat_valid = grid_search_ridge.predict(X_valid)

    # Calculamos e imprimimos RMSE y MAE en el conjunto de validación
    rmse_valid = np.sqrt(mean_squared_error(y_valid, yhat_valid))
    mae_valid = mean_absolute_error(y_valid, yhat_valid)

    print(f"RMSE on validation set: {rmse_valid:.2f}")
    print(f"MAE on validation set: {mae_valid:.2f}")

    # Log de métricas de validación
    mlflow.log_metric("rmse_valid", rmse_valid)
    mlflow.log_metric("mae_valid", mae_valid)

    filename = f'{path_mlops}/models/modelo_ridge.pickle'
    pickle.dump(grid_search_ridge.best_estimator_, open(filename, 'wb'))

    # Guardamos el modelo entrenado usando pickle
    print(f"Modelo guardado en: {filename}")