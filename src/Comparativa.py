import pickle
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import mlflow
import mlflow.sklearn
import os
from dotenv import load_dotenv

#Se obtiene el directorio actual del script
directorio_actual = Path(__file__).parent
#Se obtiene la ruta al directorio mlops
path_mlops = directorio_actual.parent
#ruta a los datos de prueba
path_test = f'{path_mlops}/data/processed/test.csv'
df_test = pd.read_csv(path_test)

# Carga variables de entorno desde .env (si existe) y establece el tracking URI
load_dotenv()
if os.getenv("MLFLOW_TRACKING_URI"):
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))

mlflow.set_experiment("bs_evaluation")

# Cargamos los modelos previamente entrenados
lr = pickle.load(open(f'{path_mlops}/models/modelo_LR.pickle', 'rb'))
lasso = pickle.load(open(f'{path_mlops}/models/modelo_lasso.pickle', 'rb'))
ridge = pickle.load(open(f'{path_mlops}/models/modelo_ridge.pickle', 'rb'))

y_test = df_test['cnt']
X_test = df_test.drop('cnt', axis=1)

#Imputamos valores faltantes en el conjunto de prueba
for col in ['yr', 'hr', 'holiday', 'workingday']:
    if col in X_test.columns:
        X_test[col] = X_test[col].fillna(X_test[col].median())

# Guardamos los modelos en un diccionario para facilitar la comparación
models = {
    'LinearRegression': lr, 
    'Lasso': lasso,
    'Ridge': ridge
}

results = {}
with mlflow.start_run(run_name="compare_on_test", tags={"stage": "test"}):
    for name, model in models.items():
        with mlflow.start_run(run_name=f"test_{name}", nested=True, tags={"model": name}):
            y_pred = model.predict(X_test)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            mlflow.log_metric("rmse_test", rmse)
            mlflow.log_metric("mae_test", mae)
            results[name] = {'RMSE': rmse, 'MAE': mae}

    # Convertimos los resultados a un DataFrame para una mejor visualización
    results_df = pd.DataFrame(results).T
    print(results_df)