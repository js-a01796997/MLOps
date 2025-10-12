import pickle
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

#Se obtiene el directorio actual del script
directorio_actual = Path(__file__).parent
#Se obtiene la ruta al directorio mlops
path_mlops = directorio_actual.parent
#ruta a los datos de prueba
path_test = f'{path_mlops}/data/processed/test.csv'
df_test = pd.read_csv(path_test)

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
    'Linear Regression': lr, 
    'Lasso Regression': lasso,
    'Ridge Regression': ridge
}

results = {}
for name, model in models.items():
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    results[name] = {'RMSE': rmse, 'MAE': mae}

# Convertimos los resultados a un DataFrame para una mejor visualización
results_df = pd.DataFrame(results).T
print(results_df)