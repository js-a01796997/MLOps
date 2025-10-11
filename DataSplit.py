from pathlib import Path
import pandas as pd
from pandas.api.types import CategoricalDtype
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

#Se obtiene el directorio actual del script
path_mlops = Path(__file__).parent
#ruta a bike_sharing_cleaned
file_path = path_mlops/'data'/'processed'/'bike_sharing_cleaned.csv'
df = pd.read_csv(file_path)
print(f"La info del DataFrame: \n\n {df.info()}")

print("Se cambiará el tipo de dato de las columnas season, mnth, weekday y weathersit a categóricas.")

cat_cols   = ["season", "mnth", "weekday", "weathersit"]   # categóricas
int_cols   = ["yr", "holiday", "workingday", "cnt", "registered", "casual","instant","mixed_type_col"]        # enteras
float_cols = ["temp", "atemp", "hum", "windspeed"]         # continuas

#Convertir a numérico correctamente
cols = [c for c in (cat_cols + int_cols + float_cols) if c in df.columns]
df[cols] = df[cols].apply(pd.to_numeric, errors="coerce")

season_dtype     = CategoricalDtype(categories=[1,2,3,4], ordered=False)
mnth_dtype       = CategoricalDtype(categories=list(range(1,13)), ordered=False)
weekday_dtype    = CategoricalDtype(categories=list(range(0,7)), ordered=False)
weathersit_dtype = CategoricalDtype(categories=[1,2,3,4], ordered=False)

dtype_map = {
    "season": season_dtype,
    "mnth": mnth_dtype,
    "weekday": weekday_dtype,
    "weathersit": weathersit_dtype,
}

for c, ctype in dtype_map.items():
    if c in df.columns:
        df[c] = df[c].astype(ctype)


for c in int_cols:
    if c in df.columns:
        df[c] = df[c].astype("Int64")


for c in float_cols:
    if c in df.columns:
       df[c] = df[c].astype("float64")

print(f"La nueva información del DataFrame: \n\n {df.info()}")

df_new = df.drop(['instant','casual','registered','dteday'], axis=1)
print("\nSe eliminaron las columnas instant, casual, registered y dteday por ser redundantes para el modelo")
#Convertimos a números a las variables categóricas
df_new = pd.get_dummies(df_new, drop_first=True)
#Reescalaremos las variables
scaler = MinMaxScaler()
df_new[['temp', 'atemp', 'hum', 'windspeed','cnt']] = scaler.fit_transform(df_new[['temp', 'atemp', 'hum', 'windspeed','cnt']])
print("\nSe escalaron las variables temp, atemp, hum, windspeed y cnt")
# SEPARACIÓN DE LOS DATOS
df_train, df_vt = train_test_split(df_new, train_size = 0.70, test_size = 0.30, random_state = 333)
df_valid, df_test = train_test_split(df_vt, train_size = 0.50, test_size = 0.50, random_state = 333)

print(f"\nEl tamaño del set de entrenamiento es: {df_train.shape}")
print(f"\nEl tamaño del set de validación es: {df_valid.shape}")
print(f"\nEl tamaño del set de prueba es: {df_test.shape}")

#Guardamos los datos
train_path = path_mlops/'data'/'processed'/'train.csv'
valid_path = path_mlops/'data'/'processed'/'valid.csv'
test_path = path_mlops/'data'/'processed'/'test.csv'
df_train.to_csv(train_path, index=False)
df_valid.to_csv(valid_path, index=False)
df_test.to_csv(test_path, index=False) 
print("\nSe guardaron los datasets en la carpeta data/processed")