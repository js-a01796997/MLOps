import pandas as pd
import numpy as np
from pathlib import Path
from clean_utils import clean_weather_var

#Se obtiene el directorio actual del script
directorio_data_cleaning = Path(__file__).parent
#Se obtiene el directorio padre (MLOps)
path_mlops = directorio_data_cleaning.parent
#ruta a bike_sharing_modified
file_path = path_mlops/'data'/'raw'/'bike_sharing_modified.csv'
bike_sharing = pd.read_csv(file_path)
#print(bike_sharing.head())

print("\n1/14 Información del DataFrame antes de la limpieza:\n")
print(bike_sharing.info())
#Columnas que se van a convertir las columnas a tipo int
to_int_cols = ['instant','season','yr','mnth','hr','holiday','weekday','workingday','weathersit','casual','registered','cnt','mixed_type_col']
# Columnas que se van a convertir las columnas a tipo float
to_float_cols = ['temp','atemp','hum','windspeed']

#Verifica si hay valores no numéricos en las columnas que se van a convertir a numéricas
cols = to_float_cols + to_int_cols
all_non_numeric_values = []
for col in cols:
  #Guarda en una lista los valores únicos de la columna que se definen en True cuando son nulos al tratar de convertirlos a numéricos
  non_numeric_values = bike_sharing[pd.to_numeric(bike_sharing[col], errors='coerce').isna()][col].unique()
  all_non_numeric_values.extend(non_numeric_values)
  #if len(non_numeric_values) > 0:
    #print(f"Column '{col}' contains the following non-numeric values: {non_numeric_values}")

all_non_numeric_values = list(set(all_non_numeric_values))
print(f'\n2/14 En las columnas: {cols}, \nse encontraron los valores no numéricos: {all_non_numeric_values}\n')

#Se reemplazan todos los valores no numéricos
for col in cols:
    bike_sharing[col] = bike_sharing[col].replace(all_non_numeric_values, np.nan)

#Se convierten a numéricos las columnas que deberían serlo
bike_sharing[to_int_cols] = bike_sharing[to_int_cols].astype(float).astype('Int64')
bike_sharing[to_float_cols] = bike_sharing[to_float_cols].astype(float)

#Finalmente se convierte la columna dteday a tipo fecha
bike_sharing['dteday'] = pd.to_datetime(bike_sharing['dteday'].str.strip(), format='%Y-%m-%d', errors='coerce')
print("\n3/14 Se convirtió la columna 'dteday' a tipo fecha.\n")

#CORRIGE SEASON

#Guarda los valores inválidos de la columna season
invalid_seasons = bike_sharing[~bike_sharing['season'].isin([1, 2, 3, 4])]['season'].unique()

# Itera en cada renglón del dataframe
for index, row in bike_sharing.iterrows():
    #Revisa si el valor está dentro de los valores inválidos
    if row['season'] in invalid_seasons:
        #Busca otros renglones con la misma fecha y un valor válido de season
        valid_season_rows = bike_sharing[(bike_sharing['dteday'] == row['dteday']) & (bike_sharing['season'].isin([1, 2, 3, 4]))]
        #Si hay un rebnglón con la misma fecha y un valor válido de season, lo remplaza por el valor vaálido
        if not valid_season_rows.empty:
            bike_sharing.loc[index, 'season'] = valid_season_rows.iloc[0]['season']

print('\n4/14 Se corrigió el ruido de la variable season')

#CORRIGE YR

#Se genera una variabl auxiliar para saber el año del registro
bike_sharing['year_aux'] = bike_sharing['dteday'].dt.year

#Guarda los valores válidos de la columna yr
valid_yr = [0,1]

# Itera en cada renglón del dataframe
for index, row in bike_sharing.iterrows():
  # Si el valor de yr no está en los válidos y no es nulo, revisa el valor que registra la columna dteday. Le asigna 0 si es 2011 o 1 si es 2012
  if pd.notna(row['yr']) and row['yr'] not in valid_yr:
    if pd.notna(row['year_aux']):
      if row['year_aux'] == 2011:
        bike_sharing.loc[index, 'yr'] = 0
      elif row['year_aux'] == 2012:
        bike_sharing.loc[index, 'yr'] = 1
      else:
        bike_sharing.loc[index, 'yr'] = np.nan
  elif pd.isna(row['yr']): # Si el valor original es NAN, trata de inferirlo de year_aux
    if pd.notna(row['year_aux']):
      if row['year_aux'] == 2011:
        bike_sharing.loc[index, 'yr'] = 0
      elif row['year_aux'] == 2012:
        bike_sharing.loc[index, 'yr'] = 1
      else:
        bike_sharing.loc[index, 'yr'] = np.nan
  else: # Si el valor es válido y no es NAN, lo conserva
    bike_sharing.loc[index, 'yr'] = row['yr']
#Elimina la columna auxiliar
bike_sharing = bike_sharing.drop('year_aux', axis=1)

print('\n5/14 Se corrigió el ruido de la variable yr')

#CORRIGE MNTH

# Guarda los valores inválidos de mnth
invalid_mnth = bike_sharing[~bike_sharing['mnth'].isin(range(1, 13))]['mnth'].unique()

# Itera en cada renglón del dataframe
for index, row in bike_sharing.iterrows():
    #Revisa si el valor de mnth está en la lista de valores inválidos
    if row['mnth'] in invalid_mnth:
        # Si el valor no es NAN, remplaza el valor de mes de la col dteday
        if pd.notna(row['dteday']):
            bike_sharing.loc[index, 'mnth'] = row['dteday'].month

        else :
          bike_sharing.loc[index, 'mnth'] = np.nan

print('\n6/14 Se corrigió el ruido de la variable mnth')

#CORRIGE WORKINGDAY

#Guarda los valores únicos que no sean cero y uno
invalid_workingdays = bike_sharing[~bike_sharing['workingday'].isin([0, 1])]['workingday'].unique()

# Itera en los renglones del df
for index, row in bike_sharing.iterrows():
    #Revisa si el valor de workingdays es inválido
    if row['workingday'] in invalid_workingdays:
        #Revisa si no es nulo
        if pd.notna(row['dteday']):
            #Revisa si es fin de semana o dia festivo
            if row['dteday'].weekday() in [5, 6] or (pd.notna(row['holiday']) and row['holiday'] == 1):
                bike_sharing.loc[index, 'workingday'] = 0
            else:
                bike_sharing.loc[index, 'workingday'] = 1
        else:
            #Si es NAN, lo deja en NAN
            bike_sharing.loc[index, 'workingday'] = np.nan

#Los convierte a valores enteros
bike_sharing['workingday'] = pd.to_numeric(bike_sharing['workingday'], errors='coerce').astype('Int64')

print('\n7/14 Se corrigió el ruido de la variable workingday')

#CORRIGE WEEKDAY

#Guarda los valores únicos de weekday
invalid_weekdays = bike_sharing[~bike_sharing['weekday'].isin(range(7))]['weekday'].unique()

# Itera en los renglones del df
for index, row in bike_sharing.iterrows():
    #Revisa si está en la lista de valores inválidos
    if row['weekday'] in invalid_weekdays:
        #Revisa si no es nulo
        if pd.notna(row['dteday']):
            #Remplaza el valor por el día de la semana
            bike_sharing.loc[index, 'weekday'] = row['dteday'].weekday()
        else:
            #Si es NAN, se queda en NAN
            bike_sharing.loc[index, 'weekday'] = np.nan

#Convierte a enteros
bike_sharing['weekday'] = pd.to_numeric(bike_sharing['weekday'], errors='coerce').astype('Int64')

print('\n8/14 Se corrigió el ruido de la variable weekday')

#CORRIGE TEMP
clean_weather_var(bike_sharing, 'temp', 10)
print('\n9/14 Se corrigió el ruido de la variable temp')

#CORRIGE ATEMP
clean_weather_var(bike_sharing, 'atemp', 11)
print('\n10/14 Se corrigió el ruido de la variable atemp')

#CORRIGE HUM
clean_weather_var(bike_sharing, 'hum', 12)
print('\n11/14 Se corrigió el ruido de la variable hum')

#CORRIGE WINDSPEED
clean_weather_var(bike_sharing, 'windspeed', 13)
print('\n12/17 Se corrigió el ruido de la variable windspeed')

print("\n13/14 Información del DataFrame después de la limpieza:\n")
print(bike_sharing.info())
#Se guarda el DataFrame limpio en un nuevo archivo CSV
output_file_path = path_mlops/'data'/'processed'/'bike_sharing_cleaned.csv'
bike_sharing.to_csv(output_file_path, index=False)

print(f"\n14/14 El DataFrame limpio se ha guardado en: {output_file_path}\n")