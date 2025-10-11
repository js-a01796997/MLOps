import pandas as pd
import numpy as np
from pathlib import Path
from clean_utils import clean_weather_var

#Se obtiene el directorio actual del script
directorio_data_cleaning = Path(__file__).parent
#Se obtiene el directorio padre (MLOps)
path_mlops = directorio_data_cleaning.parent
#ruta a bike_sharing_modified
file_path = path_mlops/'data'/'processed'/'bike_sharing_cleaned.csv'
bike_sharing = pd.read_csv(file_path)


def to_int(df, col):
  df[col] = df[col].astype('Int64')

def fix_hour(g):
  hr_col = 0
  dteday_col = 1
  n = len(g)
  if n == 24:
    # si existen los 24 registros por fecha, existe un registro para cada hora
    g['hr'] = np.arange(24)
  if n <= 23:
    # si son menos de 24 registros no podemos asegurar cual hora es cual
    # así que no se hacen modificaciones
    pass
  if n == 25:
    # si existen 25 elementos, ponemos en nan el último
    g['hr'] = np.arange(25)
    g.iloc[-1, hr_col] = np.nan
  return g


# Se tomaron los dias festivos basados en:
# https://www.timeanddate.com/holidays/us/2011
# https://www.timeanddate.com/holidays/us/2012
# Dado que el dataset es de DC, se agregaron solo los días festivos que caen entre semana
# y que son o Federales o de DC
holidays = [
  '2011-01-17',
  '2011-02-21',
  '2011-04-15',
  '2011-05-30',
  '2011-07-04',
  '2011-09-05',
  '2011-10-10',
  '2011-11-11',
  '2011-11-24',
  '2011-12-26',
  '2012-01-02',
  '2012-01-16',
  '2012-02-20',
  '2012-04-16',
  '2012-05-28',
  '2012-07-04',
  '2012-09-03',
  '2012-10-08',
  '2012-11-12',
  '2012-11-22',
  '2012-12-25',
]

# Para holidays solo se toman como holidays los dias de lista obtenida anteriormente
holidays = [pd.to_datetime(date) for date in holidays]
holidays_mask = bike_sharing['dteday'].isin(holidays)
bike_sharing['holiday'] = holidays_mask.astype(int)
print('Se corrigió el ruido de la variable holidays')

# Para la hora se borran valores duplicados en lcas variables relacionadas a la fecha
# Se remueven los valores que salen del rango y se aplica un ajuste basado en las lecturas de cada día
bike_sharing = bike_sharing.drop_duplicates(subset=['dteday', 'season', 'yr', 'mnth', 'hr', 'holiday', 'weekday', 'workingday'])
hour_mask_more_than_23 = bike_sharing["hr"] > 23
bike_sharing.loc[hour_mask_more_than_23, 'hr'] = np.nan
new_df = bike_sharing.groupby('dteday', group_keys = False)[['hr', 'dteday']].apply(fix_hour)
bike_sharing['hr'] = new_df['hr']
print('Se corrigió el ruido de la variable hr')

# para weathersit se asume que no hay cambios drásticos entre horas
# por lo que los valores nan se rellean con la siguiente lectura válida
weathersit_invalid_mask = bike_sharing['weathersit'] > 4
bike_sharing.loc[weathersit_invalid_mask, 'weathersit'] = np.nan
bike_sharing['weathersit'] = bike_sharing['weathersit'].bfill()
print('Se corrigió el ruido de la variable weathersit')

to_int(bike_sharing, 'instant')
to_int(bike_sharing, 'season')
to_int(bike_sharing, 'yr')
to_int(bike_sharing, 'mnth')
to_int(bike_sharing, 'hr')
to_int(bike_sharing, 'holiday')
to_int(bike_sharing, 'weekday')
to_int(bike_sharing, 'workingday')
to_int(bike_sharing, 'weathersit')

print(bike_sharing.info())
#Se guarda el DataFrame limpio en un nuevo archivo CSV
output_file_path = path_mlops/'data'/'processed'/'bike_sharing_cleaned.csv'
bike_sharing.to_csv(output_file_path, index=False)
