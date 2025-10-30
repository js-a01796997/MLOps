import pandas as pd
import numpy as np
from pathlib import Path
from clean_utils import clean_weather_var, to_int, fix_hour, detect_outliers

#Se obtiene el directorio actual del script
directorio_data_cleaning = Path(__file__).parent
#Se obtiene el directorio padre (MLOps)
path_mlops = directorio_data_cleaning.parent
#ruta a bike_sharing_modified
file_path = path_mlops/'data'/'raw'/'bike_sharing_modified.csv'
bike_sharing = pd.read_csv(file_path)
#print(bike_sharing.head())

print("\n1. Información del DataFrame antes de la limpieza:\n")
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
print(f'\n2. En las columnas: {cols}, \nse encontraron los valores no numéricos: {all_non_numeric_values}')

#Se reemplazan todos los valores no numéricos
for col in cols:
    bike_sharing[col] = bike_sharing[col].replace(all_non_numeric_values, np.nan)

#Se convierten a numéricos las columnas que deberían serlo
bike_sharing[to_int_cols] = bike_sharing[to_int_cols].astype(float).astype('Int64')
bike_sharing[to_float_cols] = bike_sharing[to_float_cols].astype(float)

#Finalmente se convierte la columna dteday a tipo fecha
bike_sharing['dteday'] = pd.to_datetime(bike_sharing['dteday'].str.strip(), format='%Y-%m-%d', errors='coerce')
print("\n3. Se convirtió la columna 'dteday' a tipo fecha.")

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

print('\n4. Se corrigió el ruido de la variable season')

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

print('\n5. Se corrigió el ruido de la variable yr')

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

print('\n6. Se corrigió el ruido de la variable mnth')

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

print('\n7. Se corrigió el ruido de la variable workingday')

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

print('\n8. Se corrigió el ruido de la variable weekday')

#CORRIGE TEMP
clean_weather_var(bike_sharing, 'temp', 10)
print('\n9. Se corrigió el ruido de la variable temp')

#CORRIGE ATEMP
clean_weather_var(bike_sharing, 'atemp', 11)
print('\n10. Se corrigió el ruido de la variable atemp')

#CORRIGE HUM
clean_weather_var(bike_sharing, 'hum', 12)
print('\n11. Se corrigió el ruido de la variable hum')

#CORRIGE WINDSPEED
clean_weather_var(bike_sharing, 'windspeed', 13)
print('\n12. Se corrigió el ruido de la variable windspeed')

#BORRA MIXED_TYPE_COL
bike_sharing =  bike_sharing.drop('mixed_type_col', axis=1)
print('\n13. Se eliminó la columna mixed_type_col\n')

#CORRIGE CASUAL,REGISTERED y CNT
# Round values to integers
print("="*80)
print("ROUNDING VALUES TO INTEGERS")
print("="*80)

bike_sharing['casual'] = bike_sharing['casual'].round().astype('Int64')  # Use Int64 to handle NaN
bike_sharing['registered'] = bike_sharing['registered'].round().astype('Int64')
bike_sharing['cnt'] = bike_sharing['cnt'].round().astype('Int64')

print("All values in casual, registered, and cnt have been rounded to integers")

# Use the absolute value to avoid negative values
print("="*80)
print("USING THE ABSOLUTE VALUE TO AVOID NEGATIVE VALUES")
print("="*80)

bike_sharing['cnt'] = bike_sharing['cnt'].abs()
bike_sharing['casual'] = bike_sharing['casual'].abs()
bike_sharing['registered'] = bike_sharing['registered'].abs()

print("All values in casual, registered, and cnt have been converted to absolute values")

IQR_MULTIPLIER = 2.5

print("="*80)
print(f"OUTLIER DETECTION (IQR Multiplier: {IQR_MULTIPLIER})")
print("="*80)

casual_outliers = detect_outliers(bike_sharing, 'casual', IQR_MULTIPLIER)
registered_outliers = detect_outliers(bike_sharing, 'registered', IQR_MULTIPLIER)
cnt_outliers = detect_outliers(bike_sharing, 'cnt', IQR_MULTIPLIER)

# Identify rows where outliers appear in only one column
casual_only = set(casual_outliers) - set(registered_outliers) - set(cnt_outliers)
registered_only = set(registered_outliers) - set(casual_outliers) - set(cnt_outliers)
cnt_only = set(cnt_outliers) - set(casual_outliers) - set(registered_outliers)

print("\n" + "="*80)
print("OUTLIERS IN SINGLE COLUMNS")
print("="*80)
print(f"Outliers only in 'casual': {len(casual_only)}")
print(f"Outliers only in 'registered': {len(registered_only)}")
print(f"Outliers only in 'cnt': {len(cnt_only)}")

# Calculate medians (excluding NaN)
casual_median = bike_sharing['casual'].median()
registered_median = bike_sharing['registered'].median()
cnt_median = bike_sharing['cnt'].median()

print("\n" + "="*80)
print("REPLACING SINGLE-COLUMN OUTLIERS WITH MEDIAN")
print("="*80)
print(f"Median values:")
print(f"  casual: {casual_median}")
print(f"  registered: {registered_median}")
print(f"  cnt: {cnt_median}")

# Replace casual outliers with median
if len(casual_only) > 0:
    print(f"\nReplacing {len(casual_only)} outliers in 'casual' with median ({casual_median})")
    bike_sharing.loc[list(casual_only), 'casual'] = casual_median

# Replace registered outliers with median
if len(registered_only) > 0:
    print(f"Replacing {len(registered_only)} outliers in 'registered' with median ({registered_median})")
    bike_sharing.loc[list(registered_only), 'registered'] = registered_median

# Replace cnt outliers with median
if len(cnt_only) > 0:
    print(f"Replacing {len(cnt_only)} outliers in 'cnt' with median ({cnt_median})")
    bike_sharing.loc[list(cnt_only), 'cnt'] = cnt_median

print(f"\nTotal outliers replaced: {len(casual_only) + len(registered_only) + len(cnt_only)}")

# Calculate missing values using the relationship: casual + registered = cnt
print("="*80)
print("CALCULATING MISSING VALUES USING casual + registered = cnt")
print("="*80)

# Check NaN values before filling
casual_nan_before = bike_sharing['casual'].isna().sum()
registered_nan_before = bike_sharing['registered'].isna().sum()
cnt_nan_before = bike_sharing['cnt'].isna().sum()

print(f"Missing values before calculation:")
print(f"  casual: {casual_nan_before}")
print(f"  registered: {registered_nan_before}")
print(f"  cnt: {cnt_nan_before}")

# Fill missing cnt values where casual and registered are available
bike_sharing['cnt'] = bike_sharing['cnt'].fillna(bike_sharing['casual'] + bike_sharing['registered'])

# Fill missing casual values where registered and cnt are available
bike_sharing['casual'] = bike_sharing['casual'].fillna(bike_sharing['cnt'] - bike_sharing['registered'])

# Fill missing registered values where casual and cnt are available
bike_sharing['registered'] = bike_sharing['registered'].fillna(bike_sharing['cnt'] - bike_sharing['casual'])

# Check NaN values after filling
casual_nan_after = bike_sharing['casual'].isna().sum()
registered_nan_after = bike_sharing['registered'].isna().sum()
cnt_nan_after = bike_sharing['cnt'].isna().sum()

print(f"\nMissing values after calculation:")
print(f"  casual: {casual_nan_after}")
print(f"  registered: {registered_nan_after}")
print(f"  cnt: {cnt_nan_after}")

total_filled = (casual_nan_before - casual_nan_after) + (registered_nan_before - registered_nan_after) + (cnt_nan_before - cnt_nan_after)
print(f"\nTotal missing values filled: {total_filled}")

# Recalculate rows where casual + registered != cnt
print("="*80)
print("FIXING INCONSISTENT ROWS (casual + registered != cnt)")
print("="*80)

# Find inconsistent rows (using integer comparison, no tolerance needed)
inconsistent_mask = (bike_sharing['casual'] + bike_sharing['registered']) != bike_sharing['cnt']

print(bike_sharing[inconsistent_mask][['casual', 'registered', 'cnt']])
inconsistent_count = inconsistent_mask.sum()

print(f"Rows where casual + registered != cnt: {inconsistent_count} ({inconsistent_count/len(bike_sharing)*100:.2f}%)")

if inconsistent_count > 0:
    print(f"\nRecalculating the biggest value in each inconsistent row\n")

    # Track what was recalculated
    recalc_casual = 0
    recalc_registered = 0
    recalc_cnt = 0

    for idx in bike_sharing[inconsistent_mask].index:
        casual_val = bike_sharing.loc[idx, 'casual']
        registered_val = bike_sharing.loc[idx, 'registered']
        cnt_val = bike_sharing.loc[idx, 'cnt']

        # Find which value is the biggest
        max_val = max(casual_val, registered_val, cnt_val)

        if casual_val == max_val:
            # Recalculate casual = cnt - registered
            bike_sharing.loc[idx, 'casual'] = bike_sharing.loc[idx, 'cnt'] - bike_sharing.loc[idx, 'registered']
            recalc_casual += 1
        elif registered_val == max_val:
            # Recalculate registered = cnt - casual
            bike_sharing.loc[idx, 'registered'] = bike_sharing.loc[idx, 'cnt'] - bike_sharing.loc[idx, 'casual']
            recalc_registered += 1
        else:
            # Recalculate cnt = casual + registered
            bike_sharing.loc[idx, 'cnt'] = bike_sharing.loc[idx, 'casual'] + bike_sharing.loc[idx, 'registered']
            recalc_cnt += 1
    
    print(f"Recalculation summary:")
    print(f"  casual recalculated: {recalc_casual}")
    print(f"  registered recalculated: {recalc_registered}")
    print(f"  cnt recalculated: {recalc_cnt}")
    
    # Verify all rows are now consistent
    still_inconsistent = ((bike_sharing['casual'] + bike_sharing['registered']) != bike_sharing['cnt']).sum()
    
    print(f"\nVerification after recalculation:")
    print(f"  Inconsistent rows remaining: {still_inconsistent}")
    
    if still_inconsistent == 0:
        print("\nAll rows now follow the relationship: casual + registered = cnt")
    else:
        print("\nWarning: Some inconsistent rows remain!")
else:
    print("\nAll rows already follow the relationship: casual + registered = cnt")


print(bike_sharing[inconsistent_mask][['casual', 'registered', 'cnt']])
# Delete rows with NaN values in casual, registered, or cnt columns
rows_before = len(bike_sharing)
bike_sharing = bike_sharing.dropna(subset=['casual', 'registered', 'cnt'])
rows_after = len(bike_sharing)

rows_deleted = rows_before - rows_after
print(f"Rows before deletion: {rows_before}")
print(f"Rows after deletion: {rows_after}")
print(f"Rows deleted: {rows_deleted} ({rows_deleted/rows_before*100:.2f}%)")

# Search for negative values
print("="*80)
print("CHECKING FOR NEGATIVE VALUES")
print("="*80)

negative_casual = (bike_sharing['casual'] < 0).sum()
negative_registered = (bike_sharing['registered'] < 0).sum()
negative_cnt = (bike_sharing['cnt'] < 0).sum()

print(f"Negative values found:")
print(f"  casual: {negative_casual}")
print(f"  registered: {negative_registered}")
print(f"  cnt: {negative_cnt}")

total_negatives = negative_casual + negative_registered + negative_cnt
print(f"\nTotal negative values: {total_negatives}")

if total_negatives > 0:
    negative_mask = (bike_sharing['casual'] < 0) | (bike_sharing['registered'] < 0) | (bike_sharing['cnt'] < 0)
    print(f"\nRows with negative values:")
    print(bike_sharing[negative_mask][['instant', 'dteday', 'casual', 'registered', 'cnt']])
else:
    print("\nNo negative values found!")

# Delete rows with at least one negative value
print("="*80)
print("DELETING ROWS WITH NEGATIVE VALUES")
print("="*80)

rows_before = len(bike_sharing)

# Create mask for rows with any negative value in casual, registered, or cnt
negative_mask = (bike_sharing['casual'] < 0) | (bike_sharing['registered'] < 0) | (bike_sharing['cnt'] < 0)
rows_with_negatives = negative_mask.sum()

# Drop rows with negative values
bike_sharing = bike_sharing[~negative_mask]

rows_after = len(bike_sharing)
rows_deleted = rows_before - rows_after

print(f"Rows before deletion: {rows_before}")
print(f"Rows after deletion: {rows_after}")
print(f"Rows deleted: {rows_deleted} ({rows_deleted/rows_before*100:.2f}%)")


print('\n14. Se corrigió el ruido de la variables: casual, registered y count')

# CORRIGE HOLIDAYS
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
print('\n15. Se corrigió el ruido de la variable holidays')

# CORRIGE HR
# Para la hora se borran valores duplicados en lcas variables relacionadas a la fecha
# Se remueven los valores que salen del rango y se aplica un ajuste basado en las lecturas de cada día
bike_sharing = bike_sharing.drop_duplicates(subset=['dteday', 'season', 'yr', 'mnth', 'hr', 'holiday', 'weekday', 'workingday'])
hour_mask_more_than_23 = bike_sharing["hr"] > 23
bike_sharing.loc[hour_mask_more_than_23, 'hr'] = np.nan
new_df = bike_sharing.groupby('dteday', group_keys = False)[['hr', 'dteday']].apply(fix_hour)
bike_sharing['hr'] = new_df['hr']
print('\n16. Se corrigió el ruido de la variable hr')

# CORRIGE weathersit
# para weathersit se asume que no hay cambios drásticos entre horas
# por lo que los valores nan se rellean con la siguiente lectura válida
weathersit_invalid_mask = bike_sharing['weathersit'] > 4
bike_sharing.loc[weathersit_invalid_mask, 'weathersit'] = np.nan
bike_sharing['weathersit'] = bike_sharing['weathersit'].bfill()
print('\n17. Se corrigió el ruido de la variable weathersit')

# Conviertes columns que son enteras
to_int(bike_sharing, 'instant')
to_int(bike_sharing, 'season')
to_int(bike_sharing, 'yr')
to_int(bike_sharing, 'mnth')
to_int(bike_sharing, 'hr')
to_int(bike_sharing, 'holiday')
to_int(bike_sharing, 'weekday')
to_int(bike_sharing, 'workingday')
to_int(bike_sharing, 'weathersit')
print('\n18. Se convirtieron columnas enteras')

print("\n19. Información del DataFrame después de la limpieza:\n")
print(bike_sharing.info())

#Se guarda el DataFrame limpio en un nuevo archivo CSV
output_file_path = path_mlops/'data'/'processed'/'bike_sharing_cleaned.csv'
bike_sharing.to_csv(output_file_path, index=False)

print(f"\n20. El DataFrame limpio se ha guardado en: {output_file_path}\n")