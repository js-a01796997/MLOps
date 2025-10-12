import numpy as np

def clean_weather_var(df, col_name, numeric_col):
  out_of_range_mask = df[col_name] > 1
  print(f" cantidad de nan", df[col_name].isna().sum())
  print(f" cantidad de valores fuera de rango para {col_name}", out_of_range_mask.sum())
  # Los valores fuera de rango se convierten en nan
  df.iloc[out_of_range_mask, numeric_col] = np.nan
  # Tomamos el siguiente valor no nan para rellenar vacios
  df[col_name] = df[col_name].bfill()

def to_int(df, col):
  df[col] = df[col].astype('Int64')

def fix_hour(g):
  hr_col = 0
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


# Function to detect outliers using IQR method
def detect_outliers(df, column_name, iqr_multiplier=1.5):
    """
    Detect outliers in a column using the IQR method.
    Returns the index of outlier rows.
    
    Parameters:
    - df: DataFrame
    - column_name: Column to analyze
    - iqr_multiplier: Multiplier for IQR (default=1.5)
    """
    Q1 = df[column_name].quantile(0.25)
    Q3 = df[column_name].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - iqr_multiplier * IQR
    upper_bound = Q3 + iqr_multiplier * IQR
    
    outlier_mask = (df[column_name] < lower_bound) | (df[column_name] > upper_bound)
    outlier_indices = df[outlier_mask].index
    
    print(f"\nColumn: {column_name} (IQR multiplier: {iqr_multiplier})")
    print(f"  Q1: {Q1:.2f}, Q3: {Q3:.2f}, IQR: {IQR:.2f}")
    print(f"  Lower bound: {lower_bound:.2f}, Upper bound: {upper_bound:.2f}")
    print(f"  Number of outliers: {len(outlier_indices)} ({len(outlier_indices)/len(df)*100:.2f}%)")
    
    return outlier_indices
