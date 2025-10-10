import numpy as np

def clean_weather_var(df, col_name, numeric_col):
  out_of_range_mask = df[col_name] > 1
  print(f" cantidad de nan", df[col_name].isna().sum())
  print(f" cantidad de valores fuera de rango para {col_name}", out_of_range_mask.sum())
  # Los valores fuera de rango se convierten en nan
  df.iloc[out_of_range_mask, numeric_col] = np.nan
  # Tomamos el siguiente valor no nan para rellenar vacios
  df[col_name] = df[col_name].bfill()
