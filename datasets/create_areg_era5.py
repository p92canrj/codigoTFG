import pandas as pd
import os

def create_areg_era5(forecast_path, era5_path, output_path):
    print("Leyendo dataset de forecast...")
    df_forecast = pd.read_csv(forecast_path)
    
    if 'date' in df_forecast.columns:
        print("Extrayendo year, month, day, hour, minute desde la columna 'date'...")
        df_forecast['date'] = pd.to_datetime(df_forecast['date'])
        
        df_forecast['year'] = df_forecast['date'].dt.year
        df_forecast['month'] = df_forecast['date'].dt.month
        df_forecast['day'] = df_forecast['date'].dt.day
        df_forecast['hour'] = df_forecast['date'].dt.hour
        df_forecast['minute'] = df_forecast['date'].dt.minute

    # Renombrar visibilidad_ordinal a vis
    if 'visibilidad_ordinal' in df_forecast.columns:
        df_forecast = df_forecast.rename(columns={'visibilidad_ordinal': 'vis'})
        try:
            df_forecast['vis'] = df_forecast['vis'].astype('Int64')
        except:
            pass

    print("Leyendo dataset ERA5...")
    df_era5 = pd.read_csv(era5_path)
    
    # Conservar de ERA5
    cols_era5 = ['year', 'month', 'day', 'hour', 'minute', 'tp', 'u10', 'v10']
    df_era5 = df_era5[cols_era5]
    
    print("Combinando datasets por fecha y hora...")
    
    df_merged = pd.merge(df_forecast, df_era5, on=['year', 'month', 'day', 'hour', 'minute'], how='inner')
    
    cols_to_keep = ['year', 'month', 'day', 'hour', 'minute',
                    'temp', 'dewpt', 'press', 'wind_sin', 'wind_cos', 'wind_speed']
    
    # Añadir las autoregresivas de forecast
    for col in df_merged.columns:
        if col.startswith('roll_proportion') or col.startswith('lag_'):
            cols_to_keep.append(col)
            
    # Añadir las variables meteorológicas del ERA5 y la etiqueta final
    cols_to_keep.extend(['tp', 'u10', 'v10', 'vis'])
    
    final_cols = [c for c in cols_to_keep if c in df_merged.columns]
    
    df_final = df_merged[final_cols]
    
    print(f"Escribiendo el dataset final en {output_path}...")
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    df_final.to_csv(output_path, index=False)
    print("¡Proceso completado exitosamente!")

if __name__ == "__main__":
    
    forecast_file = r"..\forecasting\LEST_sensors_forecast.csv"
    era5_file     = r"LEST_era5.csv"
    output_file   = r"LEST_sensors_areg_era5_2.csv"
    
    create_areg_era5(forecast_file, era5_file, output_file)
