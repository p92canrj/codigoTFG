import pandas as pd
import argparse
import os

def process_to_areg_only(input_path, output_path):
    print(f"Leyendo dataset desde {input_path}...")
    df = pd.read_csv(input_path)
    
    if 'date' in df.columns:
        print("Extrayendo year, month, day, hour, minute desde la columna 'date'...")
        df['date'] = pd.to_datetime(df['date'])
        
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['day'] = df['date'].dt.day
        df['hour'] = df['date'].dt.hour
        df['minute'] = df['date'].dt.minute
    else:
        print("Advertencia: No se encontró la columna 'date'.")

    # Identificar columnas a mantener (year, month, day, hour, minute y características auto-regresivas)
    cols_to_keep = ['year', 'month', 'day', 'hour', 'minute']
    
    for col in df.columns:
        if col.startswith('roll_proportion') or col.startswith('lag_'):
            cols_to_keep.append(col)
            
    # Renombrar visibilidad_ordinal a vis
    if 'visibilidad_ordinal' in df.columns:
        df = df.rename(columns={'visibilidad_ordinal': 'vis'})
        
        # Opcional: convertir a entero si no tiene decimales
        try:
            df['vis'] = df['vis'].astype('Int64')
        except:
            pass
            
        cols_to_keep.append('vis')
        
    final_cols = [c for c in cols_to_keep if c in df.columns]
    
    df_areg = df[final_cols]
    
    print(f"Escribiendo el dataset 'areg_only' en {output_path}...")
    
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    df_areg.to_csv(output_path, index=False)
    print("¡Proceso completado exitosamente!")

if __name__ == "__main__":
    
    input_file = r"../forecasting/LEST_sensors_forecast.csv"
    output_file = r"LEST_sensors_areg_only.csv"
    
    process_to_areg_only(input_file, output_file)
