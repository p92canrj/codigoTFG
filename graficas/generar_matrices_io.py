import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

datasets = [
    '../datasets/LEVX_sensors_areg_era5.csv',
    '../datasets/LEST_sensors_areg_era5.csv',
    '../datasets/LEVX_sensors_areg_only.csv',
    '../datasets/LEST_sensors_areg_only.csv',
    '../datasets/LEVX_era5.csv',
    '../datasets/LEST_era5.csv',
    '../datasets/LEST_sensors.csv',
    '../datasets/LEVX_sensors.csv'
]

out_dir = 'matrices_correlacion_entrada_salida'
os.makedirs(out_dir, exist_ok=True)

for ds in datasets:
    if os.path.exists(ds):
        print(f"Procesando {ds}...")
        df = pd.read_csv(ds)
        
        # Identificamos cuál es la variable objetivo (a veces es 'visibilidad_ordinal', otras veces 'vis')
        target_col = 'visibilidad_ordinal'
        if target_col not in df.columns:
            if 'vis' in df.columns:
                target_col = 'vis'
            else:
                print(f"  [ERROR] No se encontró la variable objetivo en {ds}")
                continue
                
        # Excluir las variables temporales
        cols_to_drop = ['year', 'month', 'day', 'hour', 'minute']
        df_clean = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors='ignore')
        
        # Renombramos las columnas de roll_proportion para que no alarguen tanto los ejes
        import re
        df_clean.columns = [re.sub(r'roll_proportion_(\d+)_class_(\d+)\.0', r'roll_pr_\1_cl\2', c) for c in df_clean.columns]
        
        # Calcular la correlación total
        corr = df_clean.corr()
        
        # Extraer solo la columna correspondiente a la variable objetivo (y quitamos la correlación consigo misma)
        # Ordenamos los valores para que se vea claramente qué variables influyen más (positiva y negativamente)
        corr_target = corr[[target_col]].drop(index=target_col).sort_values(by=target_col, ascending=False)
        
        # Configurar tamaños dinámicos según el volumen de datos (agrandado respecto al original para mayor legibilidad)
        if 'areg_era5' in ds or 'areg_only' in ds:
            plt.figure(figsize=(21, 28)) # Figura mucho más alta para datasets grandes
            annot_size = 35
            title_size = 60
            label_size = 45
            tick_size = 35
            cbar_tick_size = 35
        else:
            plt.figure(figsize=(8, 14)) # Ajustado ligeramente el ancho
            annot_size = 30
            title_size = 40
            label_size = 30
            tick_size = 28
            cbar_tick_size = 28
            
        # Graficar
        ax = sns.heatmap(corr_target, annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1, 
                    linewidths=.5, cbar_kws={"shrink": .8}, annot_kws={"size": annot_size})
        
        # Cambiar el tamaño de los números de la barra lateral (colorbar)
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=cbar_tick_size)
        
        ds_name = os.path.basename(ds).replace('.csv', '')
        titles_map = {
            'LEST_sensors': 'Solo METAR (LEST)',
            'LEVX_sensors': 'Solo METAR (LEVX)',
            'LEST_sensors_areg_only': 'Solo areg (LEST)',
            'LEVX_sensors_areg_only': 'Solo areg (LEVX)',
            'LEST_era5': 'Solo ERA5 (LEST)',
            'LEVX_era5': 'Solo ERA5 (LEVX)',
            'LEST_sensors_areg_era5': 'Conjunto de datos completo (LEST)',
            'LEVX_sensors_areg_era5': 'Conjunto de datos completo (LEVX)'
        }
        title_suffix = titles_map.get(ds_name, ds_name)
        
        plt.title(f'Correlación de Entrada vs vis\n({title_suffix})', fontsize=title_size, pad=20)
        plt.ylabel('Variables de Entrada', fontsize=label_size)
        plt.xlabel('Correlación', fontsize=label_size)
        
        plt.xticks(fontsize=tick_size)
        plt.yticks(rotation=0, fontsize=tick_size)
        
        plt.tight_layout()
        
        out_base = os.path.join(out_dir, f'matriz_{ds_name}_in_out')
        out_path_png = out_base + '.png'
        out_path_pdf = out_base + '.pdf'
        
        plt.savefig(out_path_png, dpi=300, bbox_inches='tight')
        plt.savefig(out_path_pdf, bbox_inches='tight') # Formato PDF para LaTeX
        plt.close()
        print(f"  Guardado en {out_path_png} y {out_path_pdf}")
    else:
        print(f"No se encontró el dataset: {ds}")
