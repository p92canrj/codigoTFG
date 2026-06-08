import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

datasets = [
    # '../datasets/LEVX_sensors_areg_only.csv',
    # '../datasets/LEST_sensors_areg_only.csv',
    '../datasets/LEVX_sensors_areg_era5.csv',
    '../datasets/LEST_sensors_areg_era5.csv'
    # '../datasets/LEVX_era5.csv',
    # '../datasets/LEST_era5.csv',
    # '../datasets/LEST_sensors.csv',
    # '../datasets/LEVX_sensors.csv'
]

out_dir = 'matrices_correlacion'
os.makedirs(out_dir, exist_ok=True)

for ds in datasets:
    if os.path.exists(ds):
        print(f"Procesando {ds}...")
        df = pd.read_csv(ds)
        
        # Eliminamos las variables temporales básicas y la variable objetivo
        cols_to_drop = ['year', 'month', 'day', 'hour', 'minute', 'visibilidad_ordinal']
        df_inputs = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors='ignore')
        
        # Renombramos las columnas de roll_proportion para que no se alarguen tanto los ejes
        import re
        df_inputs.columns = [re.sub(r'roll_proportion_(\d+)_class_(\d+)\.0', r'roll_pr_\1_cl\2', c) for c in df_inputs.columns]
        
        corr = df_inputs.corr()
        
        # Para datasets con muchas variables, ajustamos el tamaño de la figura y de las fuentes
        if 'areg_era5' in ds or 'areg_only' in ds:
            plt.figure(figsize=(36, 28))
            annot_size = 32
            title_size = 60
            tick_size = 30
            cbar_tick_size = 35
        else:
            plt.figure(figsize=(18, 14))
            annot_size = 30
            title_size = 40
            tick_size = 24
            cbar_tick_size = 28
        
        # Generar heatmap
        ax = sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1, 
                    linewidths=.5, cbar_kws={"shrink": .8}, annot_kws={"size": annot_size})
        
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
        
        plt.title(f'Matriz de Correlación - {title_suffix}', fontsize=title_size, pad=20)
        plt.xticks(rotation=45, ha='right', fontsize=tick_size)
        plt.yticks(rotation=0, fontsize=tick_size)
        plt.tight_layout()
        
        out_base = os.path.join(out_dir, 'matriz_' + ds_name)
        out_path_png = out_base + '.png'
        out_path_pdf = out_base + '.pdf'
        
        plt.savefig(out_path_png, dpi=300)
        plt.savefig(out_path_pdf, bbox_inches='tight')
        plt.close()
        print(f"Guardado en {out_path_png} y {out_path_pdf}")
    else:
        print(f"No se encontró el dataset: {ds}")
