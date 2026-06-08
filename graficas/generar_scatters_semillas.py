import os
import pandas as pd
from aeon.visualisation import plot_pairwise_scatter
import matplotlib.pyplot as plt

# Diccionario para renombrar los modelos
nombres_modelos = {
    'lightgbmclassifier_fast': 'LightGBM',
    'logisticat': 'LogisticAT',
    'logisticregressor': 'Reg. Logística',
    'nnop': 'NNOP',
    'nnpom': 'NNPOM',
    'ordinaldecomposition': 'Descomp. Ordinal'
}

# Diccionario para renombrar los datasets en los títulos
nombres_datasets = {
    'LEST_sensors': 'Solo METAR (LEST)',
    'LEVX_sensors': 'Solo METAR (LEVX)',
    'LEST_sensors_areg_only': 'Solo areg (LEST)',
    'LEVX_sensors_areg_only': 'Solo areg (LEVX)',
    'LEST_era5': 'Solo ERA5 (LEST)',
    'LEVX_era5': 'Solo ERA5 (LEVX)',
    'LEST_sensors_areg_era5': 'Conjunto de datos completo (LEST)',
    'LEVX_sensors_areg_era5': 'Conjunto de datos completo (LEVX)'
}

ruta_excel = r'../prepared_results_def/20260605_121221_recap_FINAL.xlsx'
nombre_pestana = 'MS'

df_metric = pd.read_excel(ruta_excel, sheet_name=nombre_pestana, index_col=0, header=0)

dataset_elegido = 'LEVX_sensors_areg_era5'

sufijo = f"_{dataset_elegido}"
columnas_dataset = [col for col in df_metric.columns if col.endswith(sufijo)]
df_dataset = df_metric[columnas_dataset].copy()

df_dataset.columns = [col.replace(sufijo, "") for col in df_dataset.columns]
# Renombramos usando los nombres que he puesto arriba
df_dataset.rename(columns=nombres_modelos, inplace=True)

# 4. Definimos qué dos modelos queremos enfrentar cara a cara
modelo_1_raw = "nnpom"
modelo_2_raw = "lightgbmclassifier_fast"

modelo_1 = nombres_modelos.get(modelo_1_raw, modelo_1_raw)
modelo_2 = nombres_modelos.get(modelo_2_raw, modelo_2_raw)

# 5. Comprobamos si es una métrica donde "menor es mejor"
lower_better = True if nombre_pestana in ['AMAE', 'MMAE', 'MAE'] else False

# Nombre que he puesto arriba del dataset
dataset_nombre_amigable = nombres_datasets.get(dataset_elegido, dataset_elegido)

# 6. Generamos el Scatter Plot
if modelo_1 in df_dataset.columns and modelo_2 in df_dataset.columns:
    fig, ax = plot_pairwise_scatter(
        df_dataset[modelo_1].values, 
        df_dataset[modelo_2].values,
        modelo_1,
        modelo_2,
        metric=nombre_pestana,
        lower_better=lower_better,
        best_on_top=False,
        statistic_tests=False,
        title=f"Comparativa de {modelo_1} vs {modelo_2} - {nombre_pestana}\n{dataset_nombre_amigable}"
    )

    # Ocultamos el texto original en inglés ("*Dashed lines...")
    for child in ax.get_children():
        if type(child).__name__ == 'AnchoredText':
            if '*Dashed lines' in child.txt.get_text():
                child.set_visible(False)

    # Creamos la carpeta 'scatters' si no existe
    carpeta_salida = 'scatters'
    os.makedirs(carpeta_salida, exist_ok=True)

    # 7. Guardamos la imagen
    mod1_file = modelo_1.replace(' ', '_').replace('.', '')
    mod2_file = modelo_2.replace(' ', '_').replace('.', '')
    
    nombre_archivo_png = os.path.join(carpeta_salida, f'scatter_{mod1_file}_{mod2_file}_{nombre_pestana}_{dataset_elegido}.png')
    nombre_archivo_pdf = os.path.join(carpeta_salida, f'scatter_{mod1_file}_{mod2_file}_{nombre_pestana}_{dataset_elegido}.pdf')
    plt.savefig(nombre_archivo_png, dpi=300, bbox_inches='tight')
    plt.savefig(nombre_archivo_pdf, bbox_inches='tight')
    plt.show()
    print(f"¡Scatter Plot generado y guardado como {nombre_archivo_png} y también en PDF!")
else:
    print(f"Ojo, alguno de los modelos indicados ({modelo_1} o {modelo_2}) no está en el dataframe del dataset.")
