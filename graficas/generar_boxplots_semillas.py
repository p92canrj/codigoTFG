import os
import pandas as pd
import seaborn as sns
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
nombre_pestana = 'AMAE'

# 1. Cargamos el Excel
df_metric = pd.read_excel(ruta_excel, sheet_name=nombre_pestana, index_col=0, header=0)

# 2. Elegimos el dataset
dataset_elegido = 'LEVX_sensors_areg_era5' 

# 3. Filtramos las columnas que pertenecen a este dataset
sufijo = f"_{dataset_elegido}"
columnas_dataset = [col for col in df_metric.columns if col.endswith(sufijo)]

df_dataset = df_metric[columnas_dataset].copy()

# 4. Renombramos las columnas
df_dataset.columns = [col.replace(sufijo, "") for col in df_dataset.columns]

# Renombramos también usando los nombres que he puesto arriba
df_dataset.rename(columns=nombres_modelos, inplace=True)

# 5. Generamos el Boxplot
plt.figure(figsize=(14, 8))
sns.boxplot(data=df_dataset)

dataset_nombre_amigable = nombres_datasets.get(dataset_elegido, dataset_elegido)
plt.title(f'Variabilidad de {nombre_pestana} en 30 ejecuciones\nDataset: {dataset_nombre_amigable}', fontsize=22)
plt.ylabel('Valor de la métrica', fontsize=22)
plt.xlabel('Modelo', fontsize=22)
plt.xticks(rotation=45, fontsize=20)
plt.yticks(fontsize=20)
plt.tight_layout()

# 6. Creamos la carpeta 'boxplots' si no existe
carpeta_salida = 'boxplots'
os.makedirs(carpeta_salida, exist_ok=True)

# Guardamos la imagen
nombre_archivo_png = os.path.join(carpeta_salida, f'boxplot_{nombre_pestana}_{dataset_elegido}.png')
nombre_archivo_pdf = os.path.join(carpeta_salida, f'boxplot_{nombre_pestana}_{dataset_elegido}.pdf')
plt.savefig(nombre_archivo_png, dpi=300, bbox_inches='tight')
plt.savefig(nombre_archivo_pdf, bbox_inches='tight')

plt.show()
print(f"¡Gráfica guardada correctamente como {nombre_archivo_png} y también en PDF!")
