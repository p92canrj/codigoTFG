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

# 1. Cargamos la hoja "Average" del excel de resultados
ruta_excel = r'../prepared_results_def/20260605_121221_recap_FINAL.xlsx'
df_avg = pd.read_excel(ruta_excel, sheet_name='Average')

# Reemplazamos los nombres de los estimadores que he puesto arriba
df_avg['estimator_name'] = df_avg['estimator_name'].replace(nombres_modelos)

# 2. Elegimos la métrica que queremos visualizar
metrica = 'MS'

# 3. Hacemos un "pivot" para que los datasets sean las filas y los estimadores sean las columnas
df_pivot = df_avg.pivot(index='dataset', columns='estimator_name', values=metrica)

# 4. Generamos el Boxplot
plt.figure(figsize=(14, 8))
sns.boxplot(data=df_pivot)
plt.title(f'Distribución de la métrica {metrica} por Modelo', fontsize=24)
plt.ylabel(metrica, fontsize=16)
plt.xlabel('Modelo', fontsize=16)
plt.xticks(rotation=45, fontsize=16)
plt.yticks(fontsize=16)
plt.tight_layout()

# Creamos la carpeta 'boxplots' si no existe
carpeta_salida = 'boxplots'
os.makedirs(carpeta_salida, exist_ok=True)

# Guardamos la imagen antes de mostrarla
nombre_archivo_png = os.path.join(carpeta_salida, f'boxplot_{metrica}_recap.png')
nombre_archivo_pdf = os.path.join(carpeta_salida, f'boxplot_{metrica}_recap.pdf')
plt.savefig(nombre_archivo_png, dpi=300, bbox_inches='tight')
plt.savefig(nombre_archivo_pdf, bbox_inches='tight')

plt.show()
print(f"¡Gráfica guardada correctamente como {nombre_archivo_png} y en PDF como {nombre_archivo_pdf}!")
