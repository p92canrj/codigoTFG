import os
import pandas as pd
from aeon.visualisation import plot_critical_difference
import matplotlib.pyplot as plt

nombres_modelos = {
    'lightgbmclassifier_fast': 'LightGBM',
    'logisticat': 'LogisticAT',
    'logisticregressor': 'Reg. Logística',
    'nnop': 'NNOP',
    'nnpom': 'NNPOM',
    'ordinaldecomposition': 'Descomp. Ordinal'
}

ruta_excel = r'../prepared_results_def/20260605_121221_recap_FINAL.xlsx'
metrica = 'MMAE' # <--- Puedes cambiar la métrica aquí

# 1. Cargamos la hoja "Average" del Excel
df_avg = pd.read_excel(ruta_excel, sheet_name='Average')

# Reemplazamos los nombres de los estimadores por los que he puesto arriba
df_avg['estimator_name'] = df_avg['estimator_name'].replace(nombres_modelos)

# Hacemos un "pivot" para tener los modelos como columnas
df_pivot = df_avg.pivot(index='dataset', columns='estimator_name', values=metrica)

# 2. Comprobamos si es una métrica donde "menor es mejor"
lower_better = True if metrica in ['AMAE', 'MMAE', 'MAE'] else False

# 3. Generamos el CDD plot usando directamente los valores y columnas de df_pivot
fig, ax = plot_critical_difference(
    df_pivot.values, 
    labels=list(df_pivot.columns), 
    alpha=0.05, 
    lower_better=lower_better
)

# 4. Creamos la carpeta 'cdd_plots' si no existe
carpeta_salida = 'cdd_plots'
os.makedirs(carpeta_salida, exist_ok=True)

# 5. Guardamos la imagen
nombre_archivo_png = os.path.join(carpeta_salida, f'cdd_{metrica}_recap.png')
nombre_archivo_pdf = os.path.join(carpeta_salida, f'cdd_{metrica}_recap.pdf')
fig.savefig(nombre_archivo_png, dpi=300, bbox_inches='tight')
fig.savefig(nombre_archivo_pdf, bbox_inches='tight') # Formato vectorial ideal para LaTeX

plt.show()

print(f"¡Gráfica CDD generada y guardada como {nombre_archivo_png} y también en PDF!")
