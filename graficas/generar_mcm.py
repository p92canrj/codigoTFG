import os
import pandas as pd
from aeon.visualisation import create_multi_comparison_matrix
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

ruta_excel = r'../prepared_results_def/20260605_121221_recap_FINAL.xlsx'
metrica = 'CCR'

# 1. Cargamos la hoja "Average" del Excel
df_avg = pd.read_excel(ruta_excel, sheet_name='Average')

# Reemplazamos los nombres de los estimadores por los que he puesto arriba
df_avg['estimator_name'] = df_avg['estimator_name'].replace(nombres_modelos)

# Hacemos un "pivot" para tener los modelos como columnas
df_pivot = df_avg.pivot(index='dataset', columns='estimator_name', values=metrica)

# 2. Definimos los modelos que queremos someter al test
modelos_estocasticos_raw = ["lightgbmclassifier_fast", "logisticregressor", "nnop", "nnpom"]

# Los traducimos a los nombres que he puesto arriba para poder filtrarlos del df_pivot
modelos_estocasticos = [nombres_modelos.get(m, m) for m in modelos_estocasticos_raw]

df_estocasticos = df_pivot[modelos_estocasticos]

# 3. Comprobamos si es una métrica donde "menor es mejor"
lower_better = True if metrica in ['AMAE', 'MMAE', 'MAE'] else False

# 4. Generamos la Matriz Visual (Wilcoxon-Holm)
f = create_multi_comparison_matrix(
    df_estocasticos, 
    font_size=14,
    pvalue_correction="holm",
    pvalue_test_params={"alternative": "two-sided"},
    higher_stat_better=not lower_better,
    order_stats_increasing=lower_better
)

# 5. Creamos la carpeta 'mcm_plots' si no existe
carpeta_salida = 'mcm_plots'
os.makedirs(carpeta_salida, exist_ok=True)

# 6. Guardamos la imagen
nombre_archivo_png = os.path.join(carpeta_salida, f'mcm_{metrica}_estocasticos_recap.png')
nombre_archivo_pdf = os.path.join(carpeta_salida, f'mcm_{metrica}_estocasticos_recap.pdf')

f.set_size_inches(12, 10)

f.savefig(nombre_archivo_png, dpi=300, bbox_inches='tight')
f.savefig(nombre_archivo_pdf, bbox_inches='tight')

plt.show()

print(f"¡Matriz MCM guardada como {nombre_archivo_png} y también en PDF!")
