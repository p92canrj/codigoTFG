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


# Configuración inicial
ruta_excel = r'../prepared_results_def/20260605_121221_recap_FINAL.xlsx'
metrica = 'AMAE' # <--- Puedes cambiar la métrica aquí

# 1. Cargamos la hoja "Average" del Excel
df_avg = pd.read_excel(ruta_excel, sheet_name='Average')

# Reemplazamos los nombres de los estimadores por los amigables
df_avg['estimator_name'] = df_avg['estimator_name'].replace(nombres_modelos)

# Hacemos un "pivot" para tener los modelos como columnas
df_pivot = df_avg.pivot(index='dataset', columns='estimator_name', values=metrica)

# 2. Definimos qué dos modelos queremos enfrentar cara a cara. 
# Si están en el diccionario, cogemos su nombre amigable, sino el original.
modelo_1_raw = "nnpom"
modelo_2_raw = "lightgbmclassifier_fast"

modelo_1 = nombres_modelos.get(modelo_1_raw, modelo_1_raw)
modelo_2 = nombres_modelos.get(modelo_2_raw, modelo_2_raw)

# 3. Comprobamos si es una métrica donde "menor es mejor"
lower_better = True if metrica in ['AMAE', 'MMAE', 'MAE'] else False

# 4. Nos aseguramos de que ambos modelos existan en nuestro pivot
if modelo_1 in df_pivot.columns and modelo_2 in df_pivot.columns:

    fig, ax = plot_pairwise_scatter(
        df_pivot[modelo_1].values, 
        df_pivot[modelo_2].values,
        modelo_1,
        modelo_2,
        metric=metrica,
        lower_better=lower_better,
        best_on_top=False,
        statistic_tests=False,
        title=f"Comparativa de {modelo_1} vs {modelo_2} - {metrica}"
    )

    # Ocultamos el texto original en inglés ("*Dashed lines...")
    for child in ax.get_children():
        if type(child).__name__ == 'AnchoredText':
            if '*Dashed lines' in child.txt.get_text():
                child.set_visible(False)

    # Creamos la carpeta 'scatters' si no existe
    carpeta_salida = 'scatters'
    os.makedirs(carpeta_salida, exist_ok=True)

    # 5. Guardamos la imagen
    # Limpiamos los nombres de los modelos para el archivo final (quitamos espacios y puntos)
    mod1_file = modelo_1.replace(' ', '_').replace('.', '')
    mod2_file = modelo_2.replace(' ', '_').replace('.', '')
    
    nombre_archivo_png = os.path.join(carpeta_salida, f'scatter_{mod1_file}_{mod2_file}_{metrica}_recap.png')
    nombre_archivo_pdf = os.path.join(carpeta_salida, f'scatter_{mod1_file}_{mod2_file}_{metrica}_recap.pdf')
    plt.savefig(nombre_archivo_png, dpi=300, bbox_inches='tight')
    plt.savefig(nombre_archivo_pdf, bbox_inches='tight')
    plt.show()
    print(f"¡Scatter Plot generado y guardado como {nombre_archivo_png} y también en PDF!")
else:
    print(f"Ojo, alguno de los modelos indicados ({modelo_1} o {modelo_2}) no está en el dataframe.")
