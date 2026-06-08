import os
import sys
from pathlib import Path

# Añadir la carpeta metar al path para que Python encuentre el módulo 'launchexp' al cargar los resultados
# Esto es necesario para los modelos logisticat, nnop y nnpom
sys.path.append(str(Path("../metar").resolve()))

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from remayn.result_set import ResultFolder
from sklearn.metrics import confusion_matrix

# Buscar la carpeta results_debug independientemente de si ejecutamos desde código/, metar/ o graficas/
if Path("./results_debug").exists():
    path = Path("./results_debug")
elif Path("../results_debug").exists():
    path = Path("../results_debug")
elif Path("../metar/results_debug").exists():
    path = Path("../metar/results_debug")
else:
    raise FileNotFoundError("No se ha encontrado la carpeta results_debug")

results = ResultFolder(path)
print(f"Results loaded from {path.absolute()}")

# Definir los modelos y datasets de los que queremos extraer las matrices
modelos = [
    "logisticat",
    "logisticregressor",
    "lightgbmclassifier_fast",
    "nnop",
    "ordinaldecomposition",
    "nnpom"
]

datasets = [
    #"LEST_sensors_autoreg",
    #"LEVX_sensors_autoreg",
    #"LEST_sensors_areg_only",
    #"LEVX_sensors_areg_only",
    #"LEST_sensors_areg_era5",
    #"LEVX_sensors_areg_era5",
    "LEST_era5",
    "LEVX_era5"
]

dataset_names = {
    'LEST_sensors_autoreg': 'Solo METAR (LEST)',
    'LEVX_sensors_autoreg': 'Solo METAR (LEVX)',
    'LEST_sensors_areg_only': 'Solo areg (LEST)',
    'LEVX_sensors_areg_only': 'Solo areg (LEVX)',
    'LEST_era5': 'Solo ERA5 (LEST)',
    'LEVX_era5': 'Solo ERA5 (LEVX)',
    'LEST_sensors_areg_era5': 'Conjunto completo (LEST)',
    'LEVX_sensors_areg_era5': 'Conjunto completo (LEVX)'
}

model_names = {
    'lightgbmclassifier_fast': 'LightGBM',
    'logisticat': 'LogisticAT',
    'logisticregressor': 'Regresión Logística',
    'nnop': 'NNOP',
    'nnpom': 'NNPOM',
    'ordinaldecomposition': 'Descomposición Ordinal'
}

# Crear carpeta para guardar las imágenes (ahora se llamará matrices_confusion)
output_dir = Path("matrices_confusion")
output_dir.mkdir(exist_ok=True)

for dataset in datasets:
    for modelo in modelos:
        print(f"Procesando {modelo} en {dataset}...")
        
        def filter_fn(result):
            return (
                result.config["estimator_name"] == modelo
                and result.config["dataset"] == dataset
            )
        
        # Filtramos los resultados para el modelo y dataset actuales
        filtered_results = results.filter(filter_fn)
        runs = list(filtered_results.results_.values())
        
        if len(runs) == 0:
            print(f"  -> No se encontraron resultados. Saltando...")
            continue
            
        matrices = []
        
        for run in runs:
            try:
                run_data = run.get_data()
                cm = confusion_matrix(run_data.targets, run_data.predictions)
                matrices.append(cm)
            except Exception as e:
                print(f"    [Error] Fallo en run: {e}")
                continue
                
        if len(matrices) == 0:
            print(f"  -> No se pudieron extraer matrices (todas fallaron). Saltando...")
            continue
            
        cm_mean = np.mean(matrices, axis=0)
        
        cm_normalized = cm_mean.astype('float') / cm_mean.sum(axis=1)[:, np.newaxis]
        cm_normalized = np.nan_to_num(cm_normalized)
        
        plt.figure(figsize=(8, 6))
        
        ax = sns.heatmap(
            cm_normalized, 
            annot=True,
            annot_kws={"size": 17},
            fmt=".2%",
            cmap="Blues",
            cbar=True,
            vmin=0.0, vmax=1.0
        )
        
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=16)
        
        dataset_title = dataset_names.get(dataset, dataset)
        modelo_title = model_names.get(modelo, modelo)
        plt.title(f'Matriz de Confusión Media\n{dataset_title} - {modelo_title}', fontsize=22, pad=15)
        plt.ylabel('Etiqueta Real', fontsize=18)
        plt.xlabel('Predicción del Modelo', fontsize=18)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16, rotation=0)
        plt.tight_layout()
        
        subfolder_name = dataset.split('_', 1)[1]
        if subfolder_name == 'sensors':
            subfolder_name = 'sensors_autoreg'
        elif subfolder_name == 'sensors_areg_only':
            subfolder_name = 'areg_only'
            
        dataset_out_dir = output_dir / subfolder_name
        dataset_out_dir.mkdir(exist_ok=True)
        
        filename_png = dataset_out_dir / f"cm_{dataset}_{modelo}.png"
        filename_pdf = dataset_out_dir / f"cm_{dataset}_{modelo}.pdf"
        plt.savefig(filename_png, dpi=300, bbox_inches='tight')
        plt.savefig(filename_pdf, bbox_inches='tight')
        plt.close()
        
print(f"¡Listo! Las matrices se han guardado en la carpeta '{output_dir.absolute()}'")
