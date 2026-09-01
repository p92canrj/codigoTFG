# Código TFG - Predicción de situaciones de baja visibilidad

Este repositorio contiene el código y los datos utilizados para el Trabajo de Fin de Grado (TFG) **"Predicción de situaciones de baja visibilidad utilizando métodos de clasificación ordinal"**.

**Autor:** Jorge Cañete Ramírez  
**Directores:** David Guijo Rubio, Pedro Antonio Gutiérrez Peña

## Estructura del Proyecto

A continuación se detalla la estructura del directorio principal y qué contiene cada archivo y carpeta:

### Directorios

* **`datasets/`**: Contiene los conjuntos de datos preprocesados en formato CSV, combinando datos de sensores y reanálisis (ERA5) para los aeropuertos analizados (LEST y LEVX). Además, tiene scripts específicos para la creación de estos conjuntos de datos (`create_areg_era5.py` y `create_areg_only.py`).
* **`documento_final/`**: Contiene el documento PDF final de la memoria del Trabajo de Fin de Grado.
* **`forecasting/`**: Incluye un notebook (`proceso_forecasting.ipynb`) con el proceso de forecasting empleado para obtener las variables autorregresivas y guardarlos en (`LEST_sensors_forecast.csv`, `LEVX_sensors_forecast.csv`).
* **`graficas/`**: Directorio dedicado a la generación y almacenamiento de visualizaciones. Contiene varios scripts de Python (`generar_boxplots.py`, `generar_mcm.py`, etc.) para crear matrices de confusión, gráficos de dispersión (scatters), diagramas de cajas (boxplots), diagramas de diferencia crítica (CDD) y matrices de comparación (MCM). También almacena las imágenes generadas en sus respectivas subcarpetas.
* **`metar/`**: Contiene la lógica y los scripts principales para ejecutar los experimentos (`run_experiment.py`), recolectar resultados (`run_results_collector.py`).
* **`prepared_results_def/`**: Almacena los archivos Excel con el resumen y la recopilación de los resultados finales de los diferentes experimentos.
* **`results_debug/`**: **Nota Importante:** Esta carpeta contiene resultados detallados, modelos guardados y datos de depuración generados durante los experimentos. Debido a su gran tamaño, esta carpeta **no se sube a GitHub**.

## Ejecución de Experimentos

Para lanzar un experimento desde la carpeta raíz del proyecto, puedes utilizar el script `run_experiment.py` pasándole los parámetros necesarios por línea de comandos. Un ejemplo de ejecución sería:

```bash
python .\metar\run_experiment.py datasets .\results_debug <nombre_modelo> <nombre_dataset> <num_semilla> -1
```

**Explicación de los parámetros:**
* `datasets`: Ruta al directorio donde se encuentran los archivos CSV de los datos.
* `metar/results_debug`: Ruta al directorio donde se guardarán los resultados detallados y el modelo entrenado (el script la creará si no existe).
* `<nombre_modelo>`: Nombre del estimador/clasificador a utilizar. Los enfoques nominales son `logisticregressor` y `lightgbmclassifier_fast`, y los ordinales son `logisticat`, `nnop`, `ordinaldecomposition` y `nnpom`.
* `<nombre_dataset>`: Nombre del dataset a cargar, sin la extensión `.csv` (a elegir entre: `LEST_sensors_autoreg`, `LEVX_sensors_autoreg`, `LEST_sensors_areg_only`, `LEVX_sensors_areg_only`, `LEST_sensors_areg_era5`, `LEVX_sensors_areg_era5`, `LEST_era5`, `LEVX_era5`).
* `<num_semilla>`: Valor de la semilla aleatoria para asegurar la reproducibilidad (por ejemplo, `0`).
* `-1`: Número de hilos de ejecución (`n_jobs`). `-1` indica que se utilizarán todos los procesadores disponibles.

## Recopilación de Resultados
Una vez finalizados los experimentos, puedes generar un informe Excel con el resumen de todas las métricas de evaluación utilizando el script `run_results_collector.py`. Para ejecutarlo desde la raíz del proyecto, usa el siguiente comando:

```bash
python .\metar\run_results_collector.py <nombre_archivo> 

```
**Explicación de los parámetros:**
* `<nombre_archivo>`: El nombre o sufijo que se le añadirá al archivo Excel generado (por ejemplo, `mis_resultados`). El archivo se guardará en la carpeta `prepared_results_def/` e incluirá la fecha y hora actuales como prefijo.

## Notas Adicionales

* Si vas a ejecutar nuevos experimentos, ten en cuenta que el directorio `results_debug/` puede crecer rápidamente. No elimines la regla del `.gitignore` para evitar problemas al hacer *commits* y *pushes* al repositorio remoto.
