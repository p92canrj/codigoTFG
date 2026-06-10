# Código TFG - Baja Visibilidad Ordinal

Este repositorio contiene el código y los datos utilizados para el Trabajo de Fin de Grado (TFG) **"Predicción de situaciones de baja visibilidad utilizando métodos de clasificación ordinal"**.

## Estructura del Proyecto

A continuación se detalla la estructura del directorio principal y qué contiene cada archivo y carpeta:

### Directorios

* **`datasets/`**: Contiene los conjuntos de datos preprocesados en formato CSV, combinando datos de sensores y reanálisis (ERA5) para los aeropuertos analizados (LEST y LEVX). Además, tiene scripts específicos para la creación de estos conjuntos de datos (`create_areg_era5.py` y `create_areg_only.py`).
* **`forecasting/`**: Incluye un notebook (`proceso_forecasting.ipynb`) con el proceso de forecasting empleado para obtener las variables autorregresivas y guardarlos en (`LEST_sensors_forecast.csv`, `LEVX_sensors_forecast.csv`).
* **`graficas/`**: Directorio dedicado a la generación y almacenamiento de visualizaciones. Contiene varios scripts de Python (`generar_boxplots.py`, `generar_mcm.py`, etc.) para crear matrices de confusión, gráficos de dispersión (scatters), diagramas de cajas (boxplots), diagramas de diferencia crítica (CDD) y matrices de comparación (MCM). También almacena las imágenes generadas en sus respectivas subcarpetas.
* **`metar/`**: Contiene la lógica y los scripts principales para ejecutar los experimentos (`run_experiment.py`), recolectar resultados (`run_results_collector.py`).
* **`prepared_results_def/`**: Almacena los archivos Excel con el resumen y la recopilación de los resultados finales de los diferentes experimentos.
* **`results_debug/`**: **Nota Importante:** Esta carpeta contiene resultados detallados, modelos guardados y datos de depuración generados durante los experimentos. Debido a su gran tamaño, esta carpeta **no se sube a GitHub**.


## Notas Adicionales

* Si vas a ejecutar nuevos experimentos, ten en cuenta que el directorio `results_debug/` puede crecer rápidamente. No elimines la regla del `.gitignore` para evitar problemas al hacer *commits* y *pushes* al repositorio remoto.
