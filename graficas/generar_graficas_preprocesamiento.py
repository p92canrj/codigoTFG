import pandas as pd
import matplotlib.pyplot as plt
import os

datasets = [
    '../datasets/LEST_sensors.csv',
    '../datasets/LEVX_sensors.csv'
]

out_dir = 'graficas_preprocesamiento'
os.makedirs(out_dir, exist_ok=True)

# Función que devuelve el nombre del aeropuerto
def get_airport_name(data_identifier):
    if 'LEST' in data_identifier:
        return 'LEST'
    elif 'LEVX' in data_identifier:
        return 'LEVX'
    else:
        return 'Desconocido'

for ds in datasets:
    if not os.path.exists(ds):
        print(f"No se encontró el dataset: {ds}")
        continue
        
    print(f"Procesando {ds}...")
    airport_code = get_airport_name(ds)
    datos = pd.read_csv(ds)
    
    # Conversión del formato fecha
    datos['datetime'] = pd.to_datetime(datos[['year', 'month', 'day', 'hour', 'minute']])
    datos = datos.set_index('datetime')
    
    cols_to_drop = ['year', 'month', 'day', 'hour', 'minute']
    datos = datos.drop(columns=[c for c in cols_to_drop if c in datos.columns], errors='ignore')

    datos = datos.asfreq('h')
    datos = datos.sort_index()
    
    # Extraemos características de tiempo del índice
    datos['month'] = datos.index.month
    datos['week_day'] = datos.index.dayofweek + 1
    datos['hour_day'] = datos.index.hour

    # Configuraciones visuales globales
    title_fs = 20
    label_fs = 16
    tick_fs = 14

    # Gráfico 1: Distribución por mes
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    
    datos.boxplot(column='vis', by='month', ax=ax1, flierprops={'markersize': 3, 'alpha': 0.1})
    datos.groupby('month')['vis'].median().plot(style='o-', linewidth=1.5, color='red', ax=ax1)

    ax1.set_ylabel('Visibilidad (m)', fontsize=label_fs)
    ax1.set_xlabel('Mes del año', fontsize=label_fs)
    ax1.set_title(f'Distribución de la Visibilidad por Mes - {airport_code}', fontsize=title_fs)
    ax1.tick_params(axis='both', which='major', labelsize=tick_fs)
    fig1.suptitle('')

    plt.tight_layout()
    
    out_base = os.path.join(out_dir, f'1_visibilidad_mensual_{airport_code}')
    plt.savefig(f'{out_base}.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{out_base}.pdf', bbox_inches='tight')
    plt.close(fig1)

    # Gráfico 2: Distribución por día de la semana
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    
    datos.boxplot(column='vis', by='week_day', ax=ax2, flierprops={'markersize': 3, 'alpha': 0.1})
    datos.groupby('week_day')['vis'].median().plot(style='o-', linewidth=1.5, color='red', ax=ax2)

    ax2.set_ylabel('Visibilidad (m)', fontsize=label_fs)
    ax2.set_xlabel('Día de la semana (1=Lunes, 7=Domingo)', fontsize=label_fs)
    ax2.set_title(f'Distribución de la Visibilidad por Día de la Semana - {airport_code}', fontsize=title_fs)
    ax2.tick_params(axis='both', which='major', labelsize=tick_fs)
    fig2.suptitle('')

    plt.tight_layout()
    
    out_base = os.path.join(out_dir, f'2_visibilidad_semanal_{airport_code}')
    plt.savefig(f'{out_base}.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{out_base}.pdf', bbox_inches='tight')
    plt.close(fig2)

    # Gráfico 3: Distribución por hora del día
    fig3, ax3 = plt.subplots(figsize=(12, 6))
    
    datos.boxplot(column='vis', by='hour_day', ax=ax3, flierprops={'markersize': 3, 'alpha': 0.1})
    datos.groupby('hour_day')['vis'].median().plot(style='o-', linewidth=1.5, color='red', ax=ax3)

    ax3.set_ylabel('Visibilidad (m)', fontsize=label_fs)
    ax3.set_xlabel('Hora del día', fontsize=label_fs)
    ax3.set_title(f'Distribución de la Visibilidad por Hora - {airport_code}', fontsize=title_fs)
    ax3.tick_params(axis='both', which='major', labelsize=tick_fs)
    fig3.suptitle('')

    plt.tight_layout()
    
    out_base = os.path.join(out_dir, f'3_visibilidad_horaria_{airport_code}')
    plt.savefig(f'{out_base}.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{out_base}.pdf', bbox_inches='tight')
    plt.close(fig3)

    # Gráfico 4: Promedio a lo largo de la semana
    fig4, ax4 = plt.subplots(figsize=(12, 6))
    
    mean_day_hour = datos.groupby(["week_day", "hour_day"])["vis"].mean()

    mean_day_hour.plot(ax=ax4, color='green', linewidth=1.5)
    ax4.set_title(f"Evolución Promedio de la Visibilidad durante la Semana - {airport_code}", fontsize=title_fs)
    
    ax4.set_xticks([i * 24 for i in range(7)])
    ax4.set_xticklabels(["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"], fontsize=tick_fs)
    ax4.tick_params(axis='y', which='major', labelsize=tick_fs)
    
    ax4.set_xlabel("Día de la semana", fontsize=label_fs)
    ax4.set_ylabel("Visibilidad Media (m)", fontsize=label_fs)

    plt.tight_layout()
    
    out_base = os.path.join(out_dir, f'4_visibilidad_promedio_semana_{airport_code}')
    plt.savefig(f'{out_base}.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{out_base}.pdf', bbox_inches='tight')
    plt.close(fig4)

print(f"¡Todos los gráficos se han generado y guardado en la carpeta '{out_dir}' en formatos PNG y PDF!")
