import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# --- Configuración ---
FOLDER_METRICS = "data/result_data/metrics" # Cambia esto al directorio donde están tus archivos JSON
FOLDER_REPORTS = "data/result_data/reports" # Cambia esto al directorio donde quieres guardar los gráficos y el README
README_FILENAME = os.path.join(FOLDER_REPORTS, "README_metrics_comparison.md")
# Ya no definimos un único PLOT_FILENAME aquí, se generarán varios

# --- 1. Leer y Extraer Datos ---
all_metrics_data = []
inferred_type_evaluation = None # Variable para guardar el tipo inferido

print(f"Buscando archivos de métricas en: {FOLDER_METRICS}")

try:
    if not os.path.isdir(FOLDER_METRICS):
        print(f"ERROR: El directorio '{FOLDER_METRICS}' no existe.")
        exit()

    json_files = [f for f in os.listdir(FOLDER_METRICS) if f.endswith(".json") and (f.startswith("TRAIN_metrics") or f.startswith("TEST_metrics"))]

    if not json_files:
        print("No se encontraron archivos JSON de métricas que coincidan con el patrón (TRAIN_metrics... o TEST_metrics...).")
        exit()

    for filename in json_files:
        filepath = os.path.join(FOLDER_METRICS, filename)
        print(f"Procesando archivo: {filename}")

        # Inferir TYPE_EVALUATION del primer archivo válido encontrado
        if inferred_type_evaluation is None:
            if filename.startswith("TRAIN_"):
                inferred_type_evaluation = "TRAIN"
            elif filename.startswith("TEST_"):
                 inferred_type_evaluation = "TEST"
            else:
                 inferred_type_evaluation = "UNKNOWN" # O manejar como error
            print(f"[*] Tipo de evaluación inferido de los nombres de archivo: {inferred_type_evaluation}")

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            config = {
                'Filename': filename,
                'Reranker': data.get('Ranked', 'N/A'),
                'Query_Type': data.get('Query_ranked', 'N/A')
            }
            metrics = data.get('metrics', {})
            if not metrics or not isinstance(metrics, dict):
                print(f"  Advertencia: No se encontró la clave 'metrics' válida en {filename}")
                continue
            entry = {**config, **metrics}
            all_metrics_data.append(entry)
        except Exception as e:
            print(f"  Error procesando {filename}: {e}")

except Exception as e:
    print(f"ERROR general al listar/procesar archivos: {e}")
    exit()

if not all_metrics_data:
    print("\nNo se cargaron datos de métricas válidos para analizar.")
    exit()

# Si no se pudo inferir el tipo (raro si hay archivos), usar un default
if inferred_type_evaluation is None:
     print("Advertencia: No se pudo inferir el tipo de evaluación de los nombres de archivo. Usando 'UNKNOWN'.")
     inferred_type_evaluation = "UNKNOWN"


# --- 2. Organizar en DataFrame ---
df_analysis = pd.DataFrame(all_metrics_data)
print(f"\nSe cargaron datos de {len(df_analysis)} configuraciones.")
df_analysis['Configuración'] = df_analysis['Reranker'] + ' (' + df_analysis['Query_Type'] + ')'
metrics_cols = ['S@10', 'P@10', 'R@10', 'F1@10', 'MAP@10', 'MRR', 'NDCG@10']
existing_metrics_cols = [col for col in metrics_cols if col in df_analysis.columns]

if not existing_metrics_cols:
    print("\nERROR: Ninguna de las columnas de métricas esperadas se encontró.")
    exit()


# --- 3. Generar Gráficos Separados por Métrica ---
print(f"\nGenerando gráficos de comparación por métrica...")
generated_plot_filenames = [] # Lista para guardar nombres de los gráficos generados

sns.set_theme(style="whitegrid") # Establecer tema de Seaborn

for metric in existing_metrics_cols:
    plot_filename = os.path.join(FOLDER_REPORTS, f"comparison_plot_{metric.replace('@','_at_')}.png")
    generated_plot_filenames.append(os.path.basename(plot_filename)) # Guardar solo el nombre base para el README
    print(f"  Generando gráfico para: {metric} -> {plot_filename}")

    try:
        plt.figure(figsize=(max(8, len(df_analysis) * 1.2), 6)) # Ajustar tamaño dinámicamente
        # Usar barplot directamente para una métrica a la vez
        ax = sns.barplot(
            data=df_analysis.sort_values(by=metric, ascending=False), # Ordenar barras por valor
            x='Configuración',
            y=metric,
            palette="viridis",
            hue='Configuración', # Usar hue para leyendas consistentes si se desea, o quitarlo
            dodge=False # No esquivar si x es la configuración
        )

        ax.set_title(f'Comparación de {metric} por Configuración', fontsize=14)
        ax.set_xlabel("Configuración", fontsize=12)
        ax.set_ylabel("Puntuación Promedio", fontsize=12)
        plt.xticks(rotation=45, ha='right') # Rotar etiquetas si son largas
        # Añadir etiquetas de valor encima de las barras (opcional)
        # for container in ax.containers:
        #    ax.bar_label(container, fmt='%.3f')

        plt.tight_layout() # Ajustar layout
        plt.savefig(plot_filename, bbox_inches='tight')
        plt.close() # Cerrar figura actual antes de crear la siguiente

    except Exception as e:
        print(f"  Error al generar o guardar el gráfico para {metric}: {e}")


# --- 4. Generar el archivo README.md ---
print(f"\nGenerando archivo README...")
try:
    # Preparar tabla para Markdown
    cols_for_readme = ['Configuración', 'Reranker', 'Query_Type'] + existing_metrics_cols
    df_readme_table = df_analysis[cols_for_readme].copy()
    df_readme_table = df_readme_table.sort_values(by='NDCG@10', ascending=False)
    for col in existing_metrics_cols:
        df_readme_table[col] = df_readme_table[col].round(4)
    markdown_table = df_readme_table.to_markdown(index=False)

    today_date = datetime.now().strftime("%Y-%m-%d")

    # Generar sección de gráficos para el README
    graficos_md_section = "## Gráficos Comparativos por Métrica\n\n"
    if generated_plot_filenames:
        for plot_name in generated_plot_filenames:
             metric_name = plot_name.replace('comparison_plot_', '').replace('.png','').replace('_at_','@')
             graficos_md_section += f"### {metric_name}\n"
             graficos_md_section += f"![Comparación de {metric_name}](./{plot_name})\n\n"
    else:
        graficos_md_section += "No se pudieron generar los gráficos.\n"

    # Encontrar mejores configuraciones
    best_ndcg_config_row = df_analysis.loc[df_analysis['NDCG@10'].idxmax()] if 'NDCG@10' in df_analysis else None
    best_map_config_row = df_analysis.loc[df_analysis['MAP@10'].idxmax()] if 'MAP@10' in df_analysis else None

    # Construir el contenido completo del README
    readme_content = f"""# Comparación de Resultados de Evaluación ({inferred_type_evaluation}) - {today_date}

Este documento resume los resultados de las métricas obtenidas al evaluar diferentes configuraciones del sistema de recuperación y re-ranking.

## Tabla Comparativa de Métricas Promedio

La siguiente tabla muestra las métricas promedio (@10) para cada configuración probada, ordenada por NDCG@10 descendente.

{markdown_table}

{graficos_md_section}
*(Nota: Los gráficos asumen que los archivos de imagen están en el mismo directorio que este README)*

## Conclusiones Preliminares
"""
    if best_ndcg_config_row is not None:
        readme_content += f"\n*   **Mejor Configuración (según NDCG@10):** `{best_ndcg_config_row['Configuración']}` (NDCG@10: {best_ndcg_config_row['NDCG@10']:.4f})"
    if best_map_config_row is not None:
         readme_content += f"\n*   **Mejor Configuración (según MAP@10):** `{best_map_config_row['Configuración']}` (MAP@10: {best_map_config_row['MAP@10']:.4f})"

    readme_content += """
*   **Observaciones:** Analiza la tabla y los gráficos individuales para identificar patrones. ¿Algún tipo de reranker (`BM25`, `TF-IDF`) funciona consistentemente mejor para ciertas métricas? ¿Qué tipo de query (`KEYWORDS`, `BODY`, `BODY + KEYWORDS`) es más efectivo en combinación con cada reranker?

*(Recuerda que estos resultados son específicos para el conjunto de datos '{inferred_type_evaluation}' y las configuraciones probadas hasta la fecha)*
"""

    # Escribir el contenido al archivo README.md
    with open(README_FILENAME, 'w', encoding='utf-8') as f:
        f.write(readme_content)

    print(f"Archivo README guardado en: {README_FILENAME}")

except Exception as e:
    print(f"Error al generar o guardar el archivo README: {e}")

print("\nAnálisis completado.")