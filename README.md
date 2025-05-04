# Proyecto de Evaluación y Re-Ranking de Consultas

Este proyecto implementa un sistema para procesar consultas, extraer palabras clave, realizar búsquedas en PubMed, reordenar resultados y evaluar métricas de desempeño. A continuación, se describe la estructura del proyecto, cómo instalar las dependencias y cómo ejecutarlo.

## Estructura del Proyecto

```
/workspaces/codespaces-blank
├── analyze_metrics.py          # Script para analizar métricas y generar gráficos
├── main.py                     # Script principal para procesar consultas y generar resultados
├── requirements.txt            # Lista de dependencias necesarias para el proyecto
├── data/
│   ├── result_data/            # Carpeta para almacenar resultados y reportes
│   │   ├── metrics/            # Métricas generadas durante la evaluación
│   │   ├── reports/            # Reportes y gráficos generados
│   ├── test_data/              # Datos de prueba
│   ├── train_data/             # Datos de entrenamiento
├── processing/                 # Módulos de procesamiento
│   ├── metrics.py              # Funciones para calcular métricas
│   ├── Reranker.py             # Implementación de re-ranking
│   ├── utils.py                # Utilidades generales
├── services/                   # Servicios externos
│   ├── Gemini.py               # Servicio para interacción con Gemini
│   ├── PubMed.py               # Servicio para interacción con PubMed
├── .env                        # Archivo de configuración con claves API y parámetros
```

## Instalación de Dependencias

1. Asegúrate de tener Python 3.8 o superior instalado.
2. Instala las dependencias listadas en `requirements.txt` ejecutando:

```bash
pip install -r requirements.txt
```

## Configuración

1. Crea un archivo `.env` en la raíz del proyecto con las siguientes variables (ya existe un ejemplo en el proyecto):

```
PUBMED_API_KEY=<tu_clave_api_pubmed>
GEMINI_API_KEY=<tu_clave_api_gemini>
TYPE_EVALUATION=TRAIN # Cambia a TEST si es necesario
RERANKER_TYPE=BM25 # Opciones: TF-IDF, BM25, PubMedBERT
RERANKER_QUERY=KEYWORDS # Opciones: BODY, BODY + KEYWORDS, KEYWORDS
```

## Ejecución

1. Para procesar las consultas y generar resultados, ejecuta el script principal:

```bash
python main.py
```

2. Los resultados se guardarán en la carpeta `data/result_data/train` o `data/result_data/test` dependiendo de la configuración en `.env`.

3. Para analizar métricas y generar gráficos, ejecuta:

```bash
python analyze_metrics.py
```

## Requerimientos

- Python 3.8 o superior
- Conexión a internet para interactuar con las APIs de PubMed y Gemini
- Claves API válidas para PubMed y Gemini

## Notas

- Los gráficos y reportes generados se almacenan en `data/result_data/reports`.
- Asegúrate de que las carpetas de entrada (`train_data` o `test_data`) contengan archivos JSON válidos con el formato esperado.