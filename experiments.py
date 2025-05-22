import os
import itertools
import subprocess
import time
from dotenv import set_key, dotenv_values # Necesitarás: pip install python-dotenv

# --- Configuración del Script de Experimentación ---
MAIN_SCRIPT_NAME = "main_se.py"  # Nombre de tu script principal
BASE_ENV_FILE = ".env.base"      # Un archivo .env base con claves API y otras fijas (opcional)
TEMP_ENV_FILE = ".env"           # El archivo .env que tu script principal leerá

# Define las opciones para cada variable de entorno que quieres variar
# TYPE_EVALUATION está fijo a TRAIN
options_metodo_extraccion = ["LLM", "WEIRD"]
options_pubmed_restructure = ["GEMINI", "NONE"] # "NONE" o cualquier string que no sea "GEMINI"
options_reranker_type = ["BM25", "TF-IDF"] # "PubMedBERT"
options_reranker_query = ["KEYWORDS", "BODY", "BODY + KEYWORDS"]

# Variable fija
fixed_type_evaluation = "TRAIN"

# Crear una lista de todas las combinaciones posibles
all_combinations = list(itertools.product(
    options_metodo_extraccion,
    options_pubmed_restructure,
    options_reranker_type,
    options_reranker_query
))

print(f"Se ejecutarán un total de {len(all_combinations)} experimentos.")
print("Presiona Enter para comenzar o Ctrl+C para cancelar...")
input()

# --- Bucle Principal de Experimentación ---
for i, combo in enumerate(all_combinations):
    metodo_extraccion, pubmed_restructure, reranker_type, reranker_query = combo

    print("\n" + "="*80)
    print(f"Experimento {i+1}/{len(all_combinations)}")
    print(f"  TYPE_EVALUATION: {fixed_type_evaluation}")
    print(f"  METODO_EXTRACCION: {metodo_extraccion}")
    print(f"  PUBMED_RESTRUCTURE_METHOD: {pubmed_restructure}")
    print(f"  RERANKER_TYPE: {reranker_type}")
    print(f"  RERANKER_QUERY: {reranker_query}")
    print("="*80 + "\n")

    # 1. Preparar el archivo .env para esta ejecución
    #    Puedes copiar de un .env.base si tienes otras variables fijas (API Keys)
    #    O simplemente escribir/actualizar las variables necesarias.
    
    # Opción A: Usar un .env.base y añadir/sobrescribir
    # if os.path.exists(BASE_ENV_FILE):
    #     current_env_vars = dotenv_values(BASE_ENV_FILE)
    # else:
    #     current_env_vars = {}
    # current_env_vars["TYPE_EVALUATION"] = fixed_type_evaluation
    # current_env_vars["METODO_EXTRACCION"] = metodo_extraccion
    # current_env_vars["PUBMED_RESTRUCTURE_METHOD"] = pubmed_restructure
    # current_env_vars["RERANKER_TYPE"] = reranker_type
    # current_env_vars["RERANKER_QUERY"] = reranker_query
    # # Asegúrate de que tus API keys estén aquí si no están en BASE_ENV_FILE
    # # current_env_vars["PUBMED_API_KEY"] = "tu_pubmed_key" 
    # # current_env_vars["GEMINI_API_KEY"] = "tu_gemini_key"

    # with open(TEMP_ENV_FILE, 'w') as f:
    #     for key, value in current_env_vars.items():
    #         f.write(f'{key}="{value}"\n') # Asegurar comillas por si hay espacios

    # Opción B: Usar python-dotenv para modificar el archivo (más robusto)
    # Primero, asegúrate de que las claves API y otras fijas estén en el archivo .env
    # o cópialas de un .env.base si es la primera vez.
    # Si usas .env.base, copia su contenido a .env
    if os.path.exists(BASE_ENV_FILE):
        with open(BASE_ENV_FILE, 'r') as src, open(TEMP_ENV_FILE, 'w') as dst:
            dst.write(src.read())
    elif not os.path.exists(TEMP_ENV_FILE):
        # Si no hay .env.base ni .env, crea un .env vacío o con valores por defecto
        # Aquí deberías añadir tus API keys si no están ya en un .env
        print(f"Advertencia: {TEMP_ENV_FILE} no existe y no se encontró {BASE_ENV_FILE}. "
              "Asegúrate de que las API keys están en .env o añádelas manualmente.")
        # Ejemplo de cómo añadir API keys si no existen (ajusta según necesidad)
        # set_key(TEMP_ENV_FILE, "PUBMED_API_KEY", "TU_PUBMED_API_KEY_AQUI")
        # set_key(TEMP_ENV_FILE, "GEMINI_API_KEY", "TU_GEMINI_API_KEY_AQUI")
        pass


    # Establecer las variables para la combinación actual
    set_key(TEMP_ENV_FILE, "TYPE_EVALUATION", fixed_type_evaluation)
    set_key(TEMP_ENV_FILE, "METODO_EXTRACCION", metodo_extraccion)
    set_key(TEMP_ENV_FILE, "PUBMED_RESTRUCTURE_METHOD", pubmed_restructure)
    set_key(TEMP_ENV_FILE, "RERANKER_TYPE", reranker_type)
    set_key(TEMP_ENV_FILE, "RERANKER_QUERY", reranker_query)

    # 2. Ejecutar el script principal
    try:
        # Usamos subprocess.run para ejecutar y esperar a que termine.
        # Capturamos stdout y stderr para poder verlos o guardarlos si es necesario.
        # El `check=True` hará que se lance una excepción si el script devuelve un código de error.
        print(f"Ejecutando: python {MAIN_SCRIPT_NAME}")
        # Si tu script main_se.py está en un directorio específico, ajusta la ruta
        completed_process = subprocess.run(
            ["python", MAIN_SCRIPT_NAME],
            capture_output=True,  # Captura stdout y stderr
            text=True,            # Decodifica la salida como texto
            check=True,           # Lanza CalledProcessError si el script falla
            encoding='utf-8'      # Especifica la codificación
        )
        
        print("--- Salida del Script ---")
        print(completed_process.stdout)
        if completed_process.stderr:
            print("--- Errores del Script (si los hubo) ---")
            print(completed_process.stderr)
        print(f"--- Experimento {i+1} completado ---")

    except subprocess.CalledProcessError as e:
        print(f"ERROR durante la ejecución del experimento {i+1}:")
        print(f"  Comando: {e.cmd}")
        print(f"  Código de retorno: {e.returncode}")
        print("  Salida (stdout):")
        print(e.stdout)
        print("  Error (stderr):")
        print(e.stderr)
        print(f"--- Experimento {i+1} falló ---")
        # Podrías decidir continuar con el siguiente experimento o detenerte
        # continue 
        # break 
    except FileNotFoundError:
        print(f"ERROR: El script '{MAIN_SCRIPT_NAME}' no se encontró. Verifica la ruta.")
        break
    except Exception as e:
        print(f"Ocurrió un error inesperado durante el experimento {i+1}: {e}")
        break


    # 3. (Opcional pero Recomendado) Renombrar/Mover archivos de resultados
    #    Tu script main_se.py ya genera nombres de archivo con fecha y configuración.
    #    Así que esto podría no ser estrictamente necesario si los nombres ya son únicos.
    #    Pero si quieres una estructura de carpetas más organizada:
    #
    #    results_dir = f"experimentos_output/exp_{i+1}_{metodo_extraccion}_{pubmed_restructure}_{reranker_type}_{reranker_query}"
    #    os.makedirs(results_dir, exist_ok=True)
    #    
    #    # Asume que main_se.py guarda en OUTPUT_DATA_FOLDER y FOLDER_METRICS
    #    # Necesitarías saber el nombre exacto del archivo generado (con la fecha y hora)
    #    # Esto es un poco más complejo porque necesitas listar los archivos más recientes
    #    # o modificar main_se.py para que devuelva los nombres de archivo.
    #
    #    # Ejemplo simplificado si los nombres son predecibles (sin timestamp):
    #    # shutil.move(f"data/result_data/train/TRAIN_results_{...}.json", results_dir)
    #    # shutil.move(f"data/result_data/metrics/TRAIN_metrics_{...}.json", results_dir)
    #    # shutil.move(f"data/result_data/metrics/TRAIN_metrics_graph_{...}.png", results_dir)

    # Pausa breve entre experimentos (opcional, para no sobrecargar APIs si aplica)
    print("Pausando 5 segundos antes del siguiente experimento...")
    time.sleep(5)


print("\n" + "="*80)
print("Todos los experimentos han finalizado.")
print("="*80)