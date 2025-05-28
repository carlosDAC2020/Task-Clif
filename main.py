import json
import time
import os
import numpy as np  # Si usas numpy directamente aquí
import pandas as pd  # Si usas pandas directamente aquí
import matplotlib.pyplot as plt
from dotenv import load_dotenv

# Modulos locales
from services.Gemini import GeminiService
from services.PubMed import PubMedService
from services.Word_list import WordListExtractor
from processing.Reranker import Reranker
from processing.utils import extract_id
from processing import metrics as mtrs

# Importaciones de configuración y placeholders (ajusta según tu estructura real)
import google.generativeai as genai
from datetime import datetime
import argparse

# Argument parser
parser = argparse.ArgumentParser(description='Run search engine with custom settings')
parser.add_argument('--type-evaluation', type=str, default='TRAIN', 
                    choices=['TRAIN', 'TEST'], help='Evaluation type')
parser.add_argument('--reranker-type', type=str, default='BM25',
                    choices=['TF-IDF', 'BM25', 'PubMedBERT'], help='Reranker type')
parser.add_argument('--reranker-query', type=str, default='BODY',
                    choices=['BODY', 'BODY + KEYWORDS', 'KEYWORDS'], help='Query type for reranking')
parser.add_argument('--metodo-extraccion', type=str, default='WEIRD',
                    choices=['LLM', 'WEIRD'], help='Keyword extraction method')
args = parser.parse_args()

# Use the arguments instead of environment variables
TYPE_EVALUATION = args.type_evaluation
RERANKER_TYPE = args.reranker_type
RERANKER_QUERY = args.reranker_query
METODO_EXTRACCION = args.metodo_extraccion

# load_dotenv()  # Carga las variables desde el archivo .env

# # Configuracion de parameyros generales de evaluacion ------------
# # Cambia a "TRAIN" o "TEST" según sea necesario en el .env
# TYPE_EVALUATION = os.getenv("TYPE_EVALUATION")
# # Tipo de re-ranking a usar (ej: TF-IDF,  BM25  o PubMedBERT)
# RERANKER_TYPE = os.getenv("RERANKER_TYPE")
# # Campo deL tipo de query a usar para el re-ranking el cual puede ser BODY, BODY + KEYWORDS o KEYWORDS
# RERANKER_QUERY = os.getenv("RERANKER_QUERY")
# # Método de extracción inicial de keywords (LLM o WEIRD)
# METODO_EXTRACCION = os.getenv("METODO_EXTRACCION")

# --- Configuración API Keys ---
print("--- Configurando API Keys ---")
pubmed_api_key = os.getenv("PUBMED_API_KEY")
print(
    f"PubMed API Key: {'*' * (len(pubmed_api_key)-4) + pubmed_api_key[-4:] if pubmed_api_key else 'No encontrada'}"
)

gemini_api_key = os.getenv("GEMINI_API_KEY")
print(
    f"Gemini API Key: {'*' * (len(gemini_api_key)-4) + gemini_api_key[-4:] if gemini_api_key else 'No encontrada'}"
)

# --- Configurar Clientes/Servicios ---
print("\n--- Configurando Servicios ---")
try:
    # Configurar el servicio de Gemini
    genai.configure(api_key=gemini_api_key)
    gemini_model_name = "gemini-1.5-flash-latest"
    safety_settings_config = [
        {"category": cat, "threshold": "BLOCK_NONE"}
        for cat in [
            "HARM_CATEGORY_HARASSMENT",
            "HARM_CATEGORY_HATE_SPEECH",
            "HARM_CATEGORY_SEXUALLY_EXPLICIT",
            "HARM_CATEGORY_DANGEROUS_CONTENT",
        ]
    ]
    gemini_llm_instance = genai.GenerativeModel(
        gemini_model_name, safety_settings=safety_settings_config
    )
    # Instanciar el servicio Gemini
    gemini_service = GeminiService(llm_model=gemini_llm_instance)
    print(f"Cliente de Gemini configurado con el modelo: {gemini_model_name}")

    # Instanciar el servicio PubMed
    pubmed_service = PubMedService(api_key=pubmed_api_key)
    print("Cliente de PubMed configurado.")

    # Instanciar el servicio de re-ranking
    reranker = Reranker()
    print("Cliente de Reranker configurado.")
# Use code with caution.
except Exception as e:
    print(f"Error configurando los servicios: {e}. Terminando script.")
    gemini_llm_instance = None

# --- Clase LTV Local (Placeholder - REEMPLAZAR CON TU CLASE REAL) ---
# print("\n--- Configurando Clasificador Local (Placeholder) ---")
# Asume que ner_pipeline y pos_pipeline están definidos o cargados antes
ner_pipeline = None
pos_pipeline = None


# ---- SIMULACIÓN ----
class LTV_Entitye_Clasifier_Local:  # Placeholder
    def __init__(self, ner, pos):
        print("Placeholder LTV_Entitye_Clasifier_Local inicializado.")

    def get(self, text):
        print(f"Placeholder LTV.get() llamado para: '{text[:50]}...'")
        return {"keywords": ["placeholder_kw1", "local_term"]}


classifier = LTV_Entitye_Clasifier_Local(ner_pipeline, pos_pipeline)
# ---- FIN SIMULACIÓN ----

# --- Listas de Stop Words ---
stop_words_generales = set(
    [
        "is",
        "a",
        "the",
        "and",
        "or",
        "of",
        "to",
        "in",
        "it",
        "that",
        "this",
        "for",
        "what",
        "can",
        "be",
        "an",
        "are",
        "do",
        "does",
        "did",
        "have",
        "has",
        "had",
        "was",
        "were",
        "will",
        "how",
        "why",
        "when",
        "where",
        "which",
        "who",
        "with",
        "as",
        "at",
        "by",
        "from",
        "if",
        "into",
        "like",
        "near",
        "on",
        "onto",
        "out",
        "over",
        "past",
        "than",
        "then",
        "through",
        "under",
        "until",
        "up",
        "upon",
        "without",
        "you",
        "your",
        "i",
        "me",
        "my",
        "we",
        "our",
        "us",
        "he",
        "him",
        "his",
        "she",
        "her",
        "they",
        "them",
        "their",
        "list",
        "describe",
        "main",
        "give",
    ]
)
stop_words_medicas_genericas = set(
    [
        "disease",
        "disorder",
        "syndrome",
        "patient",
        "study",
        "analysis",
        "results",
        "review",
        "case",
        "report",
        "effect",
        "treatment",
        "therapy",
        "clinical",
        "trial",
        "evidence",
        "role",
        "mechanism",
        "approach",
        "management",
        "association",
        "factor",
        "risk",
        "level",
        "group",
        "use",
        "related",
        "potential",
        "impact",
        "efficacy",
        "safety",
        "comparison",
        "development",
        "evaluation",
        "assessment",
        "status",
        "marker",
        "expression",
        "pathway",
        "interaction",
        "cell",
        "gene",
        "protein",
        "receptor",
        "inhibitor",
        "agonist",
        "antagonist",
    ]
)

# --- Configuración para el tipo de evaluacion ---

# --- Carga de Datos ---
print("\n--- Cargando Datos ---")
# Lista para almacenar todas las preguntas
todas_las_preguntas = []

# Recorrer los archivos en la carpeta
DATA_FOLDER = "data"
RESULT_FOLDER_TRAIN = f"{DATA_FOLDER}/result_data/train"
RESULT_FOLDER_TEST = f"{DATA_FOLDER}/result_data/test"
TEST_DATA_FOLDER = f"{DATA_FOLDER}/test_data"
TRAIN_DATA_FOLDER = f"{DATA_FOLDER}/train_data"

# Definir la carpeta de entrada y salida según el tipo de evaluación
INPUT_DATA_FOLDER = (
    TRAIN_DATA_FOLDER if TYPE_EVALUATION == "TRAIN" else TEST_DATA_FOLDER
)
print(f"Carpeta de entrada: {INPUT_DATA_FOLDER}")
OUTPUT_DATA_FOLDER = (
    RESULT_FOLDER_TRAIN if TYPE_EVALUATION == "TRAIN" else RESULT_FOLDER_TEST
)
print(f"Carpeta de salida: {OUTPUT_DATA_FOLDER}")

# Cargar datosm de entrada
for archivo in os.listdir(INPUT_DATA_FOLDER):
    if archivo.endswith(".json"):  # Verificar que es un archivo JSON
        ruta_completa = os.path.join(INPUT_DATA_FOLDER, archivo)
        print(f"Procesando archivo: {archivo}")
        # Cargar el contenido del archivo JSON
        with open(ruta_completa, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
                # Extraer la lista de preguntas si existe
                if "questions" in data and isinstance(data["questions"], list):
                    todas_las_preguntas.extend(data["questions"])
            except json.JSONDecodeError:
                print(f"Error al leer el archivo: {archivo}")

print(f"Total de preguntas: {len(todas_las_preguntas)}")

# --- Bucle Principal de evaluacion ---
# === CONFIGURACIÓN ===
limite = len(todas_las_preguntas)  # Procesar todo por defecto
USAR_FALLBACK_LOCAL = True  # ¿Intentar clasificador local si LLM falla?
USAR_REINTENTO_LLM = (
    True if gemini_service.llm_model else False
)  # ¿Reintentar búsqueda con LLM si la inicial falla?
CORE_TERMS_INICIAL = (
    1  # Core terms para la búsqueda híbrida inicial (cuando LLM o Local funcionó)
)
CORE_TERMS_LOCAL = (
    2  # Core terms si se usó solo el clasificador local (quizás menos preciso)
)
MAX_RESULTS_PUBMED_SEARCH = 500  # Máximos resultados a pedir a PubMed (ESearch)
# PAUSA_PUBMED y PAUSA_GEMINI ahora se gestionan dentro de las clases Service
SNIPPETS_TOP_N = 10  # Cuántos snippets/documentos incluir finalmente
MAX_SNIPPET_LEN = 250  # Longitud máxima del texto del snippet
PAUSA_ENTRE_ITEMS = 1.0  # Pausa en segundos entre procesar cada pregunta (para evitar rate limits generales)
# ====================

resultados_finales_json = {"questions": []}
tiempos_totales = []
num_items_procesados = 0
num_fallos_extraccion_total = 0
num_filtrado_cero = 0
num_busqueda_cero_pmids = 0
num_reintentos_llm_iniciados = 0
num_reintentos_llm_fallidos_output = 0
num_efetch_fallidos_parcial = 0  # Contar items donde EFetch falló para algunos PMIDs


for i, item_original in enumerate(todas_las_preguntas):
    if i >= limite:
        break  # Control de límite
    num_items_procesados += 1
    item_id = item_original.get("id", f"unknown_{i+1}")
    item_body = item_original.get("body", "")

    print(f"\n--- Procesando ID: {item_id} - Item {i+1}/{limite} ---")

    if not item_body:
        print(f"[!] Saltando ID: {item_id} (Body vacío)")
        output_item = item_original.copy()
        output_item["documents"] = []
        output_item["snippets"] = []
        resultados_finales_json["questions"].append(output_item)
        if PAUSA_ENTRE_ITEMS > 0:
            time.sleep(PAUSA_ENTRE_ITEMS)
        os.system("cls" if os.name == "nt" else "clear")
        continue

    print(f"Texto: {item_body[:200]}...")  # Mostrar inicio del texto
    item_start_time = time.time()

    keywords_para_filtrar = []
    metodo_exitoso_extraccion = "N/A"

    # --- PASO 1: Extracción Inicial de Keywords ---
    start_extr = time.time()
    if METODO_EXTRACCION == "LLM":
        # Usar el servicio Gemini
        keywords_llm = gemini_service.extract_keywords(item_body)
        print(keywords_llm)
        if keywords_llm:
            keywords_para_filtrar = keywords_llm
            metodo_exitoso_extraccion = "LLM"
        elif USAR_FALLBACK_LOCAL:
            print("[!] LLM falló o no devolvió keywords. Intentando Fallback Local...")
            metodo_exitoso_extraccion = (
                "LLM_FALLIDO_A_LOCAL"  # Marcar para intentar local
            )
        else:
            print("[!] LLM falló. Sin fallback configurado.")
            metodo_exitoso_extraccion = "LLM_FALLIDO"
            num_fallos_extraccion_total += 1
    if METODO_EXTRACCION == "WEIRD":
        extractor = WordListExtractor()
        sentence = item_body
        keywords_llm = extractor.extract_word_list_from_sentence(
            sentence, weirdness_threshold=10
        )
        print(keywords_llm)
        if keywords_llm:
            keywords_para_filtrar = keywords_llm
            metodo_exitoso_extraccion = "LLM"
        elif USAR_FALLBACK_LOCAL:
            print("[!] LLM falló o no devolvió keywords. Intentando Fallback Local...")
            metodo_exitoso_extraccion = (
                "LLM_FALLIDO_A_LOCAL"  # Marcar para intentar local
            )
        else:
            print("[!] LLM falló. Sin fallback configurado.")
            metodo_exitoso_extraccion = "LLM_FALLIDO"
            num_fallos_extraccion_total += 1

    # --- PASO 1.5: Fallback o Ejecución Local Directa ---
    # Ejecutar si el método es 'LOCAL' o si LLM falló y el fallback está activo
    if (
        metodo_exitoso_extraccion == "LOCAL"
        or metodo_exitoso_extraccion == "LLM_FALLIDO_A_LOCAL"
    ):
        try:
            # Usar el clasificador local (Placeholder o real)
            result_body = classifier.get(
                item_body
            )  # Asume que esto es rápido y no necesita pausa larga
            keywords_local = result_body.get("keywords", [])
            if keywords_local:
                keywords_para_filtrar = keywords_local
                # Actualizar el método exitoso si vino de fallback
                metodo_exitoso_extraccion = (
                    "LOCAL (Fallback)"
                    if metodo_exitoso_extraccion == "LLM_FALLIDO_A_LOCAL"
                    else "LOCAL"
                )
            else:
                print("[!] Extracción local no devolvió keywords.")
                # Si era fallback y falló, contar como fallo total de extracción
                if metodo_exitoso_extraccion == "LOCAL (Fallback)":
                    num_fallos_extraccion_total += 1
                metodo_exitoso_extraccion += "_FALLIDO"  # Marcar que falló
        except Exception as e:
            print(f"[!] Error durante ejecución del clasificador local: {e}")
            if metodo_exitoso_extraccion == "LOCAL (Fallback)":
                num_fallos_extraccion_total += 1
            metodo_exitoso_extraccion += "_FALLIDO"  # Marcar que falló

    end_extr = time.time()
    print(
        f"[*] Keywords Extraídos ({metodo_exitoso_extraccion}): {keywords_para_filtrar} (Tiempo Extr: {end_extr - start_extr:.2f}s)"
    )

    # Si no hay keywords después de intentar todo, crear item vacío y continuar
    if not keywords_para_filtrar:
        print(
            f"[!] No se pudieron extraer keywords para {item_id}. Creando entrada vacía."
        )
        output_item = item_original.copy()
        output_item["documents"] = []
        output_item["snippets"] = []
        resultados_finales_json["questions"].append(output_item)
        item_end_time = time.time()
        tiempos_totales.append(item_end_time - item_start_time)
        print(
            f"[*] Item {item_id} procesado (vacío) en {item_end_time - item_start_time:.2f}s."
        )
        print("=" * 80)
        if PAUSA_ENTRE_ITEMS > 0:
            time.sleep(PAUSA_ENTRE_ITEMS)
        os.system("cls" if os.name == "nt" else "clear")  # Limpiar consola (opcional)
        continue

    # --- PASO 2: Filtrado de Keywords y Búsqueda Inicial en PubMed ---
    # Aplicar stop words y filtro básico
    palabras_filtradas = [
        p.lower()
        for p in keywords_para_filtrar
        if isinstance(p, str)  # Asegurar que es string
        and p.lower() not in stop_words_generales
        and p.lower() not in stop_words_medicas_genericas
        and len(p) > 1  # Longitud mínima
        and any(c.isalnum() for c in p)  # Al menos un alfanumérico
    ]
    # Mantener orden y unicidad
    palabras_unicas_ordenadas = list(dict.fromkeys(palabras_filtradas))
    print(f"[*] Keywords Filtrados para PubMed (Inicial): {palabras_unicas_ordenadas}")

    kws_usados_en_busqueda = palabras_unicas_ordenadas  # Guardar qué se usó
    resultados_pubmed_inicial = {"count": 0, "pmids": []}
    pmids_encontrados_inicial = []
    necesita_reintento_busqueda = False
    consulta_usada_finalmente = "Híbrida (Inicial)"  # Tipo de consulta usada

    if not palabras_unicas_ordenadas:
        print("[!] No quedaron términos válidos tras filtrar para buscar en PubMed.")
        num_filtrado_cero += 1
        necesita_reintento_busqueda = (
            True  # Marcar para posible reintento LLM aunque no hubiera keywords
        )
    else:
        # Decidir cuántos core terms usar basado en el éxito de la extracción
        num_core = (
            CORE_TERMS_INICIAL
            if "FALLIDO" not in metodo_exitoso_extraccion
            else CORE_TERMS_LOCAL
        )
        # Usar el servicio PubMed
        resultados_pubmed_inicial = pubmed_service.search_hybrid(
            terms_list=palabras_unicas_ordenadas,
            num_core_terms=num_core,
            max_results=MAX_RESULTS_PUBMED_SEARCH,
            # search_field=None, # Por defecto
            # sort_by="relevance" # Por defecto
        )
        pmids_encontrados_inicial = resultados_pubmed_inicial.get("pmids", [])
        if not pmids_encontrados_inicial:
            print("[!] Búsqueda Híbrida inicial no encontró PMIDs.")
            necesita_reintento_busqueda = True

    # --- PASO 3: Reintento con Reestructuración LLM (si aplica) ---
    pmids_para_procesar = (
        pmids_encontrados_inicial  # Empezar con los resultados iniciales
    )

    if necesita_reintento_busqueda and USAR_REINTENTO_LLM:
        num_reintentos_llm_iniciados += 1
        print(f"[*] Intentando reestructurar consulta con LLM...")
        # Usar el servicio Gemini para reestructurar
        nueva_consulta = gemini_service.restructure_query(
            original_question=item_body,
            initial_keywords=kws_usados_en_busqueda,  # Keywords originales usados
            failure_type="no_pmids",
        )

        if nueva_consulta:
            print(f"[*] Ejecutando búsqueda directa con consulta reestructurada...")
            kws_usados_en_busqueda = [nueva_consulta]  # Guardar la nueva consulta usada
            consulta_usada_finalmente = "Directa (Reintento LLM)"
            # Usar el servicio PubMed para la búsqueda directa
            resultados_pubmed_reintento = pubmed_service.search_direct(
                query_string=nueva_consulta,
                max_results=MAX_RESULTS_PUBMED_SEARCH,
                # sort_by="relevance" # Por defecto
            )  # La pausa está dentro
            pmids_para_procesar = resultados_pubmed_reintento.get(
                "pmids", []
            )  # Actualizar PMIDs a procesar
            if not pmids_para_procesar:
                print("[!] Reintento LLM con búsqueda directa tampoco encontró PMIDs.")
                num_reintentos_llm_fallidos_output += (
                    1  # Contar fallo si no produjo resultados
                )
            else:
                print(f"[*] Reintento LLM encontró {len(pmids_para_procesar)} PMIDs.")
        else:
            print(
                "[!] LLM no pudo generar consulta reestructurada o la consulta era inválida."
            )
            num_reintentos_llm_fallidos_output += (
                1  # Contar fallo si no generó consulta
            )

    elif necesita_reintento_busqueda:
        # Si necesitaba reintento pero no se usó LLM
        print("[!] Búsqueda inicial sin PMIDs y sin reintento LLM configurado/activo.")
        # Contar como fallo de búsqueda si no hubo reintento exitoso
        num_busqueda_cero_pmids += 1

    # Contar como fallo si la búsqueda inicial dio 0 y *no* hubo reintento LLM exitoso
    if not pmids_para_procesar and not (
        necesita_reintento_busqueda and USAR_REINTENTO_LLM
    ):
        # Esto cubre: inicial 0 sin reintento, o inicial 0 con reintento fallido
        if (
            not necesita_reintento_busqueda
        ):  # Si la inicial ya dio 0 y no se marcó reintento
            num_busqueda_cero_pmids += 1
        # El caso de reintento fallido ya se contó arriba, evitar doble conteo
    elif not pmids_para_procesar and necesita_reintento_busqueda and USAR_REINTENTO_LLM:
        # Si llegamos aquí, el reintento se hizo pero falló en encontrar PMIDs
        # num_reintentos_llm_fallidos_output ya lo contó.
        # Podríamos añadirlo a num_busqueda_cero_pmids también si queremos totalizar búsquedas fallidas.
        pass  # Ya contado como reintento fallido

    # --- PASO 4: Obtener Detalles (EFetch) y Re-ranking ---
    pmid_details = {}
    pmids_finales_reranked = []
    pmid_scores_map = {}

    if pmids_para_procesar:
        print(
            f"[*] Obteniendo detalles para {len(pmids_para_procesar)} PMIDs vía EFetch..."
        )
        # Usar el servicio PubMed para fetch
        pmid_details = pubmed_service.fetch_details(
            pmids_para_procesar
        )  # Pausa y reintentos dentro

        # Verificar si EFetch recuperó detalles para todos los PMIDs solicitados
        if len(pmid_details) < len(pmids_para_procesar):
            print(
                f"[!] Advertencia: EFetch recuperó detalles para {len(pmid_details)} de {len(pmids_para_procesar)} PMIDs."
            )
            num_efetch_fallidos_parcial += 1  # Contar como fallo parcial

        if pmid_details:  # Solo re-rankear si obtuvimos algún detalle
            print(f"[*] Realizando re-ranking con {RERANKER_TYPE}...")

            # estructuramos el ripo de query para el re-ranking
            if RERANKER_QUERY == "BODY":
                query_rerank = item_body
            elif RERANKER_QUERY == "KEYWORDS":
                query_rerank = " ".join(palabras_unicas_ordenadas)
            elif RERANKER_QUERY == "BODY + KEYWORDS":
                query_rerank = item_body + " " + " ".join(palabras_unicas_ordenadas)

            # Usar el servicio Reranker
            if RERANKER_TYPE == "TF-IDF":
                pmids_finales_reranked, pmid_scores_map = reranker.rerank_TF_IDF(
                    original_question=query_rerank, pmid_details_map=pmid_details
                )
            elif RERANKER_TYPE == "BM25":
                pmids_finales_reranked, pmid_scores_map = reranker.rerank_bm25(
                    original_question=query_rerank, pmid_details_map=pmid_details
                )
            elif RERANKER_TYPE == "PubMedBERT":
                pmids_finales_reranked, pmid_scores_map = reranker.rerank_pubmedbert(
                    original_question=query_rerank, pmid_details_map=pmid_details
                )
        else:
            print(
                "[!] No se obtuvieron detalles de PubMed (EFetch falló completamente?), no se puede re-rankear."
            )
            # Mantener los PMIDs originales si EFetch falló pero la búsqueda no
            pmids_finales_reranked = pmids_para_procesar
            pmid_scores_map = {
                pmid: 0.0 for pmid in pmids_finales_reranked
            }  # Scores a cero

    else:
        print(
            "[!] No hay PMIDs para obtener detalles o reordenar (Búsqueda inicial/reintento falló)."
        )

    # --- PASO 5: Construir el objeto de salida JSON ---
    output_item = item_original.copy()
    # Tomar los TOP N reordenados (o los que haya si son menos de N)
    final_documents_list = pmids_finales_reranked[:SNIPPETS_TOP_N]
    output_item["documents"] = [
        f"http://www.ncbi.nlm.nih.gov/pubmed/{pmid}" for pmid in final_documents_list
    ]  # Formato URL
    output_item["snippets"] = []

    print(
        f"[*] Generando snippets para los Top {len(final_documents_list)} documentos..."
    )
    for rank, pmid in enumerate(final_documents_list, 1):
        details = pmid_details.get(
            str(pmid)
        )  # Asegurar que pmid es string para buscar en dict
        snippet_text = "[Details unavailable]"
        begin_sec, end_sec = "N/A", "N/A"
        offset_end = 0

        if details:
            title = details.get("title", "")
            abstract = details.get("abstract", "")

            # Priorizar título + abstract si ambos existen
            if title and title != "N/A" and abstract and abstract != "N/A":
                full_text = f"{title}. {abstract}"
                begin_sec, end_sec = (
                    "abstract",
                    "abstract",
                )  # O considerar "title_abstract"
            elif title and title != "N/A":
                full_text = title
                begin_sec, end_sec = "title", "title"
            elif abstract and abstract != "N/A":
                full_text = abstract
                begin_sec, end_sec = "abstract", "abstract"
            else:
                full_text = "[Title/Abstract N/A]"  # Texto si no hay nada útil

            # Truncar el texto para el snippet
            snippet_text = full_text[:MAX_SNIPPET_LEN]
            if len(full_text) > MAX_SNIPPET_LEN and MAX_SNIPPET_LEN > 3:
                snippet_text = snippet_text[:-3] + "..."  # Añadir elipsis si se truncó

            offset_end = len(snippet_text)  # Longitud final del snippet

        snippet_entry = {
            "document": f"http://www.ncbi.nlm.nih.gov/pubmed/{pmid}",  # Usar URL también aquí
            "text": snippet_text,
            "beginSection": begin_sec,
            "endSection": end_sec,
            "offsetInBeginSection": 0,  # Asumimos inicio de sección
            "offsetInEndSection": offset_end,
            # Añadir info extra si se desea (opcional)
            "rerank_score": pmid_scores_map.get(str(pmid), 0.0),  # Score del reranking
            "rerank_position": rank,  # Posición tras reranking
        }
        output_item["snippets"].append(snippet_entry)

    # Añadir el item procesado a la lista final
    resultados_finales_json["questions"].append(output_item)

    # --- Fin del Procesamiento del Item ---
    item_end_time = time.time()
    item_duration = item_end_time - item_start_time
    tiempos_totales.append(item_duration)
    print(
        f"[*] Item {item_id} procesado en {item_duration:.2f}s. Consulta: {consulta_usada_finalmente}. Documentos/Snippets en JSON: {len(output_item['documents'])}"
    )
    print("\n" + "=" * 80)

    # Pausa general entre items
    if PAUSA_ENTRE_ITEMS > 0:
        print(f"[*] Pausando {PAUSA_ENTRE_ITEMS}s antes del siguiente item...")
        time.sleep(PAUSA_ENTRE_ITEMS)

    # Limpiar output si estás en notebook para mejor visualización
    # os.system("cls" if os.name == "nt" else "clear")  # Limpiar consola (opcional)

# --- Guardar el JSON Final ---
# Obtener la fecha y hora actual para agregar al nombre del archivo
fecha_hora_actual = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
# Actualizar el nombre del archivo de salida con la fecha y hora
output_filename = f"{OUTPUT_DATA_FOLDER}/{TYPE_EVALUATION}_results_{RERANKER_TYPE}_{METODO_EXTRACCION}_{RERANKER_QUERY}{fecha_hora_actual}.json"
# Guardar el archivo JSON final
try:
    output_dir = os.path.dirname(output_filename)
    if output_dir and not os.path.exists(output_dir):
        print(f"Creando directorio de salida: {output_dir}")
        os.makedirs(output_dir)
    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(resultados_finales_json, f, indent=2, ensure_ascii=False)
        print(
            f"\n--- ¡Proceso Completado! Resultados guardados en '{output_filename}' ---"
        )
except Exception as e:
    print(f"\n--- ERROR CRÍTICO al guardar el archivo JSON: {e} ---")

# --- Estadísticas Finales de Ejecución ---
print("\n--- Estadísticas Finales de Ejecución ---")
total_preguntas_input = len(todas_las_preguntas)
print(f"Total Items en Input: {total_preguntas_input}")
print(f"Total Items Intentados Procesar (límite={limite}): {num_items_procesados}")
print("-" * 20)
print(
    f"Fallos Totales en Extracción Inicial de Keywords: {num_fallos_extraccion_total} ({num_fallos_extraccion_total / num_items_procesados if num_items_procesados else 0:.1%})"
)
print(
    f"Items con Cero Keywords Válidos Post-Filtrado: {num_filtrado_cero} ({num_filtrado_cero / num_items_procesados if num_items_procesados else 0:.1%})"
)

# Total búsquedas que terminaron sin PMIDs (inicial sin reintento O reintento fallido)
total_busquedas_sin_pmids = num_busqueda_cero_pmids + num_reintentos_llm_fallidos_output
print(
    f"Items Resultando en Cero PMIDs Tras Búsqueda(s): {total_busquedas_sin_pmids} ({total_busquedas_sin_pmids / num_items_procesados if num_items_procesados else 0:.1%})"
)
print("-" * 20)
print(f"Reintentos de Búsqueda con LLM Iniciados: {num_reintentos_llm_iniciados}")
print(
    f"Reintentos LLM Fallidos (No generó consulta o no encontró PMIDs): {num_reintentos_llm_fallidos_output}"
)
print(
    f"Items con Fallos Parciales en EFetch (No todos los detalles recuperados): {num_efetch_fallidos_parcial}"
)
print("-" * 20)
if tiempos_totales:
    tiempo_total_s = sum(tiempos_totales)
    tiempo_promedio_s = tiempo_total_s / len(tiempos_totales)
    print(f"Tiempo Total de Procesamiento: {tiempo_total_s:.2f}s")
    print(f"Tiempo promedio TOTAL por item procesado: {tiempo_promedio_s:.2f}s")
else:
    print("No se procesaron items completos para calcular tiempo promedio.")

# Imprimir muestra del primer resultado generado (si existe)
if resultados_finales_json["questions"]:
    print("\n--- Muestra del primer resultado generado: ---")
    # Usar json.dumps para pretty print en consola
    print(
        json.dumps(
            resultados_finales_json["questions"][0], indent=2, ensure_ascii=False
        )
    )
else:
    print("\nNo se generaron resultados para mostrar una muestra.")

# --- Bucle de evaluacion de resultados ---
if TYPE_EVALUATION == "TRAIN":
    print("\n--- Iniciando Evaluación de Resultados ---")
    # dataframe de resultados
    df_resultados = pd.DataFrame(resultados_finales_json["questions"])
    # dataframe de preguntas evaluadas
    df_preguntas = pd.DataFrame(todas_las_preguntas)
    # Metricas generales obtenidas
    metrics = {
        "questions": [],
        "keywords_stract": "LLM Gemini",
        "Ranked": RERANKER_TYPE,
        "Query_ranked": RERANKER_QUERY,
        "metrics": {
            "S@10": 0,
            "P@10": 0,
            "R@10": 0,
            "F1@10": 0,
            "MAP@10": 0,
            "MRR": 0,
            "NDCG@10": 0,
        },
    }

    # recorrer los resultsos e ir comparando con lo esperado de las preguntas
    for i, item in df_resultados.iterrows():
        # Obtener el ID de la pregunta
        item_id = item["id"]
        # Filtrar el dataframe de preguntas para encontrar la pregunta correspondiente
        pregunta = df_preguntas[df_preguntas["id"] == item_id]

        if not pregunta.empty:
            docs_esperados = [extract_id(url) for url in pregunta.iloc[0]["documents"]]
            docs_obtenidos = [extract_id(url) for url in item["documents"]]
            if len(docs_esperados) == 0:
                docs_esperados = ["0"]

            print("obtenidos:", docs_obtenidos)
            print("esperados:", docs_esperados)

            # Cálculo de métricas individuales
            s10 = mtrs.success_at_k(docs_obtenidos, docs_esperados)
            p10 = mtrs.precision_at_k(docs_obtenidos, docs_esperados)
            r10 = mtrs.recall_at_k(docs_obtenidos, docs_esperados)
            f1_10 = mtrs.f1_at_k(docs_obtenidos, docs_esperados)

            try:
                map_10 = mtrs.mean_average_precision(docs_obtenidos, docs_esperados)
            except:
                map_10 = 0
                mrr = mtrs.mean_reciprocal_rank(docs_obtenidos, docs_esperados)
            try:
                ndcg_10 = mtrs.ndcg_at_k(docs_obtenidos, docs_esperados)
            except:
                ndcg_10 = 0

            print(f"S@10: {s10}")
            print(f"P@10: {p10}")
            print(f"R@10: {r10}")
            print(f"F1@10: {f1_10}")
            print(f"MAP@10: {map_10}")
            print(f"MRR: {mrr}")
            print(f"NDCG@10: {ndcg_10}")

            # Guardar métricas individuales por consulta
            mtr = {
                "id": item["id"],
                "S@10": s10,
                "P@10": p10,
                "R@10": r10,
                "F1@10": f1_10,
                "MAP@10": map_10,
                "MRR": mrr,
                "NDCG@10": ndcg_10,
            }

            metrics["questions"].append(mtr)

            # Acumular valores para la media general
            metrics["metrics"]["S@10"] += s10
            metrics["metrics"]["P@10"] += p10
            metrics["metrics"]["R@10"] += r10
            metrics["metrics"]["F1@10"] += f1_10
            metrics["metrics"]["MAP@10"] += map_10
            metrics["metrics"]["MRR"] += mrr
            metrics["metrics"]["NDCG@10"] += ndcg_10

            # os.system("cls" if os.name == "nt" else "clear")

    # Cálculo de métricas generales evitando la división por cero
    total_queries = len(metrics["questions"])
    if total_queries > 0:
        for key in metrics["metrics"]:
            metrics["metrics"][key] /= total_queries

    # Guardado de resultados
    FOLDER_METRICS = "data/result_data/metrics"
    ruta_guardado = f"{FOLDER_METRICS}/{TYPE_EVALUATION}_metrics_{RERANKER_TYPE}_{RERANKER_QUERY}_{METODO_EXTRACCION}.json"
    with open(ruta_guardado, "w") as archivo:
        json.dump(metrics, archivo, indent=2)

    print(f"Resultados guardados en {ruta_guardado}")

    print("--- Evaluación Completada ---")

# mostrar graficos de metricas --------------------

    # Extraer métricas generales
    metricas_generales = metrics["metrics"]

    # Gráfica de barras para métricas generales
    plt.figure(figsize=(10, 6))
    plt.bar(metricas_generales.keys(), metricas_generales.values(), color="skyblue")
    plt.xlabel("Métrica")
    plt.ylabel("Valor")
    plt.title("Desempeño General del Modelo")
    plt.xticks(rotation=45)
    plt.ylim(0, 1)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    # gyardar la grafica
    plt.savefig(
        f"{FOLDER_METRICS}/{TYPE_EVALUATION}_metrics_{RERANKER_TYPE}_{fecha_hora_actual}.png"
    )
