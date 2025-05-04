import time
import re
import ast
import google.generativeai as genai # Asegúrate de que genai esté importado

# ======================================================================
#  CLASE DE SERVICIO PARA GOOGLE GEMINI
# ======================================================================
class GeminiService:
    """
    Gestiona las interacciones con la API de Google Gemini para extraer
    keywords y reestructurar consultas PubMed.
    """
    def __init__(self, llm_model, max_retries=3, pause_after_call=3.0):
        """
        Inicializa el servicio Gemini.

        Args:
            llm_model: La instancia preconfigurada de genai.GenerativeModel.
            max_retries (int): Número máximo de reintentos en caso de error.
            pause_after_call (float): Segundos de pausa después de una llamada exitosa a la API.
        """
        self.llm_model = llm_model
        self.max_retries = max_retries
        self.pause_after_call = pause_after_call
        if not self.llm_model:
            print("ADVERTENCIA [GeminiService]: Modelo LLM no proporcionado o inválido. Funcionalidad limitada.")

    def extract_keywords(self, question_body):
        """
        Extrae keywords biomédicas de una pregunta usando Gemini.

        Args:
            question_body (str): El texto de la pregunta.

        Returns:
            list: Una lista de strings (keywords) o una lista vacía si falla.
        """
        if not self.llm_model:
            print("ERROR [GeminiService]: No se puede extraer keywords, modelo no disponible.")
            return []

        # --- Prompt (Asegúrate de que sea el optimizado que tenías) ---
        prompt_text = f"""
        You are an expert biomedical researcher specialized in creating effective PubMed search strategies. Your task is to analyze the user's question and identify the core concepts, entities, and keywords most suitable for a PubMed query.

        Instructions:
        1. Focus on specific biomedical terms: Genes, proteins, diseases, disorders, symptoms, drugs, chemicals, techniques, procedures, organisms, key biological processes.
        2. Be concise. Extract only the most essential terms needed to capture the question's intent.
        3. If a standard abbreviation or acronym exists (e.g., 'BiFC', 'HILIC', 'EGFR', 'CKD'), prefer it. If easily known, also include its full name as a separate keyword if relevant (e.g., 'AMD', 'Age-related Macular Degeneration').
        4. If a multi-word concept is crucial (e.g., 'chronic kidney disease', 'Hemolytic Uremic Syndrome'), keep the phrase intact as a single keyword string.
        5. Do NOT include common English stop words (like 'what', 'are', 'effects', 'list', 'main', 'role', 'impact') unless part of a specific technical term.
        6. Format the output STRICTLY as a Python list of strings. For example: ['keyword1', 'keyword phrase 2', 'keyword3']
        7. Do NOT include any other text, explanation, or formatting before or after the Python list.

        --- Example 1 ---
        Question: "List symptoms of the IFAP syndrome."
        Python List: ['IFAP syndrome', 'symptoms', 'clinical features']

        --- Example 2 ---
        Question: "What are the effects of depleting protein km23-1 (DYNLRB1) in a cell?"
        Python List: ['DYNLRB1', 'protein depletion', 'gene silencing', 'cellular effects']

        --- Current Question ---
        Question: "{question_body}"

        Python List:
        """
        retries = 0
        while retries <= self.max_retries:
            try:
                print(f"[*] [Gemini] Enviando pregunta para extracción (Intento {retries + 1})...")
                time_start = time.time()
                response = self.llm_model.generate_content(
                    prompt_text,
                    generation_config=genai.types.GenerationConfig(
                        # Ajusta tokens y temperatura si es necesario
                        max_output_tokens=150,
                        temperature=0.0
                    )
                )
                time_end = time.time()
                print(f"[*] [Gemini] Respuesta recibida en {time_end - time_start:.2f}s.")

                if self.pause_after_call > 0:
                    print(f"[*] [Gemini] Pausando {self.pause_after_call}s post-llamada...")
                    time.sleep(self.pause_after_call)

                # --- Procesamiento de la Respuesta ---
                if response.prompt_feedback.block_reason:
                     print(f"WARN [Gemini]: Respuesta bloqueada. Razón: {response.prompt_feedback.block_reason}")
                     raise ValueError(f"Blocked by Safety Filter: {response.prompt_feedback.block_reason}")

                if not response.candidates:
                     print("WARN [Gemini]: No se recibieron candidatos en la respuesta.")
                     raise ValueError("No Candidates Returned")

                raw_output = response.text.strip()
                # print(f"Raw LLM Output (Keywords):\n{raw_output}") # Descomentar para depurar

                # Intenta extraer la lista directamente con regex y ast
                match = re.search(r"(\[.*?\])", raw_output, re.DOTALL)
                if match:
                    list_str = match.group(1)
                    try:
                        potential_list = ast.literal_eval(list_str)
                        if isinstance(potential_list, list) and all(isinstance(item, str) for item in potential_list):
                            # Limpieza final opcional (quitar espacios extra)
                            cleaned_list = [kw.strip() for kw in potential_list if kw.strip()]
                            print(f"[*] [Gemini] Lista de keywords parseada: {cleaned_list}")
                            return cleaned_list
                        else:
                            print("WARN [Gemini]: Se encontró string tipo lista, pero el contenido no es válido.")
                    except Exception as parse_err:
                        print(f"WARN [Gemini]: Error parseando la lista encontrada ({parse_err}). Usando fallback.")
                        # Continuar al fallback si el parseo falla

                # Fallback: si no se pudo parsear una lista, intentar dividir el texto
                print("[!] [Gemini Fallback] No se pudo parsear lista. Dividiendo output crudo.")
                # Usar regex para dividir por comas o saltos de línea, limpiando caracteres extra
                potential_kws = re.split(r'[,\n]', raw_output)
                keyword_list = [kw.strip().strip("'\"[] ") for kw in potential_kws if kw.strip().strip("'\"[] ")]
                # Filtrar elementos vacíos o muy cortos después de limpiar
                keyword_list = [kw for kw in keyword_list if len(kw) > 1]
                print(f"[*] [Gemini Fallback] Keywords obtenidos: {keyword_list}")
                return keyword_list # Devolver el resultado del fallback

            except Exception as e:
                print(f"ERROR [GeminiService:extract_keywords] (Intento {retries + 1}): {e}")
                # No reintentar si el error es por API Key inválida u otros errores fatales
                if "API key not valid" in str(e) or "API_KEY_INVALID" in str(e) or "permission" in str(e).lower():
                    print("ERROR FATAL [Gemini]: Problema con API Key o permisos. No se reintentará.")
                    return []
                if "Safety" in str(e) or "Blocked" in str(e):
                     print("WARN [Gemini]: Respuesta bloqueada por seguridad. Reintentando si es posible...")
                     # Permite el reintento para filtros de seguridad, quizás un reintento funcione

                retries += 1
                if retries <= self.max_retries:
                    sleep_time = (2 ** (retries -1)) * 0.5 # Backoff exponencial
                    print(f"[*] [Gemini] Reintentando extracción en {sleep_time:.1f}s...")
                    time.sleep(sleep_time)
                else:
                    print("ERROR [GeminiService]: Máximo de reintentos alcanzado para extracción.")
                    return [] # Fallo final

        return [] # Devolver lista vacía si todos los reintentos fallan

    def restructure_query(self, original_question, initial_keywords, failure_type):
        """
        Sugiere una consulta PubMed alternativa usando Gemini cuando la búsqueda inicial falla.

        Args:
            original_question (str): La pregunta original del usuario.
            initial_keywords (list): Lista de keywords usados en el intento fallido.
            failure_type (str): Razón del fallo (e.g., "no_pmids").

        Returns:
            str: La nueva cadena de consulta sugerida, o None si falla.
        """
        if not self.llm_model:
            print("ERROR [GeminiService]: No se puede reestructurar consulta, modelo no disponible.")
            return None

        # Construir la razón de forma legible
        if failure_type == "no_pmids":
            reason = "The initial search using the keywords did not return any PubMed results."
        else:
            reason = f"The initial search failed (Reason: {failure_type})."

        # --- Prompt para Reestructuración ---
        prompt_text = f"""
        You are an expert PubMed search strategist assisting a biomedical researcher.
        The initial attempt to find relevant articles for the following question failed.

        Original Question: "{original_question}"
        Initial Keywords Used (may be imperfect): {initial_keywords}
        Reason for Failure: {reason}

        Your Task:
        Analyze the original question and the initial failure. Generate ONE single, alternative, potentially more effective PubMed query string to try next.
        Consider these strategies:
        - Using more specific or broader terms.
        - Adding or removing terms.
        - Using boolean operators (AND, OR, NOT) strategically.
        - Suggesting MeSH terms (e.g., "Disease Name"[MeSH Terms]).
        - Searching in specific fields (e.g., (term1 AND term2)[Title/Abstract]).
        - Correcting potential misunderstandings of the initial keywords.

        Output ONLY the optimized PubMed query string. Do not include explanations, introductory phrases like "Optimized PubMed Query:", or any other text. Just the query string itself.

        Example Output 1:
        ("COVID-19"[MeSH Terms] OR "SARS-CoV-2"[Title/Abstract]) AND ("question answering" OR "information retrieval") AND challenge

        Example Output 2:
        (hemolytic uremic syndrome[MeSH Terms]) AND (treatment OR therapy)

        Optimized PubMed Query String:
        """
        retries = 0
        # Usar un max_retries menor para reestructuración, quizás 1
        restructure_max_retries = min(self.max_retries, 1)

        while retries <= restructure_max_retries:
            try:
                print(f"[*] [Gemini] Pidiendo reestructurar consulta (Intento {retries + 1})...")
                time_start = time.time()
                response = self.llm_model.generate_content(
                    prompt_text,
                    generation_config=genai.types.GenerationConfig(
                        # Ajusta tokens y temperatura si es necesario
                        max_output_tokens=250,
                        temperature=0.2 # Ligeramente más creativo
                    )
                )
                time_end = time.time()
                print(f"[*] [Gemini] Sugerencia de consulta recibida en {time_end - time_start:.2f}s.")

                if self.pause_after_call > 0:
                    print(f"[*] [Gemini] Pausando {self.pause_after_call}s post-llamada...")
                    time.sleep(self.pause_after_call)

                # --- Procesamiento de la Respuesta ---
                if response.prompt_feedback.block_reason:
                     print(f"WARN [Gemini]: Respuesta de reestructuración bloqueada. Razón: {response.prompt_feedback.block_reason}")
                     raise ValueError(f"Blocked by Safety Filter: {response.prompt_feedback.block_reason}")

                if not response.candidates:
                     print("WARN [Gemini]: No se recibieron candidatos para reestructuración.")
                     raise ValueError("No Candidates Returned for Restructure")

                new_query_string = response.text.strip()

                # Limpiar prefijos comunes que el LLM podría añadir a pesar de las instrucciones
                prefixes_to_remove = ["Optimized PubMed Query String:", "Optimized PubMed Query:", "PubMed Query:", "Query:"]
                for prefix in prefixes_to_remove:
                    if new_query_string.lower().startswith(prefix.lower()):
                        new_query_string = new_query_string[len(prefix):].strip()

                # Quitar comillas/apóstrofes iniciales/finales si el LLM las añade alrededor de toda la consulta
                new_query_string = new_query_string.strip('"\'')

                print(f"[*] [Gemini] Nueva consulta sugerida: {new_query_string}")

                # Validación básica de la consulta (no vacía, longitud mínima)
                if new_query_string and len(new_query_string) > 3:
                    return new_query_string
                else:
                    print("WARN [Gemini]: Consulta sugerida vacía o demasiado corta tras limpieza.")
                    raise ValueError("Empty or too short suggestion")

            except Exception as e:
                print(f"ERROR [GeminiService:restructure_query] (Intento {retries + 1}): {e}")
                if "API key not valid" in str(e) or "API_KEY_INVALID" in str(e) or "permission" in str(e).lower():
                    print("ERROR FATAL [Gemini]: Problema con API Key o permisos. No se reintentará.")
                    return None
                if "Safety" in str(e) or "Blocked" in str(e):
                     print("WARN [Gemini]: Reestructuración bloqueada. Reintentando si es posible...")

                retries += 1
                if retries <= restructure_max_retries:
                    sleep_time = (2**(retries -1)) # Backoff un poco más largo para reestructuración
                    print(f"[*] [Gemini] Reintentando reestructuración en {sleep_time:.1f}s...")
                    time.sleep(sleep_time)
                else:
                    print("ERROR [GeminiService]: Máximo de reintentos alcanzado para reestructuración.")
                    return None # Fallo final

        return None # Devolver None si todos los reintentos fallan

