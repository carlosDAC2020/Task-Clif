import requests
import time
import json
import xml.etree.ElementTree as ET
from requests.exceptions import RequestException, Timeout, HTTPError

class PubMedService:
    """
    Gestiona las interacciones con la API E-utilities de NCBI PubMed
    para buscar artículos (ESearch) y obtener detalles (EFetch).
    """
    BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"

    def __init__(self, api_key=None, max_retries=3, pause_between_calls=3,
                 esearch_timeout=45, efetch_timeout=60, efetch_batch_size=100):
        """
        Inicializa el servicio PubMed.

        Args:
            api_key (str, optional): Tu API key de NCBI. Defaults to None.
            max_retries (int): Máximo de reintentos para llamadas API fallidas.
            pause_between_calls (float): Pausa mínima (segundos) entre llamadas API a NCBI.
            esearch_timeout (int): Timeout (segundos) para llamadas ESearch.
            efetch_timeout (int): Timeout (segundos) para llamadas EFetch (suelen tardar más).
            efetch_batch_size (int): Número de PMIDs a pedir en cada lote de EFetch.
        """
        self.api_key = api_key
        self.max_retries = max_retries
        self.pause_between_calls = pause_between_calls # NCBI requiere <10 reqs/sec sin API key, <3 con key
        self.esearch_timeout = esearch_timeout
        self.efetch_timeout = efetch_timeout
        self.efetch_batch_size = efetch_batch_size
        self.last_call_time = 0

    def _wait_if_needed(self):
        """Asegura que se respete la pausa entre llamadas."""
        now = time.time()
        elapsed = now - self.last_call_time
        if elapsed < self.pause_between_calls:
            wait_time = self.pause_between_calls - elapsed
            #print(f"[*] [PubMed] Waiting {wait_time:.2f}s before next API call...")
            time.sleep(wait_time)
        self.last_call_time = time.time() # Actualizar después de esperar (o si no se esperó)

    def _make_request(self, eutil, params=None, data=None, method='GET', timeout=None, is_json=True):
        """
        Método helper para realizar llamadas a la API de NCBI con reintentos y pausas.
        """
        if not timeout:
            timeout = self.esearch_timeout # Default a esearch timeout

        url = self.BASE_URL + eutil
        current_params = params.copy() if params else {}
        current_data = data.copy() if data else None

        # Añadir API Key si está disponible
        if self.api_key:
            if method == 'GET':
                current_params["api_key"] = self.api_key
            elif method == 'POST' and current_data:
                 current_data["api_key"] = self.api_key

        retries = 0
        while retries <= self.max_retries:
            self._wait_if_needed() # Esperar ANTES de hacer la llamada
            try:
                print(f"[*] [PubMed] Enviando petición {method} a {eutil}... (Intento {retries + 1})")
                if method == 'GET':
                    response = requests.get(url, params=current_params, timeout=timeout)
                elif method == 'POST':
                    response = requests.post(url, data=current_data, timeout=timeout)
                else:
                    raise ValueError(f"Método HTTP no soportado: {method}")

                response.raise_for_status() # Lanza HTTPError para respuestas 4xx/5xx

                print(f"[*] [PubMed] Respuesta recibida (Status: {response.status_code})")
                if is_json:
                    return response.json()
                else:
                    return response.content # Para EFetch XML

            except Timeout:
                print(f"WARN [PubMed]: Timeout ({timeout}s) alcanzado para {eutil} (Intento {retries + 1}).")
            except HTTPError as http_err:
                print(f"ERROR [PubMed]: Error HTTP {response.status_code} para {eutil}: {http_err} (Intento {retries + 1}).")
                # Podrías decidir no reintentar ciertos errores HTTP (e.g., 400 Bad Request)
                # if response.status_code == 400: break
            except RequestException as req_err:
                print(f"ERROR [PubMed]: Error de red/conexión para {eutil}: {req_err} (Intento {retries + 1}).")
            except json.JSONDecodeError as json_err:
                 if is_json:
                     print(f"ERROR [PubMed]: Error decodificando JSON de {eutil}: {json_err}. Contenido: {response.text[:200]}...")
                 else: # No debería pasar si pedimos XML, pero por si acaso
                     print(f"WARN [PubMed]: Error JSON inesperado pidiendo contenido no-JSON de {eutil}.")
                     return response.content # Devolver contenido crudo si no era JSON y falló el parseo
            except Exception as e:
                 print(f"ERROR [PubMed]: Error inesperado durante petición a {eutil}: {e} (Intento {retries + 1}).")

            # Si llegamos aquí, hubo un error y podemos reintentar
            retries += 1
            if retries <= self.max_retries:
                sleep_time = 1 * retries # Backoff lineal simple para PubMed
                print(f"[*] [PubMed] Reintentando {eutil} en {sleep_time}s...")
                time.sleep(sleep_time)
            else:
                print(f"ERROR [PubMed]: Máximo de reintentos ({self.max_retries}) alcanzado para {eutil}.")
                return None # Indicar fallo final

        return None # Fallo final tras reintentos

    def search_hybrid(self, terms_list, num_core_terms=2, max_results=500, search_field=None, sort_by="relevance"):
        """
        Realiza una búsqueda híbrida (AND para core terms, OR para el resto) en PubMed.

        Args:
            terms_list (list): Lista de keywords a buscar.
            num_core_terms (int): Número de términos iniciales a usar con AND.
            max_results (int): Máximo número de PMIDs a devolver.
            search_field (str, optional): Campo específico donde buscar (e.g., "Title/Abstract"). Defaults to None (busca en todos).
            sort_by (str): Criterio de ordenación (e.g., "relevance", "pub+date").

        Returns:
            dict: {'count': int, 'pmids': list} o {'count': 0, 'pmids': []} si falla.
        """
        default_result = {'count': 0, 'pmids': []}
        if not terms_list:
            print("WARN [PubMed:search_hybrid]: Lista de términos vacía.")
            return default_result

        # Filtrar términos inválidos (vacíos, no strings)
        filtered_terms = [str(t).strip() for t in terms_list if isinstance(t, (str, int)) and str(t).strip()]
        # Filtrar términos muy cortos después de limpiar
        filtered_terms = [t for t in filtered_terms if len(t) > 1]

        if not filtered_terms:
            print("WARN [PubMed:search_hybrid]: No quedaron términos válidos después del filtrado.")
            return default_result

        actual_core_terms = min(num_core_terms, len(filtered_terms))
        core_terms = filtered_terms[:actual_core_terms]
        rest_terms = filtered_terms[actual_core_terms:]

        # Función interna para escapar y entrecomillar términos
        def process_term(term):
             # Quitar comillas dobles existentes para evitar problemas
             clean_term = term.replace('"', '')
             # Escapar caracteres especiales si es necesario (opcional, PubMed suele manejarlos)
             # clean_term = clean_term.replace('[', r'\[').replace(']', r'\]') # Ejemplo
             if not clean_term: return None
             # Si el término contiene espacios, encerrarlo en comillas dobles
             return f'"{clean_term}"' if ' ' in clean_term else clean_term


        query_parts = []

        # Parte AND (Core)
        if core_terms:
            processed_core = [process_term(t) for t in core_terms]
            processed_core = [t for t in processed_core if t] # Eliminar None si process_term falló
            if processed_core:
                core_query = " AND ".join(processed_core)
                # Si solo hay un término core, no necesita paréntesis extra
                query_parts.append(f"({core_query})" if len(processed_core) > 1 else core_query)

        # Parte OR (Resto)
        if rest_terms:
            processed_rest = [process_term(t) for t in rest_terms]
            processed_rest = [t for t in processed_rest if t]
            if processed_rest:
                rest_query = " OR ".join(processed_rest)
                # Siempre necesita paréntesis si hay más de un término OR
                query_parts.append(f"({rest_query})" if len(processed_rest) > 1 else rest_query)

        if not query_parts:
            print("ERROR [PubMed:search_hybrid]: No se pudieron construir partes válidas de la consulta.")
            return default_result

        # Unir partes Core y Rest con AND
        final_query_structure = " AND ".join(query_parts)

        # Añadir campo de búsqueda si se especifica
        if search_field:
            search_term = f"({final_query_structure})[{search_field}]"
        else:
            search_term = final_query_structure

        print(f"[*] [PubMed] Construyendo consulta Híbrida: {search_term}")

        # --- Llamada a ESearch ---
        params = {
            "db": "pubmed",
            "term": search_term,
            "retmode": "json",
            "retmax": max_results,
            "sort": sort_by,
            "usehistory": "n" # No necesitamos historial para esto
        }

        data = self._make_request("esearch.fcgi", params=params, method='GET', timeout=self.esearch_timeout, is_json=True)

        if data is None: return default_result # Error en la petición

        esearch_result = data.get("esearchresult", {})

        # Revisar errores y advertencias de la API
        errors = esearch_result.get('errorlist', {})
        warnings = esearch_result.get('warninglist', {})
        if errors: print(f"WARN [PubMed API Errors]: {errors}")
        if warnings: print(f"WARN [PubMed API Warnings]: {warnings}")
        # Podrías querer manejar errores específicos aquí, como 'Query translation'

        pmid_list = esearch_result.get("idlist", [])
        count_str = esearch_result.get("count", "0")
        # Asegurarse de que count sea un entero
        try:
            count = int(count_str)
        except (ValueError, TypeError):
            print(f"WARN [PubMed]: No se pudo convertir 'count' ({count_str}) a entero. Asumiendo 0.")
            count = 0

        print(f"[*] [PubMed] Búsqueda Híbrida completada. PMIDs encontrados: {len(pmid_list)}. Total estimado: {count}")
        return {'count': count, 'pmids': pmid_list}

    def search_direct(self, query_string, max_results=500, sort_by="relevance"):
        """
        Realiza una búsqueda directa en PubMed usando la cadena de consulta proporcionada.

        Args:
            query_string (str): La consulta PubMed completa.
            max_results (int): Máximo número de PMIDs a devolver.
            sort_by (str): Criterio de ordenación.

        Returns:
            dict: {'count': int, 'pmids': list} o {'count': 0, 'pmids': []} si falla.
        """
        default_result = {'count': 0, 'pmids': []}
        if not query_string or not isinstance(query_string, str):
            print("WARN [PubMed:search_direct]: Cadena de consulta inválida o vacía.")
            return default_result

        print(f"[*] [PubMed] Ejecutando consulta Directa: {query_string}")

        params = {
            "db": "pubmed",
            "term": query_string,
            "retmode": "json",
            "retmax": max_results,
            "sort": sort_by,
            "usehistory": "n"
        }

        data = self._make_request("esearch.fcgi", params=params, method='GET', timeout=self.esearch_timeout, is_json=True)

        if data is None: return default_result

        esearch_result = data.get("esearchresult", {})
        errors = esearch_result.get('errorlist', {})
        warnings = esearch_result.get('warninglist', {})
        if errors: print(f"WARN [PubMed API Errors (Directa)]: {errors}")
        if warnings: print(f"WARN [PubMed API Warnings (Directa)]: {warnings}")

        pmid_list = esearch_result.get("idlist", [])
        count_str = esearch_result.get("count", "0")
        try:
            count = int(count_str)
        except (ValueError, TypeError):
             print(f"WARN [PubMed]: No se pudo convertir 'count' ({count_str}) a entero. Asumiendo 0.")
             count = 0

        print(f"[*] [PubMed] Búsqueda Directa completada. PMIDs: {len(pmid_list)}. Total: {count}")
        return {'count': count, 'pmids': pmid_list}

    def fetch_details(self, pmid_list):
        """
        Obtiene Título y Abstract para una lista de PMIDs usando EFetch.

        Args:
            pmid_list (list): Lista de PMIDs (pueden ser strings o ints).

        Returns:
            dict: Un diccionario mapeando PMID (str) a {'title': str, 'abstract': str}.
                  Devuelve {} si la lista está vacía o falla completamente.
                  Puede devolver un diccionario parcial si algunas llamadas fallan.
        """
        if not pmid_list: return {}

        # Asegurar que todos los PMIDs son strings y únicos
        pmids_to_fetch = list(set(str(p) for p in pmid_list if p))
        if not pmids_to_fetch: return {}

        print(f"[*] [PubMed] Iniciando EFetch (XML) para {len(pmids_to_fetch)} PMIDs en lotes de {self.efetch_batch_size}...")
        details_map = {}

        for i in range(0, len(pmids_to_fetch), self.efetch_batch_size):
            batch_pmids = pmids_to_fetch[i : i + self.efetch_batch_size]
            ids_str = ",".join(batch_pmids)
            print(f"  - Procesando lote EFetch ({i+1}-{min(i+self.efetch_batch_size, len(pmids_to_fetch))})...")

            # EFetch usa POST para listas largas de IDs
            data_payload = {
                "db": "pubmed",
                "id": ids_str,
                "retmode": "xml",
                "rettype": "abstract" # Pedir 'abstract' suele incluir título
            }

            # Usar _make_request para manejar la llamada, pausa, reintentos
            xml_content = self._make_request("efetch.fcgi", data=data_payload, method='POST',
                                             timeout=self.efetch_timeout, is_json=False)

            if xml_content is None:
                print(f"  - ERROR: Falló la petición EFetch para el lote {i+1}. Saltando este lote.")
                continue # Pasar al siguiente lote

            # Parsear el XML
            try:
                # Asegurarse de que el contenido no esté vacío antes de parsear
                if not xml_content.strip():
                     print(f"  - WARN: Respuesta EFetch vacía para el lote {i+1}.")
                     continue

                root = ET.fromstring(xml_content)
                articles_found_in_batch = 0
                for article_node in root.findall('.//PubmedArticle'):
                    pmid_node = article_node.find('.//PMID')
                    pmid = pmid_node.text.strip() if pmid_node is not None and pmid_node.text else None
                    if not pmid: continue

                    # Extraer Título (manejo robusto por si falta)
                    title_node = article_node.find('.//ArticleTitle')
                    # Usar itertext() para manejar etiquetas internas como <i>, <b>
                    title = "".join(title_node.itertext()).strip() if title_node is not None else "N/A"

                    # Extraer Abstract (manejo robusto y de secciones)
                    abstract_text = "N/A"
                    abstract_node = article_node.find('.//Abstract')
                    if abstract_node is not None:
                        abstract_parts = []
                        # Buscar nodos AbstractText dentro de Abstract
                        for text_node in abstract_node.findall('.//AbstractText'):
                            # Obtener etiqueta (e.g., BACKGROUND, METHODS) si existe
                            label = text_node.get('Label')
                            text_content = "".join(text_node.itertext()).strip()
                            if label and text_content:
                                abstract_parts.append(f"{label.upper()}: {text_content}")
                            elif text_content:
                                abstract_parts.append(text_content)
                        # Unir las partes si se encontraron
                        if abstract_parts:
                            abstract_text = " ".join(abstract_parts)

                    details_map[pmid] = {"title": title, "abstract": abstract_text}
                    articles_found_in_batch += 1

                print(f"  - Lote EFetch completado. Detalles extraídos para {articles_found_in_batch} artículos.")

            except ET.ParseError as xml_err:
                print(f"  - ERROR: Falló el parseo XML para el lote {i+1}: {xml_err}")
                # Podrías guardar el xml_content en un archivo para depurar si esto ocurre
                # with open(f"efetch_error_batch_{i+1}.xml", "wb") as f_err:
                #     f_err.write(xml_content)
                continue # Pasar al siguiente lote
            except Exception as e:
                 print(f"  - ERROR: Error inesperado procesando XML del lote {i+1}: {e}")
                 continue

        print(f"[*] [PubMed] EFetch completado. Total detalles recuperados: {len(details_map)} PMIDs.")
        return details_map
