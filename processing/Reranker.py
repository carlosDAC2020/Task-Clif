from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import numpy as np
import re


class Reranker:
    """
    Clase para reordenar documentos basándose en la similitud con una consulta.
    Incluye métodos para TF-IDF y BM25.
    """

    def __init__(self, stop_words="english", min_df=1, max_features=5000):
        """
        Inicializa el Reranker con un TfidfVectorizer.

        Args:
            stop_words (str or list, optional): Palabras vacías para el vectorizador. Por defecto, 'english'.
            min_df (int, optional): Frecuencia mínima de documento para el vectorizador. Por defecto, 1.
            max_features (int, optional): Número máximo de características para el vectorizador. Por defecto, 5000.
        """
        self.vectorizer = TfidfVectorizer(
            stop_words=stop_words, min_df=min_df, max_features=max_features
        )
        print(f"Reranker inicializado con TF-IDF (max_features={max_features}).")

        self.pubmedbert_model = SentenceTransformer(
            "NeuML/pubmedbert-base-embeddings"
        )
        print("Reranker inicializado con PubMedBERT embeddings.")

    def _preprocess_text(self, text):
        """
        Preprocesa el texto: convierte a minúsculas y elimina caracteres no alfanuméricos.

        Args:
            text (str): Texto a preprocesar.

        Returns:
            list: Lista de tokens procesados.
        """
        # Convertir a minúsculas
        text = text.lower()
        # Eliminar caracteres no alfanuméricos
        text = re.sub(r"[^a-z0-9\s]", "", text)
        # Dividir en tokens
        tokens = text.split()
        return tokens

    def rerank_TF_IDF(self, original_question, pmid_details_map):
        """
        Reordena los documentos recuperados basándose en la similitud TF-IDF con la pregunta original.

        Args:
            original_question (str): La pregunta original del usuario.
            pmid_details_map (dict): Un diccionario {pmid: {"title": "...", "abstract": "..."}}.

        Returns:
            tuple: (list, dict)
                - reranked_pmids (list): Lista de PMIDs ordenada por relevancia.
                - pmid_scores (dict): Diccionario {pmid: score} con las puntuaciones de similitud.
        """
        if not original_question or not pmid_details_map:
            print(
                "Reranker.rerank: Entrada inválida (pregunta vacía o mapa de detalles vacío)."
            )
            return [], {}

        valid_pmids = []
        documents_combined_text = []

        # Preparar textos: combinar título y resumen para cada PMID
        for pmid, details in pmid_details_map.items():
            if details:
                title = details.get("title", "")
                abstract = details.get("abstract", "")
                # Combinar texto, asegurándose de manejar 'N/A' o None
                text_parts = [t for t in [title, abstract] if t and t != "N/A"]
                combined_text = " ".join(text_parts).strip()

                if combined_text:
                    valid_pmids.append(pmid)
                    documents_combined_text.append(combined_text)

        if not valid_pmids:
            print(
                "[!] Reranker: No hay textos válidos (título/resumen) para reordenar."
            )
            return list(pmid_details_map.keys()), {}

        print(
            f"[*] Reranker: Iniciando reordenamiento TF-IDF para {len(valid_pmids)} PMIDs..."
        )

        try:
            # Incluir la pregunta original al principio de la lista de textos
            all_texts = [original_question] + documents_combined_text

            # Verificar que haya textos no vacíos para evitar error en fit_transform
            if not any(all_texts):
                print(
                    "[!] Reranker: Todos los textos (pregunta + documentos) están vacíos. Saltando TF-IDF."
                )
                return valid_pmids, {}

            # Calcular la matriz TF-IDF
            tfidf_matrix = self.vectorizer.fit_transform(all_texts)

            # Separar el vector de la pregunta y los vectores de los documentos
            question_vector = tfidf_matrix[0]
            document_vectors = tfidf_matrix[1:]

            # Asegurar que hay vectores de documentos para comparar
            if document_vectors.shape[0] == 0:
                print(
                    "[!] Reranker: No hay vectores de documentos válidos para comparar. Saltando cálculo de similitud."
                )
                return valid_pmids, {}

            # Calcular similitudes coseno
            cosine_similarities = cosine_similarity(
                question_vector, document_vectors
            ).flatten()

            # Crear pares (score, pmid) y mapeo de scores
            scored_pmids = list(zip(cosine_similarities, valid_pmids))
            pmid_scores = {pmid: float(score) for score, pmid in scored_pmids}

            # Ordenar los PMIDs por score descendente
            scored_pmids.sort(key=lambda x: x[0], reverse=True)
            reranked_pmids = [pmid for score, pmid in scored_pmids]

            print(f"[*] Reranker: Reordenamiento completado.")
            return reranked_pmids, pmid_scores

        except ValueError as ve:
            print(
                f"[!] Reranker: Error de Valor durante TF-IDF (posiblemente vocabulario vacío): {ve}"
            )
            return valid_pmids, {}
        except Exception as e:
            print(f"[!] Reranker: Error inesperado durante reordenamiento: {e}")
            return valid_pmids, {}

    def rerank_bm25(self, original_question, pmid_details_map):
        """
        Reordena los documentos recuperados basándose en la similitud BM25 con la pregunta original.

        Args:
            original_question (str): La pregunta original del usuario.
            pmid_details_map (dict): Un diccionario {pmid: {"title": "...", "abstract": "..."}}.

        Returns:
            tuple: (list, dict)
                - reranked_pmids (list): Lista de PMIDs ordenada por relevancia.
                - pmid_scores (dict): Diccionario {pmid: score} con las puntuaciones BM25.
        """
        if not original_question or not pmid_details_map:
            print(
                "Reranker.rerank_bm25: Entrada inválida (pregunta vacía o mapa de detalles vacío)."
            )
            return [], {}

        valid_pmids = []
        documents_combined_text = []

        # Preparar textos: combinar título y resumen para cada PMID
        for pmid, details in pmid_details_map.items():
            if details:
                title = details.get("title", "")
                abstract = details.get("abstract", "")
                # Combinar texto, asegurándose de manejar 'N/A' o None
                text_parts = [t for t in [title, abstract] if t and t != "N/A"]
                combined_text = " ".join(text_parts).strip()

                if combined_text:
                    valid_pmids.append(pmid)
                    documents_combined_text.append(combined_text)

        if not valid_pmids:
            print(
                "[!] Reranker: No hay textos válidos (título/resumen) para reordenar."
            )
            return list(pmid_details_map.keys()), {}

        print(
            f"[*] Reranker: Iniciando reordenamiento BM25 para {len(valid_pmids)} PMIDs..."
        )

        try:
            # Preprocesar y tokenizar los documentos
            tokenized_corpus = [
                self._preprocess_text(doc) for doc in documents_combined_text
            ]

            # Inicializar el modelo BM25
            bm25 = BM25Okapi(tokenized_corpus)

            # Preprocesar y tokenizar la pregunta
            tokenized_query = self._preprocess_text(original_question)

            # Calcular las puntuaciones BM25
            scores = bm25.get_scores(tokenized_query)

            # Crear pares (score, pmid) y mapeo de scores
            scored_pmids = list(zip(scores, valid_pmids))
            pmid_scores = {pmid: float(score) for score, pmid in scored_pmids}

            # Ordenar los PMIDs por score descendente
            scored_pmids.sort(key=lambda x: x[0], reverse=True)
            reranked_pmids = [pmid for score, pmid in scored_pmids]

            print(f"[*] Reranker: Reordenamiento BM25 completado.")
            return reranked_pmids, pmid_scores

        except Exception as e:
            print(f"[!] Reranker: Error inesperado durante reordenamiento BM25: {e}")
            return valid_pmids, {}

    def rerank_pubmedbert(self, original_question, pmid_details_map):
        """
        Reordena los documentos basándose en la similitud semántica utilizando incrustaciones de PubMedBERT.

        Args:
            original_question (str): La pregunta original del usuario.
            pmid_details_map (dict): Un diccionario {pmid: {"title": "...", "abstract": "..."}}.

        Returns:
            tuple: (list, dict)
                - reranked_pmids (list): Lista de PMIDs ordenada por relevancia.
                - pmid_scores (dict): Diccionario {pmid: score} con las puntuaciones de similitud.
        """
        if not original_question or not pmid_details_map:
            print("Entrada inválida: pregunta vacía o mapa de detalles vacío.")
            return [], {}

        valid_pmids = []
        documents_combined_text = []

        for pmid, details in pmid_details_map.items():
            if details:
                title = details.get("title", "")
                abstract = details.get("abstract", "")
                text_parts = [t for t in [title, abstract] if t and t != "N/A"]
                combined_text = " ".join(text_parts).strip()
                if combined_text:
                    valid_pmids.append(pmid)
                    documents_combined_text.append(combined_text)

        if not valid_pmids:
            print("No hay documentos válidos para reordenar.")
            return list(pmid_details_map.keys()), {}

        # Codificar la pregunta y los documentos
        print(" Generando embedings")
        question_embedding = self.pubmedbert_model.encode([original_question])[0]
        document_embeddings = self.pubmedbert_model.encode(documents_combined_text)

        # Calcular similitudes coseno
        from sklearn.metrics.pairwise import cosine_similarity

        similarities = cosine_similarity(
            [question_embedding], document_embeddings
        ).flatten()

        # Crear pares (score, pmid) y mapeo de scores
        scored_pmids = list(zip(similarities, valid_pmids))
        pmid_scores = {pmid: float(score) for score, pmid in scored_pmids}

        # Ordenar los PMIDs por score descendente
        scored_pmids.sort(key=lambda x: x[0], reverse=True)
        reranked_pmids = [pmid for score, pmid in scored_pmids]

        return reranked_pmids, pmid_scores
