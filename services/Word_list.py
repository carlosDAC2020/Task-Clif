import pandas as pd
import spacy
import re
from collections import Counter
import os
import csv
import json

# csv import is not strictly needed if pandas.read_csv is used robustly.

nlp = spacy.load("en_core_web_lg")


def lemantizar(text):
    doc = nlp(text)
    spacy_lemmatized = [token.lemma_ for token in doc]
    spacy_lemmatized = " ".join(spacy_lemmatized)
    return spacy_lemmatized


def create_word_frequency_df(text):
    """
    Create a word frequency DataFrame from text.

    Args:
        text (str): Input text to analyze

    Returns:
        pandas.DataFrame: DataFrame with word frequencies
    """
    # Lemmatize the text
    text = lemantizar(text)

    # Handle text that might be a list
    if isinstance(text, list):
        text = " ".join(text)

    # Extract words and count frequencies
    words = re.findall(r"\b\w+\b", text.lower())
    word_counts = Counter(words)

    # Convert to DataFrame
    df = pd.DataFrame.from_dict(word_counts, orient="index", columns=["Count"])
    df = df.reset_index().rename(columns={"index": "Words"})
    df = df.sort_values("Count", ascending=False)
    df = df.reset_index(drop=True)
    return df


class WordListExtractor:
    """
    Extracts a list of domain-specific words from text based on "weirdness" scores.
    The weirdness score compares word frequency in the input text to a general corpus (Google Unigrams).
    Words with lower weirdness scores are considered more domain-specific.
    """

    def __init__(self, path="Eng_GoogleUnigrams.csv", general_words_json=None):
        """
        Initialize the WordListExtractor.

        Args:
            path (str): Path to the Google Unigrams CSV file.
            general_words_json (str, optional): Path to the general words representation JSON file.
        """
        self.n_gram_google = None
        self.path_ngram = path
        self.general_words_data = None

        # Load Google Unigrams
        try:
            # Open and read the Google Unigrams file
            file = open(path, "r", encoding="utf-8")
            n_gram_data = list(csv.reader(file, delimiter=";"))
            file.close()

            # Convert to DataFrame
            self.n_gram_google = pd.DataFrame(n_gram_data)

            # Convert frequency values to float
            if len(self.n_gram_google.columns) > 1:
                self.n_gram_google[1] = pd.to_numeric(
                    self.n_gram_google[1], errors="coerce"
                )
                self.n_gram_google = self.n_gram_google.dropna(subset=[1])

            print(f"Loaded {len(self.n_gram_google)} unigrams from {path}")
        except FileNotFoundError:
            print(f"File '{path}' not found. Some functionality will be limited.")
        except Exception as e:
            print(f"Error loading unigrams file: {e}")

        # Load general words JSON if provided
        if general_words_json:
            self.load_general_words(general_words_json)

    def load_general_words(self, json_path):
        """
        Load general words representation from a JSON file.

        Args:
            json_path (str): Path to the JSON file with general words representation

        Returns:
            bool: True if loading was successful, False otherwise
        """
        try:
            with open(json_path, "r", encoding="utf-8") as file:
                self.general_words_data = json.load(file)
            print(f"Loaded general words data from {json_path}")
            return True
        except FileNotFoundError:
            print(
                f"File '{json_path}' not found. General words functionality will be limited."
            )
            return False
        except Exception as e:
            print(f"Error loading general words file: {e}")
            return False

    def top_words_from_keys(self, keys, n_top=3, exclude_words=None):
        """
        Get the top words associated with given keys from the general_words_data.

        Args:
            keys (list): List of key words to look up
            n_top (int): Number of top words to return
            exclude_words (list, optional): Words to exclude from results

        Returns:
            list: Top words associated with the keys plus the original keys
        """
        if not self.general_words_data:
            print(
                "Warning: No general words data loaded. Use load_general_words() first."
            )
            return keys

        total_counter = Counter()

        for key in keys:
            if key in self.general_words_data:
                inner_dict = self.general_words_data[key]
                # Exclude specific words if requested
                if exclude_words:
                    inner_dict = {
                        k: v for k, v in inner_dict.items() if k not in exclude_words
                    }
                total_counter.update(inner_dict)
            else:
                print(f"Warning: Key '{key}' not found in the general words data.")

        # Get the top N words and add the original keys
        top_words = [word for word, _ in total_counter.most_common(n_top)]
        top_words.extend(keys)
        return top_words

    def Weirdness(self, lexicon_cat):
        """
        Calculate weirdness score for words by comparing domain-specific frequency
        with general corpus (Google Unigrams) frequency.

        Args:
            lexicon_cat (pandas.DataFrame): Domain-specific word frequency data with columns ["Words", "Count"]

        Returns:
            pandas.DataFrame: DataFrame with words and their weirdness scores
        """
        if self.n_gram_google is None or len(self.n_gram_google) == 0:
            print(
                "Warning: Google Unigrams data not available. Cannot calculate weirdness scores."
            )
            return lexicon_cat

        # Identificar palabras que están en el lexicón de entrada pero no en Google Unigrams
        # Estas son potencialmente las más específicas del dominio
        unique_words_mask = ~lexicon_cat["Words"].isin(self.n_gram_google[0])
        unique_words = lexicon_cat[unique_words_mask].copy()  # Crear copia explícita

        # Find words that exist in both domain corpus and Google corpus
        New_Google = self.n_gram_google[
            self.n_gram_google[0].isin(lexicon_cat["Words"])
        ]
        New_Lexicon = lexicon_cat[lexicon_cat["Words"].isin(New_Google[0])]

        # Si no hay palabras comunes, devolver solo las palabras únicas
        # con un valor de Weirdness artificialmente bajo (0.1) para indicar alta especificidad
        if len(New_Google) == 0 or len(New_Lexicon) == 0:
            print(
                "Warning: No common words found between input text and reference corpus."
            )
            if len(unique_words) > 0:
                unique_words.loc[:, "Weirdness"] = (
                    0.1  # Usar .loc para evitar SettingWithCopyWarning
                )
                return unique_words
            else:
                return pd.DataFrame(columns=["Words", "Count", "Weirdness"])

        # Sort and convert to correct data types
        New_Lexicon = New_Lexicon.sort_values(by=["Words"])
        New_Lexicon["Count"] = New_Lexicon["Count"].astype(float)
        New_Google = New_Google.sort_values(by=[0])
        New_Google[1] = New_Google[1].astype(float)

        # Reset indices for alignment
        New_Google = New_Google.reset_index(drop=True)
        New_Lexicon = New_Lexicon.reset_index(drop=True)

        # Calculate weirdness score
        All_Words = len(self.n_gram_google)
        New_New_Lexicon = New_Lexicon.copy()
        New_New_Lexicon["Count"] = New_New_Lexicon["Count"].mul(All_Words)
        New_New_Lexicon["Weirdness"] = (
            (New_Google[1]) / New_New_Lexicon["Count"]
        ) * 1000
        New_New_Lexicon = New_New_Lexicon.sort_values(by=["Weirdness"], ascending=True)

        # Si hay palabras únicas, les asignamos un valor de Weirdness artificialmente bajo
        # para indicar que son altamente específicas del dominio
        if len(unique_words) > 0:
            unique_words.loc[:, "Weirdness"] = 0.1  # Valor artificialmente bajo
            # Concatenate unique words with common words
            New_New_Lexicon = pd.concat(
                [unique_words, New_New_Lexicon], ignore_index=True
            )
            New_New_Lexicon = New_New_Lexicon.sort_values(
                by=["Weirdness"], ascending=True
            )
        print(f"New_New_Lexicon {New_New_Lexicon}")
        return New_New_Lexicon

    def extract_word_list_from_sentence(self, sentence, weirdness_threshold=1.0):
        """
        Extract domain-specific words from a sentence based on weirdness scores.

        Args:
            sentence (str): Input sentence to analyze
            weirdness_threshold (float): Maximum weirdness score for words to be considered domain-specific

        Returns:
            list: List of domain-specific words from the sentence
        """
        # Get word frequency dataframe for the sentence
        word_frequency_df = create_word_frequency_df(sentence)
        # print(f"word_frequency_df {word_frequency_df}")
        # Calculate weirdness scores
        weirdness_df = self.Weirdness(word_frequency_df)
        # print(f"weirdness_df {weirdness_df}")
        # Filter words with weirdness below threshold
        if "Weirdness" in weirdness_df.columns:
            domain_words = weirdness_df[
                weirdness_df["Weirdness"] < weirdness_threshold
            ]["Words"].tolist()
        else:
            # If weirdness couldn't be calculated, return top frequency words
            domain_words = word_frequency_df.head(3)["Words"].tolist()

        return domain_words


# Ejemplo de ejecución con datos BioASQ
if __name__ == "__main__":
    import json

    # Cargar datos de ejemplo del archivo BioASQ
    try:
        with open("BioASQ-task13bPhaseA-testset4.json", "r", encoding="utf-8") as file:
            bioasq_data = json.load(file)

        # Extraer las preguntas del archivo JSON
        questions = bioasq_data.get("questions", [])

        # Limitar a 5 preguntas para el ejemplo
        sample_questions = questions[:30]

        print(f"Procesando {len(sample_questions)} preguntas de BioASQ...\n")

        # Inicializar el extractor de palabras
        word_extractor = WordListExtractor()

        # Cargar los datos de palabras generales
        word_extractor.load_general_words("general_words_representation.json")

        # Procesar cada pregunta y mostrar los resultados
        for i, question in enumerate(sample_questions, 1):
            question_text = question.get("body", "")
            question_type = question.get("type", "unknown")

            print(f"Pregunta {i}: [{question_type}] {question_text}")

            # Extraer palabras específicas de dominio
            domain_words = word_extractor.extract_word_list_from_sentence(
                question_text, 10
            )

            print(f"Palabras específicas del dominio: {domain_words}")

            # Si tenemos palabras de dominio, úsalas como claves para encontrar palabras relacionadas
            if domain_words:
                # Obtener palabras importantes relacionadas con las palabras de dominio
                related_words = word_extractor.top_words_from_keys(
                    keys=domain_words,
                    n_top=5,
                    exclude_words=["the", "is", "are", "in", "of", "and", "to", "a"],
                )
                print(
                    f"Palabras relacionadas de la representación general: {related_words}"
                )

            print("-" * 80)

            # Limitar a solo unas pocas preguntas para la demostración
            if i >= 3:
                break

    except FileNotFoundError:
        print(
            "Error: No se pudo encontrar el archivo BioASQ-task13bPhaseA-testset4.json o general_words_representation.json"
        )
    except json.JSONDecodeError:
        print("Error: Uno de los archivos JSON no contiene JSON válido")
    except Exception as e:
        print(f"Error al procesar los datos: {e}")
