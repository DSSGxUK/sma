from cucco import Cucco
import spacy
from nltk.stem.snowball import SnowballStemmer

NORMALIZER = Cucco()
LEMMATIZER = spacy.load("es_core_news_sm", disable=["parser", "ner"])
STEMMER = SnowballStemmer(language="spanish")

def normalize_text(text, lang="es"):
    """Normalize text by applying various transformations

    Args:
        text (str): A text document to normalize
        lang (str, optional): Specifies the language used for stop word removal. Defaults to "es".

    Returns:
        str: A normalized text string in lower case
    """
    normalization_steps = [
        "replace_punctuation",
        "remove_accent_marks",
        "remove_extra_white_spaces",
        "replace_urls",
        ("remove_stop_words", {"language": lang})
    ]
    text = NORMALIZER.normalize(text, normalizations=normalization_steps)
    return text.lower()

def lemmatize_and_stem_text(raw_document):
    """Lemmatize and stem a document

    Args:
        raw_document (str): A raw document to be lemmatized and stemmed

    Returns:
        str: A lemmatized and stemmed document
    """
    tokens = LEMMATIZER(text=raw_document)
    document_lemma = [token.lemma_ for token in tokens]
    document_stem = [STEMMER.stem(lemma_token) for lemma_token in document_lemma]
    text = " ".join(document_stem)
    return text