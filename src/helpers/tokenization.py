import spacy
# The following two lines should be uncommented once "es_core_web_sm" is downloaded
# sp = spacy.load("es_core_web_sm")
# stopwords = sp.Defaults.stop_words

def remove_stopwords(text: str):
    """ Remove stop words from a string
    :param text: the text with the stop words to be removed
    :type text: str
    :return: The text with no stop words, and the tokens
    :rtype: dict
    """
    tokens = [token for token in text.split() if token.lower() not in stopwords]
    text = " ".join(tokens)
    return { "text": text, "tokens": tokens }
