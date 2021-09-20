import time
import numpy as np
import pandas as pd
import re
import string
import spacy
import es_core_news_sm

# Import Snowball stemmer
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer
from unidecode import unidecode

def character_replacement(text: str):
    if(pd.isna(text)): return np.nan
    text = str(text)
    u = unidecode(text, "utf-8")
    return unidecode(u)

def normalize_ascii(dataframe: pd.DataFrame, columns: list = [], all_cols = False):
    if(all_cols):
        return dataframe.apply(character_replacement)
    
    for column in columns:
        dataframe[column] = dataframe[column].astype(str).apply(character_replacement)
    return dataframe.replace('nan', np.nan)

def count_vectorizer_preprocessor(data_row):
    # make text lower case and remove punctuation
    data_row = data_row.lower()
    data_row = "".join(c for c in data_row if c not in set(string.punctuation))
    return data_row

def make_stop_word_list():
    spacy_model = es_core_news_sm.load()
    stop_words = spacy_model.Defaults.stop_words
    # Add important stopwords which are not in the Spacy list
    additional_stopwords = ['y', 'a', 'por']
    stop_words.update(additional_stopwords)
    stop_words = [character_replacement(w) for w in stop_words]
    return list(stop_words)
    # return " ".join(w.lower() for w in word_tokenize(text) if w not in stop_words)

def count_vectorization(text):
    stop_words = make_stop_word_list()
    count_vectorizer = CountVectorizer(stop_words=stop_words, preprocessor=count_vectorizer_preprocessor)
    count_vecotrize_features = count_vectorizer.fit_transform(text["ComplaintDetail"])
    count_vectorizer.fit(text)
    return count_vecotrize_features
    
    # if (remove_punct):
    #     text = remove_punctuation(text)
    # text = pd.Series(text).apply(remove_stopwords, discard_accents=discard_accents)
    # return text

def clean_text(data: pd.DataFrame, text_column: str, discard_accents: bool =True):
    """
    Adds a column to the 'data' dataframe to store the processed text of the column specified in the argument 'text_column'. Processing involves tokenization, as well as removal of punctuation and stop words.

    :param data: The dataframe containing the text column we want to clean
    :type data: pd.DataFrame
    :param text_column: The name of the column to perform text cleaning on
    :type text_column: string
    :param discard_accents: Specifies whether the text column contains accents or not
    :type discard_accents: boolean
    :return: The input dataframe, with an extra column storing the cleaned text column
    :rtype: pd.DataFrame
    """
    
    # Change text column to string type
    data[text_column] = data[text_column].astype(str)
        
    # Remove punctuation in each text row
    table = str.maketrans(dict.fromkeys(string.punctuation))
    token_list = []
    for tokens in data[text_column]:
        this_token = tokens.translate(table) # Output: string without punctuation
        token_list.append(this_token)
    
    
    # Make all letters lowercase
    data['cleaned_text'] = data[text_column].apply(lambda w: w.lower())

    # Load the SpaCy model
    spacy_model = es_core_news_sm.load()

    # Get the list of Spanish stopwords from SpaCy
    stop_words = spacy_model.Defaults.stop_words
    # Add important stopwords which are not in the Spacy list
    additional_stopwords = ['y', 'a', 'por']
    stop_words.update(additional_stopwords)
    
    if discard_accents:
        # Remove accents from our stopword list, for consistency with our cleaned data
        stop_words = [character_replacement(w) for w in stop_words]
    
    # Stopword removal function
    def stopword_remove(s: str):
        return " ".join(w.lower() for w in s.split() if w not in stop_words)
    
    # Apply stopword removal
    data['cleaned_text'] = data['cleaned_text'].apply(stopword_remove)
    return data
    
def lemmatize(data: pd.DataFrame, text_column: str, discard_accents: bool =True):
    """
    Applies the SpaCy lemmatizer to the text from the column specified in 'text_column'.

    :param data: The dataframe containing the text column we want to lemmatize
    :type data: pd.DataFrame
    :param text_column: The name of the column to perform lemmatization on
    :type text_column: string
    :param discard_accents: Specifies whether the text column contains accents or not
    :type discard_accents: boolean
    :return: The input dataframe, with the words in the specified text column lemmatized
    :rtype: pd.DataFrame
    """
    # Load the SpaCy model
    spacy_model = es_core_news_sm.load()
    
    # The SpaCy lemmatizer
    def lemmatization(s: str):
        doc = spacy_model(s)
        return " ".join([token.lemma_ for token in doc])
        
    # Apply the lemmatizer
    data['cleaned_text'] = data['cleaned_text'].apply(lemmatization)
    
    if discard_accents:
        # We need to remove any accents that the lemmatizer has added in
        data = normalize_ascii(data, columns=['cleaned_text'])

    ## Finally, we need to remove any stopwords introduced by the lemmatizer
    # Get the list of Spanish stopwords from Spacy
    stop_words = spacy_model.Defaults.stop_words
    # Add important stopwords which are not in the Spacy list
    additional_stopwords = ['y', 'a', 'por']
    [stop_words.add(word) for word in additional_stopwords]
    
    if discard_accents:
        # Remove accents from our stopword list, for consistency with our cleaned data
        stop_words = [character_replacement(w) for w in stop_words]
    
    # Remove the stopwords
    def stopword_remove(s: str):
        return " ".join(w.lower() for w in s.split() if w not in stop_words)
    
    data['cleaned_text'] = data['cleaned_text'].apply(stopword_remove)
    return data

def stemmer(data: pd.DataFrame, text_column: str):
    """
    Applies the Snowball stemmer to the text from the column specified in 'text_column'.

    :param data: The dataframe containing the text column we want to stem
    :type data: pd.DataFrame
    :param text_column: The name of the column to perform stemming on
    :type text_column: string
    :return: The input dataframe, with the words in the specified text column stemmed
    :rtype: pd.DataFrame
    """

    # Set the language
    snowball = SnowballStemmer(language='spanish')

    def stem(sentence: str):
        """
        Takes a string as input and stems every word in that string. Output is a string of the stemmed words.
        """
        if pd.isna(sentence):
            return sentence
        else:
            # Apply the stemmer to each word in the input string
            stemmed_words = [snowball.stem(word) for word in sentence.split()]
        # Combine the words back into a single string
        stemmed_sentence = " ".join(stemmed_words)
        return stemmed_sentence

    # Apply the stemming function to the text column of interest and save it as a new column
    data['stemmed'] = data[text_column].apply(stem)

    return data
