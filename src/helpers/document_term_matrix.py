import pandas as pd
import numpy as np

# For the NLP steps
import re
import string
import spacy
import es_core_news_sm
from spacy.lang.es.stop_words import STOP_WORDS
from scipy.sparse import coo_matrix, vstack

# To create the document-term matrix
from sklearn.feature_extraction.text import CountVectorizer

# Our text cleaning helper function
from text_cleaning import count_vectorization, clean_text

def get_count_vetorization_matrix(text):
    return count_vectorization(text=text)

def document_term_matrices(data: pd.DataFrame, text_column: str, group_data_col: str = None, output_dataframe: bool = False):
    """
    Generates the document-term matrix for each category of the variable 'variable'.

    :param data: The dataframe containing the text column for which we want to generate the document-term matrices
    :type data: pd.DataFrame
    :param group_data_col: The variable of the dataframe to group by
    :type group_data_col: string
    :param text_column: The name of the column to perform stemming on
    :type text_column: string
    :return: A dictionary of dataframes, where each key is the name of a category of the variable of interest, and the corresponding value is the document-term matrix for that category
    :rtype: dict
    """

    # Clean the text
    cleaned_data = clean_text(data, text_column)

    # Collapse the details into a single row per complaint
    complaint_detail = cleaned_data[['ComplaintId', group_data_col, 'cleaned_text']]
    grouped = complaint_detail.groupby('ComplaintId').transform(lambda x: ', '.join(x))
    grouped = pd.concat([complaint_detail[['ComplaintId', group_data_col]], grouped], axis=1)
    grouped = grouped.drop_duplicates()
    
    # Create a dictionary to store the document-term matrices for each category of the specified variable
    doc_term_matrices = {}
    # For each category, create the document-term matrix
    for category in grouped[group_data_col].fillna('NaN').unique():
        # Restrict to the rows corresponding to that category
        if category=='NaN':
            subset = grouped[grouped[group_data_col].isna()]
        else:
            subset = grouped[grouped[group_data_col]==category]

        # Create the term-document matrix
        vectorizer = CountVectorizer()
        tf_idf_features = vectorizer.fit_transform(subset['cleaned_text'].tolist())

    if(output_dataframe):
        doc_term_matrix = pd.DataFrame(tf_idf_features.toarray(), columns=vectorizer.get_feature_names())
        doc_term_matrices[category] = doc_term_matrix
        return doc_term_matrices
    return tf_idf_features

# @TODO: add param to export N terms in tf-idf