from typing import Iterable
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from helpers.helper_function import (normalize_text, lemmatize_and_stem_text)

class DenseTFIDFVectorization(TfidfVectorizer):
    def __init__(self, ngram_range=(1, 1), max_features=100):
        """Create a dense representation of the tf-idf features

        Args:
            ngram_range (tuple, optional): Specifies the ngrams to use during the tf-idf fearure transformation. Defaults to (1, 1).
            max_features (int, optional): The maximum number of tf-idf terms to return. Defaults to 100.

        Example:
            >>> df = pd.DataFrame({"text": ["A document in the dataset", "And this is another document for example"]})
            >>> raw_text = df["text"]
            >>> dense_tfidf = DenseTFIDFVectorization(ngram_range=(1, 1), max_features=20)
            >>> tfidf_features = dense_tfidf.fit_transform(raw_documents=raw_text)
        """
        TfidfVectorizer.__init__(self)
        self.ngram_range = ngram_range
        self.max_features = max_features
    def fit(self, raw_documents: Iterable, y=None):
        """Learn vocabulary and idf from training set

        Args:
            raw_documents (Iterable): An iterable which yields strings
            y (None, optional): This parammeter is not needed to calculate the tfidf. Defaults to None.

        Returns:
            self: The fitted vectorizer
        """
        # ### Added in by Henri to get the required stopword removal & stemming behaviour
        # assert isinstance(raw_documents, pd.Series) == True | isinstance(raw_documents, list) == True
        # if(isinstance(raw_documents, list)):
        #     print("Transforming data")
        #     raw_documents = pd.Series(raw_documents)

        # print("Transforming data")
        # raw_documents = raw_documents.apply(normalize_text).apply(lemmatize_and_stem_text)
        # ### END of Henri's added code

        return super().fit(raw_documents=raw_documents, y=y)
    
    def transform(self, raw_documents: Iterable):
        """Transform terms to document term matrix

        Args:
            raw_documents (Iterable): An iterable that yields strings

        Returns:
            pd.DataFrame: A dataframe containing tf-idf features and weights
        """
        assert isinstance(raw_documents, pd.Series) == True | isinstance(raw_documents, list) == True
        if(isinstance(raw_documents, list)):
            print("Transforming data")
            raw_documents = pd.Series(raw_documents)

        print("Transforming data")
        raw_documents = raw_documents.apply(normalize_text).apply(lemmatize_and_stem_text)
        
        X = super().transform(raw_documents=raw_documents)
        df = pd.DataFrame(X.toarray(), columns=self.get_feature_names())
        return df

    def fit_transform(self, raw_documents, y=None):
        """Learn the vocabulary and transform the terms to tfidf. This is the same as calling fit followed by transform

        Args:
            raw_documents (Iterable): An Iterable that yields strings
            y (None, optional): This parameter is not required for tfidf. Defaults to None.

        Returns:
            pd.DataFrame: Tfidf feature names and weights
        """
        return self.fit(raw_documents=raw_documents).transform(raw_documents=raw_documents)
