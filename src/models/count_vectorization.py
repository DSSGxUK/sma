import os
import sys
import warnings
import string
import es_core_news_sm
import pandas as pd

from nltk.stem.snowball import SnowballStemmer
from sklearn.base import BaseEstimator, TransformerMixin

helpers_dir = (os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) + '/helpers/')
sys.path.append(helpers_dir)

from feature_extraction import create_tfidf_vectorizer_df, word_count_vectorizer
from parse_csv import normalize_ascii

warnings.filterwarnings("ignore")

class CountVectorizeTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, group_by=None, text_col=None, tfidf_terms: int = 20):
        self._group_by = group_by
        self._text_col = text_col
        self._tfidf_terms = tfidf_terms
        self._nlp = es_core_news_sm.load()
        self._snowball_stemmer = SnowballStemmer(language='spanish')

    def fit(self, X, y=None):
        return self

    def transform_features(self, X):
        tf_idf_scores = create_tfidf_vectorizer_df(X, self._group_by, self._text_col, self._tfidf_terms)
        word_vector_df = word_count_vectorizer(tf_idf_scores, self._group_by, self._text_col)
        return word_vector_df

    def count_vectorize_preprocessor(self, data_row: str):
        # transform text to lowercase and remove punct
        data_row = data_row.lower()
        data_row = "".join(word for word in data_row if word not in set(string.punctuation))
        return data_row

    def lemmatize_data(self, text):
        doc = self._nlp(text)
        lemmatized_text = " ".join([token.lemma_ for token in doc])
        return lemmatized_text

    def stem_data(self, text):
        if(pd.isna(text)):
            return text
        stemmed_text = "".join([self._snowball_stemmer.stem(word) for word in text])
        return stemmed_text

    def transform(self, X, y=None):
        X = normalize_ascii(dataframe=X, columns=[self._text_col])
        X[self._text_col] = X[self._text_col].apply(self.count_vectorize_preprocessor)
        X[self._text_col] = X[self._text_col].apply(self.lemmatize_data)
        X[self._text_col] = X[self._text_col].apply(self.stem_data)
        features = self.transform_features(X)
        features = features.drop(columns=["ComplaintDetail"], axis=1)
        return features
    def fit_transform(self, X, y, **fit_params):
        return self.fit(X, y).transform(X, y)

# class TfidfDataFrame(TfidfVectorizer):
#     def count_vectorize_preprocessor(self, data_row: str):
#         # transform text to lowercase and remove punct
#         data_row = data_row.lower()
#         data_row = "".join(word for word in data_row if word not in set(string.punctuation))
#         return data_row

#     def lemmatize_data(self, text):
#         doc = _nlp(text)
#         lemmatized_text = " ".join([token.lemma_ for token in doc])
#         return lemmatized_text

#     def stem_data(self, text):
#         if(pd.isna(text)):
#             return text
#         stemmed_text = "".join([_snowball_stemmer.stem(word) for word in text])
#         return stemmed_text

#     def transform(self, raw_documents, copy=True):
#         X = super().transform(raw_documents, copy=copy)
#         tfidf_df = pd.DataFrame(X.toarray(), columns=self.get_feature_names())
#         return tfidf_df

#     def fit_transform(self, raw_documents, y=None):
#         raw_documents = normalize_ascii(raw_documents, ["ComplaintDetail"]).squeeze()
#         raw_documents = raw_documents.apply(self.count_vectorize_preprocessor)
#         raw_documents = raw_documents.apply(self.lemmatize_data)
#         raw_documents = raw_documents.apply(self.stem_data)
#         X = super().fit_transform(raw_documents, y=y)
#         tfifd_df = pd.DataFrame(X.toarray, columns=self.get_feature_names())
#         return tfifd_df