from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

class DataPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, add_binary_col=True, binary_from=None, drop_cols=False):
        self._make_binary_from = binary_from
        self._add_binary_column = add_binary_col
        self._drop_cols = drop_cols

    def fit(self, X, y=None, **fit_params):
        return self;
    
    def add_binary_column(self, EndType: str):
        if(EndType == "Derivacion Total a Organismo Competente" or EndType == "Archivo I"):
            return "Irrelevant"
        elif (EndType == "Formulacion de Cargos" or EndType == "Archivo II"):
            return "Relevant"
        else: 
            return np.nan
    
    def transform(self, X, y = None):
        # Only apply the transfrom on the column if it exist, else return
        if(self._add_binary_column) and (self._make_binary_from in X):
            X["Relevancy"] = X[self._make_binary_from].apply(self.add_binary_column)
        return X

    def fit_transform(self, X, y, **fit_params):
        return self.fit(X, y).transform(X, y)
