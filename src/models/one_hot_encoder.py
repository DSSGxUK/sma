import numpy as np
import pandas as pd
import warnings

from sklearn.base import BaseEstimator, TransformerMixin

from helpers.one_hot_encoding import OneHotEncode

warnings.filterwarnings("ignore")

class OneHotEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        super().__init__()
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None, **transfrom_params):
        encoded_col = OneHotEncode(df = X)# append instead of replacing
        encoded_col_df = pd.DataFrame(encoded_col)
        res_df = pd.concat([X, encoded_col_df], axis=1)
        return res_df