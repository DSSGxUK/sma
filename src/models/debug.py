import pandas as pd
import numpy as np
from sklearn.base import TransformerMixin, BaseEstimator

class Debug(BaseEstimator, TransformerMixin):
    def __init__(self, step, initiate_debugger=False) -> None:
        self.step = step
        self.init_debugger = initiate_debugger

    def fit(self, X, y=None, at = None, **fit_params):
        return self
    
    def transform(self, X, y=None, **fit_params):
        print(f"Step ===> {self.step}")
        if(isinstance(X, pd.DataFrame) or isinstance(X, pd.Series) or isinstance(X, np.ndarray)):
            print(X.shape)
        if(self.init_debugger):
            import pdb
            pdb.set_trace()
        if(not isinstance(X, list)):
            print(X.shape)
        return X
