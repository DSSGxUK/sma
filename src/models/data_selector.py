from sklearn.base import BaseEstimator, TransformerMixin

class DataSelector(BaseEstimator, TransformerMixin):
    def __init__(self, column_names=None, values=None, drop_cols=None, datatype=None):
        self._column_names = column_names
        self._values = values
        self._drop_cols = drop_cols
        self._X = None
        self._datatype = datatype

    def fit(self, X, y = None):
        return self;
    
    def transform(self, X, y = None):
        if(self._drop_cols and self._column_names == "all"):
            return X.drop(columns=self._drop_cols, axis=1, inplace=True)

        if(self._column_names == "all" or self._column_names is None):
            return X

        if(self._datatype is not None):
            return X.select_dtypes(include=self._datatype)
    
        if(self._values):
            return X[self._column_names].squeeze().to_list()
        
        self._X = X[self._column_names]
        return self._X
        
    def fit_transform(self, X, y = None):
        return self.fit(X, y).transform(X, y)
