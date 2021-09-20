from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import (f_regression, chi2)
import pandas as pd
import numpy as np

class FeatureSelection():
    def __init__(self, n_features="all", estimator_type="classifier"):
        """Select the top N features from a feature space

        Args:
            n_features (str, int, optional): Determines the number of features to select. If "all" is passed, function returns all features. Defaults to "all".
            estimator_type (str, optional): Determins what scoring function to use. This will use `chi2` for classifiers and `f_regression` for regressors. Defaults to "classifier".
        """
        self.n_features = n_features
        self.estimator_type = estimator_type
        self._support_mask = None
        self._all_feature_names = None
    
    def fit(self, features, target=None):
        """Fit the features to the estimator

        Args:
            features (pd.DataFrame, numpy.ndarray): The list of features to select from
            target (pd.Series, numpy.ndarray, optional): The target value for each data point. Defaults to None.

        Returns:
            self: The estimator
        """
        return self
    
    def transform(self, features, target=None):
        """Transform the fitted selector

        Args:
            features (pd.DataFrame, numpy.ndarray): The list of features to select from
            target (pd.Series, numpy.ndarray, optional): The target values for each data point. Defaults to None.

        Raises:
            AssertionError: Raises error if estimator type is not one of [classifier, regressor]

        Returns:
            numpy.ndarray: A feature space with the most important features selected
        """
        selector = None
        if(self.estimator_type == "regressor"):
            selector = SelectKBest(score_func=f_regression, k=self.n_features)
        elif(self.estimator_type == "classifier"):
            selector = SelectKBest(score_func=chi2, k=self.n_features)
        else:
            raise AssertionError("Estimator type must be one of [classifier, regressor]")
        if(isinstance(features, pd.DataFrame)):
            self._all_feature_names = features.columns

        new_features = selector.fit_transform(features, target)
        self._support_mask = selector.get_support()
        self.param_names = selector._get_param_names()
        return new_features
    
    def fit_transform(self, features, target=None):
        """Fit the features to the estimator, and apply the tranformation

        Args:
            features (pd.DataFrame, numpy.ndarray): The list of features to select from
            target (pd.Series, numpy.ndarray, optional): The target values for each data point. Defaults to None.

        Returns:
            numpy.ndarray: A feature space with the most important features selected
        """
        return self.fit(features).transform(features=features, target=target)

    def get_feature_names(self):
        """Returns the names of the features using the generated feature support mask

        Returns:
            list: The names of the most important features
        """
        fns = np.array(self._all_feature_names)
        feature_names = fns[self._support_mask]
        return feature_names.tolist()