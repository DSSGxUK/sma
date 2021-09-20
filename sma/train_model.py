from pandas.core.algorithms import mode
from sklearn.ensemble import (GradientBoostingRegressor,
                              RandomForestClassifier, RandomForestRegressor)
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.svm import LinearSVC
import numpy as np


class TrainModel():
    def __init__(self, X_train, y_train, model_type, hyper_params=None, training_configs=None):
        """Train a machine learning model

        Args:
            X_train (np.ndarray): Training features
            y_train (array-like): Training targets
            model_type (str): Specifies the type of model to create. Should be one of [lr, smv, nb, rf, rfr, xg]
            hyper_params (dict, optional): Parameters to pass to the model. Defaults to None.
            training_configs (dict, optional): Configurations to use f. Defaults to None.
        """
        self.X_train = X_train
        self.y_train = y_train
        self.model_type = model_type
        self.hyper_params = hyper_params
        self.training_configs = training_configs
        self.feature_importances_ = None
        self._model_params = None

    def train(self):
        """Train a model using training dataset and target values

        Train a model using an sklearn estimator implementation. 
        A `model_type` must be specified for the training step to run. Otherwise, an error is raised.
        The `model_type` can be one of [logistic_regression, linear_svc, naive_bayes, random_forest, random_forest_regressor, xgboost].
        The `model_type` can also be specified using short-hand codes to specify the model.
        The short-hand codes can be one of [lr, smv, nb, rf, rfr, xg]

        Returns:
            object: A fitted model

        Raises:
            AssertionError: Raises an error if the specified model type is not one of
            [logistic_regression, linear_svc, naive_bayes, random_forest]

        Examples:
            >>> X_train = pd.DataFrame({'feature_1': [0, 1, 1, 0, 1, 0, 1, 1, 1, 0],
                            'feature_2': [1, 3, 5, 4, 4, 5, 2, 3, 2, 1]})
            >>> y_train = [0, 0, 1, 0, 1, 1, 0, 1, 1, 0]
            >>> tm = TrainModel(X_train=X_train, y_train=y_train, model_type="lr")
            >>> model = tm.train()
            >>> # predict 
            >>> predicted_class = model.predict(pd.DataFrame({'feature_1': [1], 'feature_2': [2]}))
            >>> print(predicted_class)
        """
        try:
            if(self.model_type in ["logistic_regression", "lr"]):
                clf = LogisticRegression()
                if(self.hyper_params is None):
                    self.hyper_params = clf.get_params()
                clf = clf.set_params(**self.hyper_params)
                fitted_model = clf.fit(self.X_train, self.y_train)
                self.feature_importances_ = clf.coef_
                self._model_params = clf.get_params()
                return fitted_model

            if(self.model_type in ["linear_svc", "svm"]):
                clf = LinearSVC()
                if(self.hyper_params is None):
                    self.hyper_params = clf.get_params()
                clf = clf.set_params(**self.hyper_params)
                fitted_model = clf.fit(self.X_train, self.y_train)
                self.feature_importances_ = clf.coef_
                self._model_params = clf.get_params()
                return fitted_model

            if(self.model_type in ["naive_bayes", "nb"]):
                clf = GaussianNB()
                if(self.hyper_params is None):
                    self.hyper_params = clf.get_params()
                clf = clf.set_params(**self.hyper_params)
                fitted_model = clf.fit(self.X_train, self.y_train)
                self._model_params = clf.get_params()
                return fitted_model

            if(self.model_type in ["random_forest", "rf"]):
                clf = RandomForestClassifier()
                if(self.hyper_params is None):
                    self.hyper_params = clf.get_params()
                clf = clf.set_params(**self.hyper_params)
                fitted_model = clf.fit(self.X_train, self.y_train)
                self._model_params = clf.get_params()
                self.feature_importances_ = clf.feature_importances_
                return fitted_model

            if(self.model_type in ["random_forest_regressor", "rfr"]):
                clf = RandomForestRegressor()
                if(self.hyper_params is None):
                    self.hyper_params = clf.get_params()
                clf = clf.set_params(**self.hyper_params)
                fitted_model = clf.fit(self.X_train, self.y_train)
                self._model_params = clf.get_params()
                self.feature_importances_ = clf.feature_importances_
                return fitted_model

            if (self.model_type in ["xgboost", "xg"]):
                clf = GradientBoostingRegressor()
                if(self.hyper_params is None):
                    self.hyper_params = clf.get_params()
                clf = clf.set_params(**self.hyper_params)
                fitted_model = clf.fit(self.X_train, self.y_train)
                self._model_params = clf.get_params()
                self.feature_importances_ = clf.feature_importances_
                return fitted_model
        except AssertionError as err:
            err.args += """Model type not recognized. Model type should be one of 
            [logistic_regression, linear_svc, naive_bayes, random_forest, random_forest_regressor, xgboost]"""
            raise
