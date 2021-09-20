from sklearn import metrics
import seaborn as sn
import matplotlib.pyplot as plt
import mlflow
import json
import pandas as pd
import numpy as np
from yellowbrick.regressor import ResidualsPlot
from yellowbrick.regressor import PredictionError
import operator
from itertools import islice
import tempfile
import os

# Get project root path
PROJECT_ROOT_DIR = os.path.abspath(os.curdir)

class ModelMetrics():
    """Get the model evaluation metrics for a classifier, including accuracy score, precision recall and F1Score

    Attributes:
        y_true (array-like): Ground truth (correct) labels
        y_pred (array-like): Estimated targets as returned by the classifier

    Example:
        >>> y_true = [0, 1, 2, 1, 2, 0, 0]
        >>> y_pred = [0, 1, 0, 1, 2, 0, 1]

        >>> metrics = ModelMetrics(y_true=y_true, y_pred=y_pred)
        >>> model_classification_metrics = metrics.classification_metrics(include_report=False)
        >>> print(model_classification_metrics)
    """

    def __init__(self, y_true, y_pred) -> None:
        self.y_true = y_true
        self.y_pred = y_pred

    def classification_metrics(self, include_report=False, average="weighted"):
        """Generate evaluation metrics for a classification model

        Args:
            include_report (bool, optional): Indicated if the classification report should be included. Defaults to False.

        Returns:
            tuple[dict, str | None]: Returns the classification metric along with the classification report as a json object if specified

        """

        confusion_mtrx = metrics.confusion_matrix(self.y_true, self.y_pred)
        c_matrix = pd.DataFrame(confusion_mtrx, range(
            len(confusion_mtrx)), range(len(confusion_mtrx)))
        visualize_matrix = sn.heatmap(c_matrix, annot=True)
        with tempfile.TemporaryDirectory() as temp_dir:
            visualize_matrix.figure.savefig(f"{temp_dir}/confusion_matrix.png")
            mlflow.log_artifact(
                f"{temp_dir}/confusion_matrix.png", artifact_path="visualizations")
            plt.close()

        evaluation_metrics = dict(
            model_accuracy=metrics.accuracy_score(
                y_true=self.y_true, y_pred=self.y_pred),
            f1_score=metrics.f1_score(
                y_true=self.y_true, y_pred=self.y_pred, average=average),
            precision_score=metrics.precision_score(
                y_true=self.y_true, y_pred=self.y_pred, average=average),
            recall_score=metrics.recall_score(
                y_true=self.y_true, y_pred=self.y_pred, average=average)
        )
        if(len(confusion_mtrx) == 4):
            tn, fp, fn, tp = confusion_mtrx
            evaluation_metrics['tn'] = tn
            evaluation_metrics['tp'] = tp
            evaluation_metrics['fn'] = fn
            evaluation_metrics['fp'] = fp
        if(include_report):
            classification_report = metrics.classification_report(
                y_true=self.y_true, y_pred=self.y_pred, output_dict=True)
            report_json = json.dumps(classification_report, indent=2)
            return evaluation_metrics, report_json
        return evaluation_metrics, None

    def regression_metrics(self, include_report=False, classification_metric=False):
        """Generate evaluation metrics for a regression estimator. Classification metrics will be appended to the output if the param `classification_metrics` is set to `True`.

        Args:
            include_report (bool, optional): Include a classification report. The report contains a json object with information about every class in the classification target. Defaults to False.
            classification_metric (bool, optional): Include the classification metrics along with the regression metrics. Defaults to False
        Returns:
            dict: Evaluation metrics for the model
        """
        # lr_metrics = dict(
        #     RMSE=np.sqrt(metrics.mean_squared_error(self.y_true, self.y_pred)),
        #     max_error=metrics.max_error(self.y_true, self.y_pred),
        #     mean_absolute_error=metrics.mean_absolute_error(
        #         self.y_true, self.y_pred),
        #     explained_variance_score=metrics.explained_variance_score(
        #         self.y_true, self.y_pred),
        #     r_squared=metrics.r2_score(self.y_true, self.y_pred)
        # )
        if(classification_metric):
            clf_metrics, report = self.classification_metrics(
                include_report=include_report)
    
        return clf_metrics

    def log_residuals_plot(self, model):
        residual_visualizer = ResidualsPlot(model, hist=False, qqplot=True)
        residual_visualizer.fit(model.X_train, model.y_train.values.ravel())
        residual_visualizer.score(model.X_test, model.y_test)

        prediction_error = PredictionError(model)
        prediction_error.fit(model.X_train, model.y_train)
        prediction_error.score(model.X_test, model.y_test)

        with tempfile.TemporaryDirectory() as temp_dir:
            residual_visualizer.show(outpath=f"{temp_dir}/residual_plot.png")
            prediction_error.show(outpath=f"{temp_dir}/prediction_error_plot.png")
            mlflow.log_artifact(
                f"{temp_dir}/residual_plot.png", artifact_path="visualizations")
            mlflow.log_artifact(
                f"{temp_dir}/prediction_error_plot.png", artifact_path="visualizations")

    def log_hyperparameters_to_mlflow(self, hyper_params=None):
        """Logs the model hyper parameters to MLflow

        Args:
            hyper_params (Dict([str, Any]), optional): Key, Value pairs of the model hyperparameters. Hyper parameter values will be converted to strings. Defaults to None.
        """
        if hyper_params is not None:
            assert isinstance(hyper_params, dict) == True
            mlflow.log_params(hyper_params)

    def log_metric_to_mlflow(self, model_metrics):
        """Log the model metrics to mlflow

        Args:
            model_metrics (dict): Key, value pair of metrics to log to mlflow
        """
        assert isinstance(model_metrics, dict) == True
        for key, value in model_metrics.items():
            mlflow.log_metric(key=key, value=value)

    def log_model_features(self, features_list):
        """Log all the feature names to mlflow in a text file

        Args:
            features_list (List[str]): A list of features to track in mlflow
        """
        features_list = [str(feature) for feature in features_list]
        mlflow.log_text("\n".join(features_list),
                        artifact_file="features/feature_list.txt")

    def log_feature_importance(self, model, feature_names=None, n_features=None):
        """Logs the important features from the model training to MLflow. These features will be logged alongside the model parameters

        Args:
            model: The fitted model
            feature_names (List[str], optional): A list of feature names used in the training. Defaults to None.
            n_features (int): Specifies the number of features to log. If no value is set, all features are logged to mlflow. Defaults to None.
        """
        features_and_importance_score = None
        if feature_names is None:
            # this line should not be here. Rename the column before using it!
            # model.X_train = model.X_train.rename(columns ={'Contaminación lumínica *(Solo para Regiones II, III, IV)':'Contaminación lumínica Solo para Regiones II/III/IV'})
            feature_names = model.X_train.columns
        if hasattr(model, "feature_importances_"):
            importances = zip(feature_names, model.feature_importances_)
            if(n_features is not None):
                importances = zip(feature_names, model.feature_importances_[:n_features])
            importances = sorted(importances, key=lambda k: k[1], reverse=True)
            features_and_importance_score = dict(importances)

        if(features_and_importance_score is not None):
            features_as_json = json.loads('{}')
            features_as_json.update(features_and_importance_score)
            mlflow.log_text(json.dumps(features_as_json, indent=2), artifact_file="features/feature_importance.json")
    
    def save_model_to_disk(self, model, file_path = "model_pkl/"):
        """Write the model file to disk

        Args:
            model (fitted_model): A fitted sklearn model
            file_path (str, optional): Specifies the path where the pikeled model will be saved. Defaults to "model_pkl/".
        """
        mlflow.sklearn.save_model(model, path=file_path, serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE)
