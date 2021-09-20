import logging
from context import text_cleaning
# from text_cleaning import count_vectorizer_preprocessor, make_stop_word_list
from urllib.parse import urlparse
from sklearn import linear_model
from sklearn.pipeline import Pipeline

import yaml
from tqdm import tqdm
from datetime import datetime
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from typing import List, Any

from .feature_selector import FeatureSelector
from pre_process_data import DataPreprocessor
from tfidf_vectorization import TfidfVectorization

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

def import_data(path: str) -> pd.DataFrame:
    """Import data for model
    
    Parameters:
        `path`: Path to the data file
    Returns:
        A dataframe containing the data for modeling
    """
    df = pd.read_csv(path, encoding="utf-8")
    return df

def split_data(df: pd.DataFrame, feature_col: str, label_col: str, train_size: float = 0.7) -> List[Any]:
    """Split the data into a train, test set

    Parameters:
        `df`: A dataframe to split
        `feature_col`: Specifies which column in the data will be used as a features column
        `label_col`: Specifies which column in the data will be used for labels
    Returns:
        A list of train, test sets, and the unique values of the labels
    """
    df = df[[feature_col, label_col]].dropna()
    X = df[feature_col]
    y = df[label_col]

    labels = y.unique().tolist()
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, random_state=21)
    return X_train, X_test, y_train, y_test, labels

def build_model(X_train, y_train, alpha: float, l1_ratio: float) -> Pipeline:
    """Build the model pipeline
    
    Parameters:
        `X_train`: The training dataset
        `y_train`: The labels for the training dataset
        `alpha`: Alpha hyper-parameter for the model training step
        `l1_ratio`: L1 ratio hyper-parameter for the model training step
    Returns:
        A model pipeline that can be used for prediction
    """

    # The future pipeline will look like this 
    # ('feature_selection', FeatureSelector(feature_names=feature_names)),
    model = Pipeline([('feature_pre_proecessing', DataPreprocessor()),
       ('count', CountVectorizer(stop_words=text_cleaning.make_stop_word_list(), preprocessor=text_cleaning.count_vectorizer_preprocessor)),
       ('vect', TfidfTransformer()),
       ('clf', SGDClassifier(loss="hinge", alpha=alpha, l1_ratio=l1_ratio)),
    ])

    # model = Pipeline([('vect', CountVectorizer(stop_words="english")),
    #        ('tfidf', TfidfTransformer()),
    #        ('clf', SGDClassifier(loss="hinge", alpha=alpha, l1_ratio=l1_ratio)),
    #       ])
    return model.fit(X_train, y_train) if y_train is not None else model.fit(X_train)

def predict_with_model(model: np.ndarray, data_point: Any):
    """Predict new outcomes using the model
    
    Parameters:
        `model`: The model pipeline
        `data_point`: The data point to use in the prediction
    Returns:
        A predicted label for the data point
    """
    return model.predict(data_point)

def eval_metrics(ground_truth, predicted, target_names=None) -> tuple:
    """Get basic evaluation metrics from the model
    
    Parameters:
        `ground_truth`: Ground truths (correct) target values
        `predicted`: Predicted target values returned by the classifier
        `target_name`: Optional display names matching the labels
    Returns:
        Model accuracy and the classification summary (F1 Score, Precision, Recall, etc)
    """
    accuracy = accuracy_score(ground_truth, predicted)
    report = classification_report(ground_truth, predicted, target_names=target_names, output_dict=True)
    return accuracy, report

def train(hyper_params, model_config, auto_log=False):
    """Execute the model building steps

    """
    # get the hyper params
    alpha = hyper_params["alpha"]
    l1_ratio = hyper_params["l1_ratio"]

    # model params from config file
    data_path = f'{model_config["data"]["loc"]}/{model_config["data"]["file"]}'
    feature_col = model_config["features"]["feature_col"]
    label_col = model_config["features"]["target_col"]

    # import and process data
    data = import_data(path=data_path)

    # split into train, text sets
    X_train, X_test, y_train, y_test, labels = split_data(df=data, feature_col=feature_col, label_col=label_col)
    
    # set model logging
    if(auto_log):
        mlflow.autolog()
        
    # initialize mlflow
    with mlflow.start_run(nested=True):
        # build the model
        model = build_model(X_train, y_train, alpha=alpha, l1_ratio=l1_ratio)
        predicted = predict_with_model(model, X_test)
        accuracy, report = eval_metrics(y_test, predicted, target_names=labels)

        # log metrics
        mlflow.log_params(hyper_params)
        # @TODO write a simple function that parses the dictionary outputs
        # from the classification report and track the output metrics with mlflow
        
        # mlflow.log_metrics(report)
        mlflow.log_metric("Accuracy", accuracy)
        # set model tracking
        tracking_uri = urlparse(mlflow.get_tracking_uri()).scheme
        if tracking_uri != "file":
            mlflow.sklearn.log_model(model, "model", registered_model_name="SGDClassifier")
        else:
            mlflow.sklearn.log_model(model, "model")
    mlflow.end_run()

if __name__ == "__main__":
    with open("configs/svm_model_config.yaml") as config:
        model_config = yaml.safe_load(config)
    
    experiment_id = f'{model_config["meta"]["model"]}_{datetime.now().isoformat(sep="_", timespec="milliseconds")}'
    mlflow.set_experiment(experiment_id)

    # run experiments
    for i, alpha in enumerate(tqdm(model_config["parameters"]["alpha"])):
        l1_ratio = model_config["parameters"]["l1_ratio"][i]
        hyper_params = {'alpha': alpha, 'l1_ratio': l1_ratio}
        train(hyper_params=hyper_params, model_config=model_config)
    print(f"Experiment {experiment_id} completed. Run `mlflow ui` to view results")