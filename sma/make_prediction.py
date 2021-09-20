from argparse import ArgumentParser
from datetime import datetime

import mlflow
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder

from feature_selection import FeatureSelection
from feature_union import FeatureUnification
from model_metrics import ModelMetrics
from rake_extraction import DenseRakeFeatures
from tfidf_transform import DenseTFIDFVectorization
from topic_models import TopicScores
from train_model import TrainModel


def transform(df):
    X = df["ComplaintDetail"]

    # initialize pipeline steps
    dense_tfidf = DenseTFIDFVectorization(ngram_range=(1, 1), max_features=25)
    dense_rake = DenseRakeFeatures()
    env_topics = df["EnvironmentalTopic"].to_numpy().reshape(-1, 1)
    complaint_type = df["ComplaintType"].to_numpy().reshape(-1, 1)
    env_onehot_encoder = OneHotEncoder()
    complaints_onehot_encoder = OneHotEncoder()

    # fit data to pipeline
    environment_topic_features = env_onehot_encoder.fit_transform(env_topics)
    complaint_type_features = complaints_onehot_encoder.fit_transform(
        complaint_type)
    dtidf = dense_tfidf.fit_transform(X)
    rake_features = dense_rake.fit_transform(X)

    # To perform RAKE feature extraction on multi classes call the fit_with_classes method.
    # Example
    # X = df["ComplaintDetail"]
    # y = df["EndType"]
    # rake_features = dense_rake.fit_with_classes(X, y)

    topic_scores = TopicScores(data=X, num_topics=11)
    env_topic_features_df = pd.DataFrame(environment_topic_features.toarray(
    ), columns=env_onehot_encoder.get_feature_names())
    complaint_type_features_df = pd.DataFrame(complaint_type_features.toarray(
    ), columns=complaints_onehot_encoder.get_feature_names())
    #number_feature = df["Number"].to_frame()

    # unify all features
    features = FeatureUnification().unify([
        dtidf,
        topic_scores,
        env_topic_features_df,
        complaint_type_features_df,
        rake_features
        # number_feature
    ])
    #fs = FeatureSelection(n_features=5, estimator_type="classifier")
    #top_features = fs.fit_transform(features)
    pca = PCA(n_components=10)
    top_features = pca.fit_transform(features)
    return top_features


def make_prediction(run_uuid, data):
    if run_uuid is None:
        raise ValueError("A UUID for the run must be specified")
    logged_model = f'runs:/{run_uuid}/model'

    # Load model as a PyFuncModel.
    loaded_model = mlflow.pyfunc.load_model(logged_model)

    # Predict on a Pandas DataFrame.
    featues = transform(data)
    return loaded_model.predict(featues)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--run_id", help="Specify the UUID of the MLflow run")
    parser.add_argument(
        "--data", help="Path of the data to use for classification")
    args = parser.parse_args()

    sample_prediction_data = pd.read_csv(args.data)
    complaint_ids = sample_prediction_data["ComplaintId"]

    predictions = make_prediction(args.run_id, sample_prediction_data)
    # get the model type from mlflow
    model_type = mlflow.get_run(args.run_id).data.tags.get("model_type", pd.NA)

    # These predictions can either be logged to a database or sent back via an API call
    # or saved as a csv file
    model_predictions_df = pd.DataFrame({
        "complaint_id": complaint_ids,
        "relevancy_prediction": predictions,
        "model_type": model_type,
        "prediction_timestamp": datetime.now().isoformat(),
    })

    print(model_predictions_df)
    model_predictions_df.to_csv("model_predictions.csv")
