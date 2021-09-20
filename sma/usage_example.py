import mlflow
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from feature_union import FeatureUnification
from model_metrics import ModelMetrics
from rake_extraction import DenseRakeFeatures
from tfidf_transform import DenseTFIDFVectorization
from topic_models import TopicScores
from train_model import TrainModel

PCA_COMPONENTS = 10
NUMBER_OF_LDA_TOPICS = 11
NUMBER_OF_TFIDF_FEATURES = 50 
FEATURE_NAMES = []

def transform_raw_data_to_features(data_path):
    # Import data
    df = pd.read_csv(data_path)
    df = df.dropna(subset=["ComplaintDetail", "Relevancy"])
    X = df["ComplaintDetail"]
    y = df["Relevancy"]

    # initialize pipeline steps
    dense_tfidf = DenseTFIDFVectorization(ngram_range=(
        1, 1), max_features=NUMBER_OF_TFIDF_FEATURES)
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

    topic_scores = TopicScores(data=X, num_topics=NUMBER_OF_LDA_TOPICS)
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

    # set the feature names before running PCA
    # Running PCA changes the feature names, so we want to get the names before running PCA
    FEATURE_NAMES.extend(features.columns)

    # apply PCA for feature dimensionality reduction
    pca = PCA(n_components=PCA_COMPONENTS)
    features = pca.fit_transform(features)

    # create PCA feature names
    pca_column_names = [f"PCA_{i}" for i in range(features.shape[1])]
    features = pd.DataFrame(data=features, columns=pca_column_names)

    # split the data to training and testing
    X_train, X_test, y_train, y_test = train_test_split(
        features, y, train_size=0.7, random_state=17)
    return X_train, X_test, y_train, y_test


def run_experiment(X_train, X_test, y_train, y_test, model_type="rf", experiment_name="Relevance Model"):
    """Sets up and runs an mlflow experiment

    Args:
        X_train (numpy.ndarray, scipy.sparse.csr_matrix, pd.DataFrame): The training data
        X_text (numpy.ndarray, scipy.sparse.csr_matrix, pd.DataFrame): The data used for testing the model performance
        y_train (numpy.ndarray, scipy.sparse.csr_matrix, pd.DataFrame): The target values used for training the model
        y_test (numpy.ndarray, scipy.sparse.csr_matrix, pd.DataFrame): The target values used for evaluating the model
        model_type (str, optional): A model type parameter. The `model type` can be specified using the full name qualifier or the short name.
        `model_type` must be one of [lr, smv, nb, rf, rfr, xg]. Defaults to "rf". See [train_model](train_model.md) for more details
        experiment_name (str, optional): Specifies the experiment name. Defaults to "Relevance Model".
    """
    # IMPORTANT: set the experiment name here
    mlflow.set_experiment(experiment_name=experiment_name)

    run_id = None
    with mlflow.start_run(nested=True) as experiment_run:
        # get the experiment run id
        run_id = experiment_run.info.run_id

        # train model
        m = TrainModel(X_train=X_train, y_train=y_train, model_type=model_type)
        model = m.train()

        # make predictions with model using test set
        predictions = model.predict(X_test)

        # encode labels to prevent misrepresentation in categorical labels
        label_encoder = LabelEncoder()
        predicted_labels = label_encoder.fit_transform(predictions)
        actual_labels = label_encoder.fit_transform(y_test)

        # get the model metrics
        model_metrics = ModelMetrics(
            y_true=actual_labels, y_pred=predicted_labels)
        lr_metrics = model_metrics.regression_metrics(
            include_report=True, classification_metric=True)

        # track feature importance
        model_metrics.log_model_features(FEATURE_NAMES)
        model_metrics.log_feature_importance(m)
        model_metrics.log_hyperparameters_to_mlflow(m._model_params)


        # track the metrics in mlflow
        model_metrics.log_metric_to_mlflow(lr_metrics)
        # track the model in mlflow
        mlflow.sklearn.log_model(model, "model")
        mlflow.set_tag("model_type", model_type)
    mlflow.end_run()
    return run_id


def main():
    """Run the experiment and track all outputs in mlflow
    """
    data_path = "/files/data/data/processed/db/complaints_registry_rel.csv"
    X_train, X_test, y_train, y_test = transform_raw_data_to_features(
        data_path)
    experiment_id = run_experiment(X_train, X_test, y_train, y_test,
                                   model_type="random_forest", experiment_name="Relevance Model")

    print("Experiment Done. Run $ `mlflow ui` to view results in mlflow dashboard")
    print(f"Run $ `python make_prediction.py --run_id ${experiment_id} --data ../sample_data/prediction_sample_data.csv` to use model for predictions")


if __name__ == "__main__":
    main()
