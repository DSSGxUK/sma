# The Relevance Model
This page will walk through setting up the relevance model for classification and logging outputs to mlflow
<hr>

## Introduction
This guide will go through preparing data and training a simple model for classifying complaints received by SMA into three categories: `Relevant` (the complaint is relevant to SMA), `Derivacion` (the complaint should be redirected to a different organisation) or `Archivo I` (the complaint should be archived).
The classification is made based on the complaint details and other features that we can extract from the data. This guide will walk through setting up data to making predictions with a model in five steps

## Step 1 - Data Exploration and Data Preprocessing
The first step would be retrieving the data for classification and exploring this data to see what features (columns) are available and how we can transform these columns to features.

!!! note "Features and Columns"
    The words "Features" and "Columns" may be used interchangeably throughout this context. They will generally refer to the data passed into the training loop. However, distinctions will be made when necessary.

### Getting Ready
For this quickstart walkthrough, we will be using the sample data stored in `$PROJECT_ROOT/sample_data/prediction_sample_data.csv`.

1. Create a new `.py` script in `$PROJECT_ROOT/sma/`.
```shell
$ touch sma/walkthrough.py
```
2. Import dependencies into the file. Open the file created in the previous step in a text editor and import the following dependencies

```python
# import project dependencies
import os
import sys
import pandas as pd
import numpy as np
from unidecode import unidecode
from collections import Counter
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler, SMOTE
import itertools
import random

import gensim
import gensim.corpora as corpora
from gensim.models.phrases import Phrases, Phraser
from gensim.utils import simple_preprocess

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import CountVectorizer

helpers_dir = (os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) + '/helpers/')
sys.path.append(helpers_dir)
sma_dir = (os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')) + '/sma/')
sys.path.append(sma_dir)
sma_project_dir = (os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')) + '/sma-project/')
sys.path.append(sma_project_dir)
from feature_extraction import concat_complaint_details, pivot_env_topics, \
                                num_words_all, num_details_all, \
                                create_tfidf_vectorizer_df, word_count_vectorizer
from feature_transformation import num_words, natural_region, facility_mentioned, \
                                    proportion_urban, proportion_protected, num_past_sanctions
from parse_csv import normalize_ascii
from helpers.helper_function import (normalize_text, lemmatize_and_stem_text)

from feature_union import FeatureUnification
from model_metrics import ModelMetrics
from rake_extraction import DenseRakeFeatures
from tfidf_transform import DenseTFIDFVectorization
from feature_selection import FeatureSelection
from train_model import TrainModel
from topic_models import TopicScores
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import mlflow
```

In the previous step, we imported both the `sklearn` and `mlflow` dependencies as well as other modules included in this project. We will provide a brief explanation of each module as we progress.

## Step 2 - Data Selection
Now that we have imported the dependencies we can begin by exploring the data and selecting the columns we will use for classification. We will create a `transform_raw_data_to_features()` function that will transform the raw data into various features for classification.

Let's start by defining the function as
```python
def transform_raw_data_to_features(data_path):
    ...
```

The function accepts one argument, `data_path` which will be the path of the sample data file from the Step 1

Next, we want to read the sample data into a dataframe.
```python
# import the sample data as a dataframe
input_dataframe = pd.read_csv(data_path)

# Grab the columns we are interested in
complaint_details = input_dataframe["ComplaintDetail"]
environmental_topics = input_dataframe["EnvironmentalTopic"].to_numpy().reshape(-1, 1)
complaint_type = input_dataframe["ComplaintType"].to_numpy().reshape(-1, 1)

# Get the classification target column
classification_target = input_dataframe["Relevancy"]
```

In the lines above, we import the the sample data and we take three columns from the data: the `ComplaintDetail`, `EnvironmentalTopic` and `ComplaintType`. We will use these three columns to build our feature set which will be used for classification.

## Step 3 - Feature Transformation Pipeline
In the next few steps, we will setup the feature transformation steps and apply the feature transformations for each feature. But first, here is a list of the feature transformations we are going to perform for each column

|Column Name|Transformation(s)|
|:---|:---|
|`ComplaintDetail`|TF-IDF<br>LDA Topic Modelling<br>RAKE keywords |
|`EnvironmentalTopic`|OneHotEncoding|
|`ComplaintType`|OneHotEncoding|

### Apply feature transformations
We will apply three different transformations to the complaint detail text. The TF-IDF extracts term frequency inverse document frequency scores from the corpus, while the LDA topic modelling extracts various topics from the data. And finally, we will use a RAKE (Rapid Keyword Extraction) to extract key phrases from the data, then create a binary column that checks if each data point in the data corpus contains the any of the RAKE keywords

Next we will apply a OneHotEncode on the Environmental Topic and Complaint Type columns.

```python
# initialize pipeline steps
dense_tfidf = DenseTFIDFVectorization(ngram_range=(
    1, 1), max_features=20)
dense_rake = DenseRakeFeatures()
env_onehot_encoder = OneHotEncoder()
complaints_onehot_encoder = OneHotEncoder()

# fit data to pipeline
dtidf = dense_tfidf.fit_transform(complaint_details)
rake_features = dense_rake.fit_transform(complaint_details)
topic_scores = TopicScores(data=complaint_details, num_topics=10)

# get the environmental topics as a dataframe
environment_topic_features = env_onehot_encoder.fit_transform(environmental_topics)
env_topic_features_df = pd.DataFrame(environment_topic_features.toarray(
), columns=env_onehot_encoder.get_feature_names())

complaint_type_features = complaints_onehot_encoder.fit_transform(complaint_type)
# get the complaint features as a dataframe
complaint_type_features_df = pd.DataFrame(complaint_type_features.toarray(
), columns=complaints_onehot_encoder.get_feature_names())
```

### Explanations of the steps
1. The TF-IDF step builds a TFIDF matrix by using the complaint details. The maximum number of tf-idf terms to extract from the corpus is specified by the `max_features` parameter. See [tfidf_transform](api-docs/tfidf_transform.md)]
2. The TopicScores function uses Latent Dirichlet Allocation (LDA) to extract various topic scores from the complaint detail corpus. See [TopicModelling](api-docs/topic_modelling.md) for more details

### Unify all features
Now that we have performed the feature transformations, we can combine these features into one dataframe by using the `FeatureUnification` class.
```python
features = FeatureUnification().unify([
        dtidf,
        topic_scores,
        env_topic_features_df,
        complaint_type_features_df
    ])
```

In this step we define a variable `features` to hold all the features. We then call the `FeatureUnification().unify()` method passing in a list of all the features we want to join together. This returns a dataframe that we can then split into a training and testing set.

!!! note "Pro-tip"
    You can add other features to the feature union lists including other columns from the dataset to create the unified features. Although keep in mind that the length of each feature has to be the same

## Step 4 - Model Training
Now that we have a set of features, we can split these features into a train-test split using Sklearn. To do so, run
```python
# split the data to training and testing
X_train, X_test, y_train, y_test = train_test_split(features, y, train_size=0.7, random_state=17)
```

In this example, we are using a 70% of the available data fro training and the other 30% for testing.

At this point, the `transform_raw_data_to_features` function should look like this:

::: sma.usage_example.transform_raw_data_to_features

Next, we will define a `run_experiment` function that will take in the training and testing data and run the experiment. 

!!! caution "Remember to set the experiment name"
    It is important to set the experiment name before beginning the experiment.

In the `run_experiment` function we start by setting the experiment name as follows:
```python
mlflow.set_experiment(experiment_name=experiment_name)
```

Next, we will use `MLflow` to setup and run the experiment, and log various metrics and parameters to MLflow. Using a context manager, define the experiment as
```python
with mlflow.start_run(nested=True) as experiment_run:
    run_id = experiment_run.info.run_id
    # get the experiment run id
    run_id = experiment_run.info.run_id

    # train model
    sanction_gravity_model = TrainModel(X_train=X_train, y_train=y_train, model_type=model_type)
    model = sanction_gravity_model.train()
```

### Explanation
1. Once we define the context, we run the experinment as in the lines above. We also grab the experiment `run_id` so that we can use this to make predictions once the model is done training. 
2. We then define a train model step by calling the `TrainModel` class with the model training parameters.
!!! note "The `model_type` parameter
    The model type parameter defines what kind of estimator (algorithm) to use for the training step. The model type can be one of `[logistic_regression, svm_classifier, xgboost, random_forest, naive_bayes ]`
3. Calling the `.train()` method in the `TrainModel` class initializes the training step

Once we the model is done training, we can make predictions by calling
```python
predictions = model.predict(X_test)
```
These predictions are then written into the `model_prediction` SQL table. [THIS NEEDS MORE DETAILS]

# Conclusion
In this walk through, we saw how to create an run a classification model for sanction severity prediction. The results from the model are logged to Mlflow via custom classes. The predictions from the model are then logged to the predictions database. In the next section, we will discuss the details of the feature engineering, providing details on how to perform various feature transformations, recommended parameter values to use during feature transformation, and how to keep track of the experiments. The *[Deploying a Model]* section will provide details of how to deploy the model and use the model in production.