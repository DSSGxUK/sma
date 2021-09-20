# The Sanction Gravity Model
This page will walk through setting up the sanction gravity classification model and logging outputs to mlflow.
<hr>

## Introduction
This guide will go through how to prepar data and train a simple model for classifying sanction severity (low:0, high:1) after a complaint is predicted or assigned as relevant to SMA's remit by Relevancy Model. 

The classification is made based on the complaint details and other features that we can extract from the data. This guide will walk through setting up data to making predictions with a model in several steps.

## Step 1 - Getting Ready

The model training takes place in the `sanction_gravity.py` script. We will walk through the main steps here.

Import dependencies into the file. Open the file created in the previous step in a text editor and import the following dependencies.

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

In this step, we imported both the `sklearn` and `mlflow` dependencies as well as other modules included in this project. We will provide a brief explanation of each module as we progress.

## Step 2 -  Setting the parameters and features to use
Now that we have imported the dependencies we can begin by exploring the data and selecting the columns we will use for classification. We will create a `transform_raw_data_to_features()` function that will transform the raw data into various features for classification.

Let's start by setting the constant values for the number of TF-IDF keywords and topcs for LDA to train For now, we will use the parameters and features which provided the best results in our model development phase, but these can be changed in the future. 

```python
# Set required parameters and the features to use
NUMBER_OF_LDA_TOPICS = 50
NUMBER_OF_TFIDF_FEATURES = 45

exp_features = ['TF-IDF', 'LDA', 'RAKE', 'num_words', 'FacilityRegion', 'EnvironmentalTopic', 
                'natural_region', 'facility_mentioned', 'proportion_urban', 'proportion_protected',
                'num_past_sanctions'
                ]
# Random Forest parameters
h_params = {'max_depth':30, 'max_features': 'sqrt',             'n_estimators': 550, 'max_leaf_nodes' : 4,'min_samples_split': 3, 'min_samples_leaf': 1,'bootstrap': False}
h_params['random_state'] = 17
```

!!! note "Features and Columns"
    The words "Features" and "Columns" may be used interchangeably throughout this context. They will generally refer to the data passed into the training loop. However, distinctions will be made when necessary.

## Step 3 - Set the path to the data files

Now we need to set the and also defining the feature path to each of the required data files, so that the model knows where to find the CSV files it need to read the data in from. We set these data paths in a dictionary as follows (replace `<PATH_TO_FILE>` with the path to each file)

- the `complaints` file containing the complaint details (text) and environmental topics
- the `complaints_facilities` file containing the joined complaints and facilities registries
- the `facilities_sanction_id` file containing the facilities registry joined with the sanction IDs associated with each facility
- the `sanctions` file containing the sanction details

See the [Data Preparation](../set-up/preparing-the-data.md) section for more information about how the data should be prepared.

transformation function as:

```python
# Path to the data files
data_paths = {'complaints':'<PATH_TO_FILE>',
              'complaints_facilities':'<PATH_TO_FILE>',
              'complaints_sanctions':'<PATH_TO_FILE>',
              'sanctions':'<PATH_TO_FILE>'
              }
```
## Step 4 - Data Preparation
We are now ready to create the dataframe which we will pass into our Random Forest model. This step is carried out in the `transform_raw_data_to_features()` function, which converts the raw data into various features for classification. There are many steps to this process and it would be very long and cumbersome to explain all the details of this function. To keep the documentation short and readable, we will describe the main steps of the data preparation pipeline which we have used for this model. This will ensure that anyone is able to run our model and have a high-level understanding of the process. If the reader is interested in modifying the pipeline, we refer them to the code, which should be easy to follow and modify, given the explanations provided here.

The `transform_raw_data_to_features()` functions accept two arguments:

- `exp_features`: the names of the features on which to train the model
- `data_paths`: the dictionary of data paths as defined in Step 3

```python
def transform_raw_data_to_features(exp_features, data_path):
    ...
```

The function accepts two arguments,`exp_features` takes in the list of feature names we want to feed into the model, while `data_path` which will be the path of the sample data file from the Step 3.

For example: 
    exp_features = ['TF-IDf','LDA']
Then we are taking in two predictors, the TF-IDF keywords and topics trained from Topic Modelling.

#### List of Feature Transformations
In the next few steps, we will set up the feature transformation steps and prepare the data to be passed into the model. But first, here is a list of the feature transformations we are going to perform for each column

|Column Name(s)|Transformation(s)|
|:---|:---|
|`ComplaintDetail`|TF-IDF<br>LDA Topic Modeling<br>RAKE keywords<br>Number of words|
|`EnvironmentalTopic`|OneHotEncoding|
|`ComplaintType`|OneHotEncoding|
|`FacilityRegion`|OneHotEncoding<br>Natural region|
|`FacilityId`<br>`DateComplaint`<br>`Sanctions Data`|Number of past different level of sanction infractions for the facility|

!!! note Facility Information
    In some cases, the complainant is unable to identify a specific facility. In these situations, the model is able to deal with this by filling in the fields in question with sensible values.
    
#### Feature Transformation Pipeline
We will now briefly walk through the order of the steps in the feature transformation pipeline. Some of the small intermediary steps have been ommitted in the interest of brevity, but these can be easily understood from the code.

##### 1. Read in and pre-process the data
We start by reading in the data from the specified data file paths.
```python
# Load the data
sanction_df = normalize_ascii(pd.read_csv(data_path['sanction_registry']))
com_reg = normalize_ascii(pd.read_csv(data_path['complaint_registry'])).drop(['Unnamed: 0','Unnamed: 0.1'],1)
com_fac = normalize_ascii(pd.read_csv(data_path['complaints_facilities_registry']))
com_sac_reg = normalize_ascii(pd.read_csv(data_path['complaints_sanctions_registry']))
facility_df = pd.read_csv(data_path['complaints_facilities_registry'])
facilities_sanctions = normalize_ascii(pd.read_csv(data_path['facilities_sanction_id']))
sanctions = pd.read_csv(data_path['sanctions_registry'])

```
##### 2. Create the Target variable
Create the target variables we need to refer to two functions to finish the job, add_target() and add_label().

```python
# Create the Target variable as a column of the dataframe
df = add_target(com_sac_df, method = 'worst').dropna(subset=['SanctionLevel', 'MonetaryPenalty'], how='all')

df['Target'] = df.apply (lambda row: add_label(row), axis=1)
```

##### 3. Concatenate the text details
Concatenate the text from all the complaint details into a single complaint text per `ComplaintId` and add this as a column of the dataframe.

```python
    # Aggregate complaint texts belonging to the same complaintId and add text number feature
    # Create corpus corresponding to each category by extracting the label and concatenate the strings according to category
    com_agg = com_df.groupby(['ComplaintId'], as_index = False).agg({'ComplaintDetail': ' '.join})
    n_complaint = com_df.groupby(['ComplaintId'], as_index = False).Number.max()
    com_agg = (
        com_agg
        .drop_duplicates(subset = ['ComplaintId'])
        .merge(n_complaint, how = 'left')
        .merge(com_df[['ComplaintId', 'EndType']], how = 'inner')
        .drop_duplicates(subset = ['ComplaintId'])
        .dropna(subset = ['Number']))

    #Delete observations out of SMA's remit(outliers).
    to_dropped = com_agg.loc[(com_agg['EndType'] == 'Derivación Total a Organismo Competente') | (com_agg['EndType'] == 'Archivo I')]
    df = (df.merge(com_agg.drop(to_dropped.index, axis = 0), how = 'left')).dropna(subset=['ComplaintDetail'])

    # Conduct text cleaning on ComplaintDetail
    df = (
        df.pipe(clean_text, text_column = 'ComplaintDetail')
        .pipe(lemmatize, text_column = 'cleaned_text')
        .pipe(stemmer, text_column = 'cleaned_text')
        .drop(['cleaned_text','ComplaintDetail','MonetaryPenalty','SanctionLevel','EndType'], axis = 1)
        .rename(columns = {'stemmed':'concat_text'})
        .dropna(subset = ['Number'])
        .dropna(subset = ['concat_text'])
    )
    df['concat_text'] = df['concat_text'].fillna('').astype(str)
```

##### 4. One-hot encode the environmental topics
Get the one-hot encoded environmental topics (if `EnvironmentalTopic` is included in the set of features to pass into our Random Forest) for each of the complaints and add these columns to the dataframe.
```python
# One-hot encode the Environmanetal topics of the complaints
    if 'EnvironmentalTopic' in exp_features:
        env_topic = pivot_env_topics(com_topic)
        df = pd.merge(df, env_topic, on = 'ComplaintId', how = 'left').fillna(0)
```

##### 5. Apply feature transformations
Apply the desired feature transformations from the `feature_transformation.py` file and add these as columns of the dataframe.

```python
if 'populated_districts' in exp_features:
    pop_df = populated_districts(facility_df)
    pop_df = pop_df[['FacilityId','populated_districts']]
    com_fac_df = pd.merge(facility_df[['ComplaintId','FacilityId']], pop_df, on = 'FacilityId', how = 'left')
    com_fac_df['ComplaintId'] = com_fac_df['ComplaintId'].astype(str)
    com_fac_df['FacilityId'] = com_fac_df['ComplaintId'].astype(str)
    com_fac_df = com_fac_df.groupby('ComplaintId').sum().reset_index()
    com_fac_df['ComplaintId'] = com_fac_df['ComplaintId'].astype(int)
    com_fac_df['FacilityId'] = com_fac_df['ComplaintId'].astype(int)
    df = df.merge(com_fac_df[['ComplaintId','populated_districts']], on = 'ComplaintId', how = 'left').drop_duplicates(subset = ['ComplaintId'])
    #df = feature_df.drop(['FacilityId'], axis = 1).fillna(0)


if 'month' in exp_features:
    com_reg = month(com_reg)
    df = df.merge(com_reg[['ComplaintId','month']], on = 'ComplaintId', how = 'left').drop_duplicates(subset = ['ComplaintId'])


if 'quarter' in exp_features:
    com_reg = quarter(com_reg)
    df = df.merge(com_reg[['ComplaintId','quater']], on = 'ComplaintId', how = 'left')
        
if 'FacilityEconomicSector' in exp_features:
    eco_df = pd.concat([facility_df[['FacilityId']],pd.get_dummies(facility_df['FacilityEconomicSector'])], axis=1)
    com_fac_eco = pd.merge(facility_df[['ComplaintId','FacilityId']], eco_df, on = 'FacilityId', how = 'left')
    com_fac_eco['ComplaintId'] = com_fac_eco['ComplaintId'].astype(str)
    com_fac_eco['FacilityId'] = com_fac_eco['ComplaintId'].astype(str)
    com_fac_eco = com_fac_eco.groupby('ComplaintId').sum().reset_index()
    com_fac_eco['ComplaintId'] = com_fac_eco['ComplaintId'].astype(int)
    com_fac_eco['FacilityId'] = com_fac_eco['ComplaintId'].astype(int)
    df = df.merge(com_fac_eco, on = 'ComplaintId', how = 'left').drop(['FacilityId'], axis = 1)


if 'ComplaintType' in exp_features:
    ComType = pd.concat([com_reg[['ComplaintId']],pd.get_dummies(com_reg[['ComplaintType']])], axis=1)
    df = df.merge(ComType, on = 'ComplaintId', how = 'left').fillna(0).drop_duplicates(subset = ['ComplaintId'])


if 'num_details' in exp_features:
    df['num_details'] = df['Number']



# Select the feature transformations to include from the feature_transformations.py file
if 'min_num_words' in exp_features:
    df = min_num_words(df)


if 'max_num_words' in exp_features:
    df = max_num_words(df)


if 'natural_region' in exp_features:
    facility_region = natural_region(facility_df)
    region_df = pd.concat([facility_region[['FacilityId']],pd.get_dummies(facility_region['natural_region'])], axis=1)
    com_fac_df = pd.merge(facility_df[['ComplaintId','FacilityId']],region_df, on = 'FacilityId', how = 'left')
    com_fac_df['ComplaintId'] = com_fac_df['ComplaintId'].astype(str)
    com_fac_df['FacilityId'] = com_fac_df['ComplaintId'].astype(str)
    com_fac_df = com_fac_df.groupby('ComplaintId').sum().reset_index()
    com_fac_df['ComplaintId'] = com_fac_df['ComplaintId'].astype(int)
    com_fac_df['FacilityId'] = com_fac_df['ComplaintId'].astype(int)
    df = df.merge(com_fac_df, on = 'ComplaintId', how = 'left').drop(['FacilityId'], axis = 1).fillna(0)


if 'num_words' in exp_features:
    df = num_words(df)


if 'facility_num_infractions' in exp_features:
    fac_df = get_complaint_sanctions(df=df, complaints_facilities=facility_df, facilities_sanctions=facilities_sanctions, sanctions=sanctions)
    infraction_number = fac_df.groupby(['ComplaintId'], as_index = False).size()
    df = df.merge(infraction_number, how = 'left').rename(columns = {'size':'InfractionNumber'}).drop_duplicates(subset = ['ComplaintId'])
    df['InfractionNumber'] = df['InfractionNumber'].fillna(0)
    low_level_number = fac_df.groupby(['ComplaintId','InfractionCategory'], as_index = False).size()
    low_level = low_level_number[low_level_number['InfractionCategory'] == 'Leves'].drop(['InfractionCategory'], axis = 1)
    df = df.merge(low_level, how = 'left').rename(columns = {'size':'LowInfractionNumber'}).drop_duplicates(subset = ['ComplaintId'])
    df['LowInfractionNumber'] = df['LowInfractionNumber'].fillna(0)
    medium_level_number = fac_df.groupby(['ComplaintId','InfractionCategory'], as_index = False).size()
    medium_level = medium_level_number[medium_level_number['InfractionCategory'] == 'Graves'].drop(['InfractionCategory'], axis = 1)
    df = df.merge(medium_level, how = 'left').rename(columns = {'size':'MediumInfractionNumber'}).drop_duplicates(subset = ['ComplaintId'])
    df['MediumInfractionNumber'] = df['MediumInfractionNumber'].fillna(0)


if 'money_fined' in exp_features:
    fac_df = get_complaint_sanctions(df=df, complaints_facilities=facility_df, facilities_sanctions=facilities_sanctions, sanctions=sanctions)
    money_fined = fac_df.groupby(['ComplaintId'], as_index = False).MonetaryPenalty.sum()
    df = df.merge(money_fined, how = 'left').rename(columns = {'MonetaryPenalty':'MoneyFined'}).drop_duplicates(subset = ['ComplaintId'])
    df['MoneyFined'] = df['MoneyFined'].fillna(0)


if 'facility_num_sanctions' in exp_features:
    fac_df = get_complaint_sanctions(df=df, complaints_facilities=facility_df, facilities_sanctions=facilities_sanctions, sanctions=sanctions)
    sanction_number = (
        fac_df.groupby(['ComplaintId','SanctionId'], as_index = False)
        .size()
        .groupby(['ComplaintId'], as_index = False)
        .size())
    df = df.merge(sanction_number, how = 'left').rename(columns = {'size':'SanctionNumber'}).drop_duplicates(subset = ['ComplaintId'])
    df['SanctionNumber'] = df['SanctionNumber'].fillna(0)

```
##### 6. Split the data into training and test sets
Now let's split the data into a training set and a test set. In this example, we are using a 70% of the available data fro training and the other 30% for testing.
```python
# Split the data to training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=h_params['random_state'])
```

##### 7. Apply the text feature engineering steps
Apply the text feature engineering steps (LDA, TF-IDF and RAKE). It is important that these features are constructed using only on the training set, before being applied to both the training and test sets. We also carry out some pre-processing of the text before apllying these algorithms: stopwords are removed and all words are lemmatized and stemmed. Add the resulting features to the dataframe (if applicable).
```python
# Preprocess the text in the concat_text field to prepare for the text feature extraction
X_train['concat_text'] = X_train['concat_text'].apply(normalize_text).apply(lemmatize_and_stem_text)
X_test['concat_text'] = X_test['concat_text'].apply(normalize_text).apply(lemmatize_and_stem_text)

# Extract the text features
if 'TF-IDF' in exp_features:
    tfidf_words = top_tfidf_words(X_train, NUMBER_OF_TFIDF_FEATURES)
    X_train = tfidf_word_counts(X_train, tfidf_words)
    X_test = tfidf_word_counts(X_test, tfidf_words)

if 'LDA' in exp_features:
    X_train, X_test = perform_topic_modelling(X_train, X_test, NUMBER_OF_LDA_TOPICS)
    print('LDA done.')
```
##### 10. Output the training and test sets
The data has now been fully prepared. The function returns `X_train`, `X_test`, `y_train` and `y_test` for use in the modelling step.

## Step 5 - Train and evaluate the model
Now that we have prepared all the features, only one step remains: the model training and evaluation step. Once again, we have a function which wraps up all the necessary steps together: the `run_experiment()` function. We will provide an overview of this function here.

##### 1. Set the experiment name
In the `run_experiment()` function we start by setting the experiment name as follows:
```python
mlflow.set_experiment(experiment_name=experiment_name)
```

!!! caution "Remember to set the experiment name"
    It is important to set the experiment name before beginning the experiment. This determines where the experiment will be logged in MLflow.

##### 2. Set up the experiment run to train the model
Next, we will use `MLflow` to set up and run the experiment, and log various metrics and parameters to the dashboard. Using a context manager, define the experiment as
```python
with mlflow.start_run(nested=True) as experiment_run:
    run_id = experiment_run.info.run_id
    # get the experiment run id
    run_id = experiment_run.info.run_id

    # train model
    m = TrainModel(X_train=X_train, y_train=y_train, model_type=model_type, hyper_params=h_params)
    model = m.train()
```

A few things are happening here:

1. Once we define the context, we run the experinment as in the lines above. We also grab the experiment `run_id` so that we can use this to make predictions once the model is done training. 
2. We then define a train model step by calling the `TrainModel` class with the model training parameters.
!!! note "The `model_type` parameter
    The model type parameter defines what kind of estimator (algorithm) to use for the training step. The model type can be one of `[logistic_regression, svm_classifier, xgboost, random_forest, naive_bayes ]`
3. Calling the `.train()` method in the `TrainModel` class initializes the training step

##### 3. Use the trained model to predict on the test set
Once the model is done training, we can make predictions by calling
```python
# make predictions with model using test set
predictions = model.predict(X_test)
```

##### 4. Get the evaluation metrics
Use the `ModelMetrics` class to calculate the evaluation metrics.
```python
# get the model metrics
model_metrics = ModelMetrics(
    y_true=actual_labels, y_pred=predicted_labels)
lr_metrics = model_metrics.regression_metrics(
    include_report=True, classification_metric=True)
```

##### 5. Log the parameters and feature information to MLflow
Log the set of features used to create the dataframe, the feature importance returned by the sklearn implementation of the Random Forest and the hyperparameters to MLflow. The feature importance tells us which features contributed more to the classifier than others.
```python
# track feature importance
model_metrics.log_model_features(exp_features)
model_metrics.log_feature_importance(m)
model_metrics.log_hyperparameters_to_mlflow(m._model_params)
print(lr_metrics)
```

##### 6. Log the metrics and the model to MLflow
Log the evaluation metrics to MLflow, as well as the model type (in this case, Random Forest) and the model itself. This will create a .pkl file for the model.
```python
# track the metrics in mlflow
model_metrics.log_metric_to_mlflow(lr_metrics)
# track the model in mlflow
mlflow.sklearn.log_model(model, "model")
mlflow.set_tag("model_type", model_type)
```

##### 7. End th MLflow experiment
Finally, end the MLflow run and return the run ID.
```python
mlflow.end_run()
return run_id
```

## Step 6 - Put everything together
We can now run everything with the following code, which will prepare the data, train the model, evaluate the results on the test set and log everything to MLflow.

!!! note Remember the experiment name
    Remember to set the experiment name, which will determine which section of MLflow all your runs are logged to.

```python
def main():
    X_train, X_test, y_train, y_test = transform_raw_data_to_features(exp_features, data_path)
    experiment_id = run_experiment(X_train, X_test, y_train, y_test, model_type="random_forest", experiment_name="Sanction Gravity Model")

if __name__ == "__main__":
    main()
```

## Step 7 - Run the entire script
To execute the run, simply run the script from the command line with the following python command. Make sure that you are in the correct directory.
```shell
$ python sma/walkthrough.py
```

## Step 8 - View the results in MLflow
All the logged information is now available to view in MLflow. Run the following command in the command line to launch the MLflow dashboard, then open a browser window and paste the web address indicated.
```shell
$ mlflow ui
```


# Conclusion
In this walk through, we saw how to train and run a simple sanction gravity prediction model. The results from the model are logged to Mlflow via custom classes. The predictions from the model are then logged to the predictions database. In the next section, we will discuss the details of the feature engineering, providing details on the various feature transformations we used and considered. The [Model Deployment](../model-deployment/deployment-environment-setup.md) section will provide details of how to deploy the model and use the model in production.
