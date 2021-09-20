# Relevance Model - Walkthrough
This page will walk through setting up the training and testing for the relevance classifier and logging the outputs to MLflow.
<hr>

## Introduction
This guide will go through preparing data and training a simple model for classifying complaints received by SMA into three categories: `Relevant` (the complaint is relevant to SMA), `Derivacion` (the complaint should be redirected to a different organisation) or `Archivo I` (the complaint should be archived).
The classification is made based on the complaint details and other features that we can extract from the data. This guide will walk the user through the entire process, from setting up data to making predictions with the model in just a few steps.

## Step 1 - Getting Ready

The model training takes place in the `relevance_test_train.py` script. We will walk through the main steps here.

Firstly, the required dependencies are imported into the file.

```python
# Import project dependencies
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

In this step, we imported both the `sklearn` and `mlflow` dependencies as well as other modules included in this project.

## Step 2 - Setting the parameters and features to use
The second thing to define is the set of parameters and features to use in our model. For now, we will use the parameters and features which provided the best results in our model development phase, but these can be changed in the future if the model needs to be tweaked.
```python
# Set required parameters and the features to use
NUMBER_OF_LDA_TOPICS = 50
NUMBER_OF_TFIDF_FEATURES = 100
NUMBER_OF_RAKE_PHRASES = 15000
FEATURE_NAMES = ['TF-IDF', 'LDA', 'RAKE', 'num_words', 'FacilityRegion', 'EnvironmentalTopic', 
                'natural_region', 'facility_mentioned', 'proportion_urban', 'proportion_protected',
                'num_past_sanctions'
                ]
# Random Forest parameters
h_params = {'max_depth': 9, 'max_features': 50, 'n_estimators': 100, 'max_leaf_nodes': None}
h_params['random_state'] = 17
```

!!! note "Features and Columns"
    The words "Features" and "Columns" may be used interchangeably throughout this context. They will generally refer to the data passed into the training loop. However, distinctions will be made when necessary.

## Step 3 - Set the path to the data files
We now need to set the path to each of the required data files, so that the model knows where to find the CSV files it need to read the data in from. The CSV files required for this are:

- the `complaints` file containing the complaint details (text) and environmental topics
- the `complaints_facilities` file containing the joined complaints and facilities registries
- the `facilities_sanction_id` file containing the facilities registry joined with the sanction IDs associated with each facility
- the `sanctions` file containing the sanction details

See the [Data Preparation](../set-up/preparing-the-data.md) section for more information about how the data should be prepared.

We set these data paths in a dictionary as follows (replace `<PATH_TO_FILE>` with the path to each file).
```python
# Path to the data files
data_paths = {'complaints': '<PATH_TO_FILE>',
              'complaints_facilities': '<PATH_TO_FILE>',
              'facilities_sanction_id': '<PATH_TO_FILE>',
              'sanctions': '<PATH_TO_FILE>'}
```

## Step 4 - Data Preparation
We are now ready to create the dataframe which we will pass into our Random Forest model. This step is carried out in the `transform_raw_data_to_features()` function, which converts the raw data into various features for classification. There are many steps to this process and it would be very long and cumbersome to explain all the details of this function. To keep the documentation short and readable, we will describe the main steps of the data preparation pipeline which we have used for this model. This will ensure that anyone is able to run our model and have a high-level understanding of the process. If the reader is interested in modifying the pipeline, we refer them to the code, which should be easy to follow and modify, given the explanations provided here.

The `transform_raw_data_to_features()` functions accept two arguments:

- `FEATURE_NAMES`: the names of the features on which to train the model
- `data_paths`: the dictionary of data paths as defined in Step 3

#### List of Feature Transformations
In the next few steps, we will set up the feature transformation steps and prepare the data to be passed into the model. But first, here is a list of the feature transformations we are going to perform for each column

|Column Name(s)|Transformation(s)|
|:---|:---|
|`ComplaintDetail`|TF-IDF<br>LDA Topic Modeling<br>RAKE keywords<br>Number of words|
|`EnvironmentalTopic`|OneHotEncoding|
|`ComplaintType`|OneHotEncoding|
|`FacilityRegion`|OneHotEncoding<br>Natural region|
|`FacilityId`|Binary (facility mentioned or not)|
|`surface_km2`<br>`urban_zones_km2`|Proportion of district covered by urban zones|
|`surface_km2`<br>`protected_areas_km2`|Proportion of district covered by protected areas|
|`FacilityId`<br>`DateComplaint`<br>`Sanctions Data`|Number of past sanctions for the facility|

!!! note Facility Information
    In some cases, the complainant is unable to identify a specific facility. In these situations, the model is able to deal with this by filling in the affected fields with sensible values.

#### Feature Transformation Pipeline
We will now briefly walk through the order of the steps in the feature transformation pipeline. Some of the small intermediary steps have been omitted in the interest of brevity, but these can be easily understood from the code.

##### 1. Read in and pre-process the data
We start by reading in the data from the specified data file paths and normalizing the characters to ascii, in case this has not been done in the data preparation phase.
```python
# Load the data
complaints = pd.read_csv(data_paths['complaints'])
complaints_facilities = pd.read_csv(data_paths['complaints_facilities'])
facilities_sanction_id = pd.read_csv(data_paths['facilities_sanction_id'])
sanctions = pd.read_csv(data_paths['sanctions'])

# Convert the text to ascii
complaints = normalize_ascii(complaints)
complaints_facilities = normalize_ascii(complaints_facilities)
facilities_sanction_id = normalize_ascii(facilities_sanction_id)
sanctions = normalize_ascii(sanctions)
```

##### 2. Create the Target variable
Create the target variable (combining the `EndType` categories of `Formulacion de Cargos` and `Archivo II` into a single `Relevant` category).
```python
# Create the Target variable as a column of the dataframe
df = create_target_variable(complaints_facilities)
```

##### 3. Concatenate the text details
Concatenate the text from all the complaint details into a single complaint text per `ComplaintId` and add this as a column of the dataframe.
```python
# Concatenate the text by complaint
df = concatenate_text(df, complaints)
```

##### 4. One-hot encode the environmental topics
Get the one-hot encoded environmental topics (if `EnvironmentalTopic` is included in the set of features to pass into our Random Forest) for each of the complaints and add these columns to the dataframe.
```python
# One-hot encode the Environmanetal topics of the complaints
if 'EnvironmentalTopic' in FEATURE_NAMES:
    df = environmental_topic(df, complaints)
    print('Environmental Topic Done.')
```

##### 5. One-hot encode the event description data
One-hot encode the event description data and add these columns to the dataframe (if applicable).
```python
# One-hot encode additional description data
if 'Distance_from_event' in FEATURE_NAMES:
    grouped = complaints.groupby('ComplaintId')['Distance_from_event'].apply(set).apply(list)
    grouped_str = grouped.apply(list_to_string)
    df = df.merge(grouped_str.str.get_dummies(), how='left', on='ComplaintId')
if 'Frequency_of_event' in FEATURE_NAMES:
    grouped = complaints.groupby('ComplaintId')['Frequency_of_event'].apply(set).apply(list)
    grouped_str = grouped.apply(list_to_string)
    df = df.merge(grouped_str.str.get_dummies(), how='left', on='ComplaintId')
if 'Day_of_event' in FEATURE_NAMES:
    grouped = complaints.groupby('ComplaintId')['Day_of_event'].apply(set).apply(list)
    grouped_str = grouped.apply(list_to_string)
    df = df.merge(grouped_str.str.get_dummies(), how='left', on='ComplaintId')
if 'Time_of_event' in FEATURE_NAMES:
    grouped = complaints.groupby('ComplaintId')['Time_of_event'].apply(set).apply(list)
    grouped_str = grouped.apply(list_to_string)
    df = df.merge(grouped_str.str.get_dummies(), how='left', on='ComplaintId')
if 'Affected_population' in FEATURE_NAMES:
    grouped = complaints.groupby('ComplaintId')['Affected_population'].apply(set).apply(list)
    grouped_str = grouped.apply(list_to_string)
    df = df.merge(grouped_str.str.get_dummies(), how='left', on='ComplaintId')
if 'Health_Impact' in FEATURE_NAMES:
    grouped = complaints.groupby('ComplaintId')['Health_Impact'].apply(set).apply(list)
    grouped_str = grouped.apply(list_to_string)
    df = df.merge(grouped_str.str.get_dummies(), how='left', on='ComplaintId')
if 'Effect_on_Environment' in FEATURE_NAMES:
    grouped = complaints.groupby('ComplaintId')['Effect_on_Environment'].apply(set).apply(list)
    grouped_str = grouped.apply(list_to_string)
    df = df.merge(grouped_str.str.get_dummies(), how='left', on='ComplaintId')
```

##### 6. Apply feature transformations
Apply the desired feature transformations from the `feature_transformation.py` file and add these as columns of the dataframe.
```python
# Select the feature transformations to include from the feature_transformations.py file
if 'num_words' in FEATURE_NAMES:
    df = num_words(df)
if 'min_num_words' in FEATURE_NAMES:
    df = min_num_words(df)
if 'max_num_words' in FEATURE_NAMES:
    df = max_num_words(df)
if 'num_details' in FEATURE_NAMES:
    df['num_details'] = num_details_all(complaints)
if 'natural_region' in FEATURE_NAMES:
    df = natural_region(df)
if 'populated_districts' in FEATURE_NAMES:
    df = populated_districts(df)
if 'month' in FEATURE_NAMES:
    df = month(df)
if 'quarter' in FEATURE_NAMES:
    df = quarter(df)
if 'weekday' in FEATURE_NAMES:
    df = weekday(df)
if 'facility_mentioned' in FEATURE_NAMES:
    df = facility_mentioned(df)
if 'ComplaintType_archivo1' in FEATURE_NAMES:
    df = ComplaintType_archivo1(df)
if 'proportion_urban' in FEATURE_NAMES:
    df = proportion_urban(df)
if 'proportion_protected' in FEATURE_NAMES:
    df = proportion_protected(df)
if 'num_past_sanctions' in FEATURE_NAMES:
    df['num_past_sanctions'] = num_past_sanctions(df, complaints_facilities, facilities_sanction_id, sanctions)
    df = df[df['Target'].notna()]
if 'total_past_fines' in FEATURE_NAMES:
    df['total_past_fines'] = total_past_fines(df, complaints_facilities, facilities_sanction_id, sanctions)
    df = df[df['Target'].notna()]
```

##### 7. Apply the feature transformations based on geographical data
Apply feature engineering based on the geographical data provided by SMA and add these as columns of the dataframe (if applicable).
```python
# Engineered features based on geographical data provided by SMA
if 'proportion_urban' in FEATURE_NAMES:
    if 'surface_km2' in df.columns and 'surface_km2' not in FEATURE_NAMES:
        columns_to_drop.append('surface_km2')
    if 'urban_zones_km2' in df.columns and 'urban_zones_km2' not in FEATURE_NAMES:
        columns_to_drop.append('urban_zones_km2')
if 'proportion_protected' in FEATURE_NAMES:
    if 'surface_km2' in df.columns and 'surface_km2' not in FEATURE_NAMES:
        columns_to_drop.append('surface_km2')
    if 'protected_areas_km2' in df.columns and 'protected_areas_km2' not in FEATURE_NAMES:
        columns_to_drop.append('protected_areas_km2')
if 'proportion_poor_air' in FEATURE_NAMES:
    if 'surface_km2' in df.columns and 'surface_km2' not in FEATURE_NAMES:
        columns_to_drop.append('surface_km2')
    if 'declared_area_poor_air_quality_km2' in df.columns and 'declared_area_poor_air_quality_km2' not in FEATURE_NAMES:
        columns_to_drop.append('declared_area_poor_air_quality_km2')
```

##### 8. One-hot encode the necessary features
Some of the categorical variables still need to be one-hot encoded (for example the `FacilityRegion` variable). We can do this easily using the `get_dummies()` function from `pandas`, which will automatically encode only the categorical variables of the dataframe. Once we have done the one-hot encoding, we need to save the feature names to a CSV file, as we will need this in the prediction stage, to ensure that the model is being passed all the features it expects (see the corresponding step in the [Relevance Model Prediction](relevance-prediction.md) section for more details on this).
```python
# One-hot encode the necessary columns
X = pd.get_dummies(X)
# Save the column names for use in the prediction script
with open('relevance_column_names.csv', 'w') as f:
      write = csv.writer(f)
      write.writerow(X.columns.to_list())
```

##### 9. Split the data into training and test sets
Now let's split the data into a training set and a test set. In this example, we are using a 70% of the available data for training and the other 30% for testing.
```python
# Split the data to training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=h_params['random_state'])
```

##### 10. Apply the text feature engineering steps
Apply the text feature engineering steps (LDA, TF-IDF and RAKE). It is important that these features are constructed using only on the training set, before being applied to both the training and test sets. We also carry out some pre-processing of the text before apllying these algorithms: stopwords are removed and all words are lemmatized and stemmed. Add the resulting features to the dataframe (if applicable).

We also need to save the top TF-IDF words, fitted RAKE and LDA models, as well as the dictionary for LDA. This is because we will need these things to make our predictions on unseen data, as we will not be re-training our text algorithms on the unseen data. For more details about the predictions on unseen data, see the section on [predictions for the Relevance model](relevance-prediciton.md).
```python
# Preprocess the text in the concat_text field to prepare for the text feature extraction
X_train['concat_text'] = X_train['concat_text'].apply(normalize_text).apply(lemmatize_and_stem_text)
X_test['concat_text'] = X_test['concat_text'].apply(normalize_text).apply(lemmatize_and_stem_text)

# Extract the text features
if 'TF-IDF' in FEATURE_NAMES:
    tfidf_words = top_tfidf_words(X_train, NUMBER_OF_TFIDF_FEATURES)
    X_train = tfidf_word_counts(X_train, tfidf_words)
    X_test = tfidf_word_counts(X_test, tfidf_words)
    # Save the top TF-IDF words for use in the predicition script
    with open('relevance_tfidf_words.csv', 'w') as f:
      write = csv.writer(f)
      write.writerow(tfidf_words.to_list())

if 'RAKE' in FEATURE_NAMES:
    rake_features = DenseRakeFeatures(num_phrases=NUMBER_OF_RAKE_PHRASES)
    # Save the fitted RAKE model as a .pkl file for use in the prediction script
    pickle.dump(rake_features, open('relevance_rake.pkl', 'wb'))
    # Apply the fitted RAKE model to the training and test sets separately
    X_test = X_test.reset_index(drop=True)
    X_train['rake_feature'] = rake_features.fit_transform(X_train['concat_text'])
    X_test['rake_feature'] = rake_features.transform(X_test['concat_text'])

if 'LDA' in FEATURE_NAMES:
    X_train, X_test, lda_model, id2word = perform_topic_modelling(X_train, X_test, NUMBER_OF_LDA_TOPICS)
    # Save the fitted LDA model as a .pkl file for use in the prediction script
    pickle.dump(lda_model, open('relevance_lda_model.pkl', 'wb'))
    # Save the id2word dictionary for use in the prediction script
    id2word.save('relevance_id2word.dict')
```

##### 11. Output the training and test sets
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

1. Once we define the context, we run the experiment as in the lines above. We also grab the experiment `run_id` so that we can use this to make predictions once the model is done training. 
2. We then define a train model step by calling the `TrainModel` class with the model training parameters.
3. Calling the `.train()` method in the `TrainModel` class initializes the training step

!!! note "The `model_type` parameter
    The `model_type` parameter passed into the `TrainModel` class defines what kind of estimator (algorithm) to use for the training step. The model type can be one of `[logistic_regression, svm_classifier, xgboost, random_forest, naive_bayes ]`

##### 3. Use the trained model to predict on the test set
Once the model is done training, we can make predictions on the test set by calling
```python
# make predictions with model using test set
predictions = model.predict(X_test)
```

##### 4. Calculate the proportion of relevant complaints missed
Calculate the proportion of `Relevant` complaints which the model has misclassified into one of the other two categories. This is the most important metric to track, as it is vital to ensure that SMA are not redirecting or archiving too many relevant complaints.
```python
# Calculate the proportion of relevant complaints missed
ind_relevant = np.where(y_test==class_to_numeric(['Relevant']))[0]
pred_relevant = predictions[ind_relevant]
relevant_missed = pred_relevant!=class_to_numeric(['Relevant'])
prop_relevant_missed = sum(relevant_missed) / max(len(ind_relevant), 1)
```

##### 5. Get the evaluation metrics
Use the `ModelMetrics` class to calculate the evaluation metrics.
```python
# get the model metrics
model_metrics = ModelMetrics(
    y_true=actual_labels, y_pred=predicted_labels)
lr_metrics = model_metrics.regression_metrics(
    include_report=True, classification_metric=True)
lr_metrics['proportion_relevant_missed'] = prop_relevant_missed
```

##### 6. Log the parameters and feature information to MLflow
Log the set of features used to create the dataframe, the feature importance returned by the sklearn implementation of the Random Forest and the hyperparameters to MLflow. The feature importance tells us which features contributed more to the classifier than others.
```python
# track feature importance
model_metrics.log_model_features(FEATURE_NAMES)
model_metrics.log_feature_importance(m)
model_metrics.log_hyperparameters_to_mlflow(m._model_params)
print(lr_metrics)
```

##### 7. Log the metrics and the model to MLflow
Log the evaluation metrics to MLflow, as well as the model type (in this case, Random Forest) and the model itself. This will create a .pkl file for the model.
```python
# track the metrics in mlflow
model_metrics.log_metric_to_mlflow(lr_metrics)
# track the model in mlflow
mlflow.sklearn.log_model(model, "model")
mlflow.set_tag("model_type", model_type)
```

##### 8. Save the trained model
Save the trained model as a `.pkl` file for use in the prediction stage later.
```python
# log the model to disk
model_metrics.save_model_to_disk(model, file_path=model_output_path)
```

##### 9. End the MLflow experiment
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
    X_train, X_test, y_train, y_test = transform_raw_data_to_features(FEATURE_NAMES, data=<DATA>)
    
    # Drop the ComplaintId
    X_train.drop(["ComplaintId"], inplace=True, axis=1)
    X_test.drop(["ComplaintId"], inplace=True, axis=1)
    
    experiment_id = run_experiment(X_train, X_test, y_train, y_test, model_type="random_forest", experiment_name="3-class Relevance Model")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", help="Data folder where the data files will be read from")
    parser.add_argument("--output", help="Path where the model pickle file will be saved")
    
    args = parser.parse_args()
    main(args.data, args.output)
```
In the code above, replace the `<DATA>` argument of the `transform_raw_data_to_features` function with the preprocessed data in the form required.

## Step 7 - Run the entire script
To execute the run, simply run the script from the command line with the following python command. Make sure that you are in the correct directory. Replace the `<PATH_TO_DATA_DIRECTORY>` with the path to the directory where the data files are located and the `<PATH_TO_SAVE_MODEL_TO>` to the directory where the model pickle file should be saved.
```shell
$ python relevance_test_train.py --data=<PATH_TO_DATA_DIRECTORY> --output=<PATH_TO_SAVE_MODEL_TO>
```

## Step 8 - View the results in MLflow
All the logged information is now available to view in MLflow. Run the following command in the command line to launch the MLflow dashboard, then open a browser window and paste the web address indicated.
```shell
$ mlflow ui
```

# Conclusion
In this walkthrough, we saw how to create and run the relevance model. The results from the model are logged to Mlflow via custom classes. The predictions from the model will then be logged to the database. In the next section, we will discuss the details of the feature engineering, providing details on the various feature transformations we used and considered. The [Model Deployment](../model-deployment/deployment-environment-setup.md) section will provide details of how to deploy the model and use the model in production.
