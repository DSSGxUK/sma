# Define the experiment features
exp_features = ['ComplaintType','FacilityRegion','FacilityEconomicSubSector','FacilityEconomicSector','EnvironmentalTopic', 'natural_region', 'facilities_num_sanctions', 'num_words', 'facilities_num_sanctions', 'facilities_num_inspections','sanction_inspection_ratio', 'month', 'quarter', 'LDA','TF-IDF']

# ['ComplaintType','EnvironmentalTopic', 'facilities_num_sanctions', 'facilities_num_inspections',
#                 'FacilityRegion','FacilityEconomicSector','FacilityEconomicSubSector','sanction_inspection_ratio',
#                 'month'] # Best run

columns_to_drop = ['ComplaintStatus', 'ComplaintType', 'DateComplaint',
                   'DateResponse', 'EndType', 'DigitalComplaintId', 'PreClasification',
                   'Clasification', 'FacilityId', 'FacilityRegion', 'FacilityDistrict',
                   'FacilityEconomicSector', 'FacilityEconomicSubSector', 'concat_text']

if ('month' in exp_features) or ('quarter' in exp_features):
    columns_to_drop.remove('DateComplaint')
for col in columns_to_drop:
    if col in exp_features:
        columns_to_drop.remove(col)


import os
import sys
import pandas as pd
import numpy as np
from collections import Counter

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix,classification_report

helpers_dir = (os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) + '/helpers/')
sys.path.append(helpers_dir)
sma_dir = (os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')) + '/sma/')
sys.path.append(sma_dir)
sma_project_dir = (os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')) + '/sma-project/')
sys.path.append(sma_project_dir)
from feature_extraction import concat_complaint_details, pivot_env_topics, \
                                num_words_all, num_details_all
from feature_transformation import *
# from tfidf_pipeline import DenseTFIDFVectorization

# -- Collins example code
from feature_union import FeatureUnification
from model_metrics import ModelMetrics
# from rake_extraction import DenseRakeFeatures
from tfidf_transform import DenseTFIDFVectorization
from train_model import TrainModel
from topic_models import TopicScores
from sklearn.preprocessing import OneHotEncoder
import mlflow



# Functions

def class_to_numeric(classes):
    class_to_num = {'Relevant': 0, 'Irrelevant': 1}
    return [class_to_num[c] for c in classes]
    
def numeric_to_class(numbers):
    num_to_class = {0: 'Relevant', 1: 'Irrelevant'}
    return [num_to_class[n] for n in numbers]

def binary_relevance(endtype):
    if isinstance(endtype, float):
        return endtype
    elif endtype in ['Archivo II','Formulación de Cargos']:
        return 'Relevant'
    else:
        return 'Irrelevant'
    
    
# Load the data
complaints = pd.read_csv('../../../../files/data_merge/complaint_registry.csv')
complaints_facilities = pd.read_csv('../../../../files/data_merge/complaints_facilities_registry.csv')
## complaints_sanctions = pd.read_csv('../../../../files/data_merge/complaints_sanctions_registry.csv')
facilities_sanction_id = pd.read_csv('../../../../files/data_merge/facilities_sanction_id.csv')
facilities_inspection_id = pd.read_csv('../../../../files/data_merge/facilities_inspection_id.csv')
complaints_inspections = pd.read_csv('../../../../files/data_merge/complaints_inspections_registry.csv')

# Create the target variable
complaints_facilities['Target'] = complaints_facilities['EndType'].apply(binary_relevance)
df = complaints_facilities.copy()
concat_text = concat_complaint_details(complaints) # Get the concatenated details text
def get_concat_text(complaint_id):
    if complaint_id in concat_text.keys():
        return concat_text[complaint_id]
    else:
        return ' '
df['concat_text'] = df['ComplaintId'].apply(get_concat_text)
# df = df.join(concat_text, how='inner')
# df.rename(columns=({'ComplaintDetail': 'concat_text'}), inplace=True)
# print(df.columns)
df = df[df['Target'].notna()]
df['Target'] = class_to_numeric(df['Target'].values.tolist())

# Set the list of features to unify
features_to_unify = []

# -- Collins example code : features (I added the if statements)
# initialize pipeline steps
X = df['concat_text'].fillna(' ')
if 'TF-IDF' in exp_features:
    dense_tfidf = DenseTFIDFVectorization(ngram_range=(2, 2), max_features=25)
    # fit data to pipeline
    dtidf = dense_tfidf.fit_transform(X)
    features_to_unify.append(dtidf)
# if 'EnvironmentalTopic' in exp_features:
#     env_topics = complaints["EnvironmentalTopic"].values.reshape(-1,1)
#     onehot_encoder = OneHotEncoder()
#     encoder_features = onehot_encoder.fit_transform(env_topics)
#     encoder_features_df = pd.DataFrame(encoder_features.toarray(), columns=onehot_encoder.get_feature_names())
#     features_to_unify.append(encoder_features_df)
if 'LDA' in exp_features:
    topic_scores = TopicScores(data=X, num_topics = 11)
    topic_scores = topic_scores.drop(columns='row_number')
    print(topic_scores)
    features_to_unify.append(topic_scores)
# if 'RAKE' in exp_features:
    # rake_features = DenseRakeFeatures()
    # rakefeatures = rake_features.fit_transform(X)

# Extract the text features from complaints_registry (if required)
# df['concat_text'] = concat_complaint_details(complaints) # always include this feature
if 'EnvironmentalTopic' in exp_features:
    pivot_topics = pivot_env_topics(complaints)
    df = df.join(pivot_topics, on='ComplaintId', rsuffix='_r', how='left')
if 'num_details' in exp_features:
    df['num_details'] = num_details_all(complaints)

# Select the feature transformations to include from the feature_transformations.py file
if 'min_num_words' in exp_features:
    df = min_num_words(df)
if 'max_num_words' in exp_features:
    df = max_num_words(df)
if 'natural_region' in exp_features:
    df = natural_region(df)
if 'num_words' in exp_features:
    df = num_words(df)
if 'facility_num_sanctions' in exp_features:
    # df = facility_num_sanctions(df, complaints_facilities, facilities_sanction_id)
    if 'FacilityId' in columns_to_drop:
        columns_to_drop.remove('FacilityId')
if 'facility_has_sanctions' in exp_features:
    # df = facility_has_sanctions(df, complaints_facilities, facilities_inspection_id)
    if 'FacilityId' in columns_to_drop:
        columns_to_drop.remove('FacilityId')
if 'facility_num_inspections' in exp_features:
    # df = facility_num_inspections(df, complaints_facilities, facilities_inspection_id)
    if 'FacilityId' in columns_to_drop:
        columns_to_drop.remove('FacilityId')
if 'facility_has_inspections' in exp_features:
    # df = facility_has_inspections(df, complaints_facilities, facilities_inspection_id)
    if 'FacilityId' in columns_to_drop:
        columns_to_drop.remove('FacilityId')
if 'has_many_inspections' in exp_features:
    # df = has_many_inspections(df, complaints_facilities, facilities_inspection_id)
    if 'FacilityId' in columns_to_drop:
        columns_to_drop.remove('FacilityId')
if 'sanction_inspection_ratio' in exp_features:
    # df = sanction_inspection_ratio(df,complaints_facilities,facilities_inspection_id,facilities_sanction_id)
    if 'FacilityId' in columns_to_drop:
        columns_to_drop.remove('FacilityId')
if 'populated_districts' in exp_features:
    df = populated_districts(df)
if 'month' in exp_features:
    df = month(df)
if 'quarter' in exp_features:
    df = quarter(df)

# Remove the DateComplaint field from the features if it is still there
if 'DateComplaint' in df.columns:
    df = df.drop(columns=['DateComplaint'])
    if 'DateComplaint' in columns_to_drop:
        columns_to_drop.remove('DateComplaint')
        

df = df.reset_index()
df = df.drop(columns=['index'])
df = df.drop(columns=columns_to_drop)
print(df.columns)

features_to_unify_temp = features_to_unify.copy()
features_to_unify_temp.append(df)
df = FeatureUnification().unify(features_to_unify_temp)

# Define the inputs and target
X = df.drop(columns=['ComplaintId', 'Target'])
y = df['Target']

# One-hot-encode the categorical features
# X = pd.get_dummies(X, columns=one_hot_required)
X = pd.get_dummies(X)
# Convert the target into the right shape
y = y.values.reshape(len(y),)

print('X.COLUMNS', X.columns)

# Add y back to X, to do FeatureUnification together
X['Target'] = y

# The final df to unify
features_to_unify.append(X)
print([f.shape for f in features_to_unify])

#-- Collins example code

# Define the inputs and target
y = X['Target'].values
X = X.drop(columns=['Target'])

# Split the data to training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=17)

# Apply the feature transformations which depend on the train-test split
if 'facility_num_sanctions' in exp_features:
    X_train, X_test = facility_num_sanctions(X_train, X_test, complaints_facilities, facilities_sanction_id)
if 'facility_has_sanctions' in exp_features:
    X_train, X_test = facility_has_sanctions(X_train, X_test, complaints_facilities, facilities_inspection_id)
if 'facility_num_inspections' in exp_features:
    X_train, X_test = facility_num_inspections(X_train, X_test, complaints_facilities, facilities_inspection_id)
if 'facility_has_inspections' in exp_features:
    X_train, X_test = facility_has_inspections(X_train, X_test, complaints_facilities, facilities_inspection_id)
if 'has_many_inspections' in exp_features:
    X_train, X_test = has_many_inspections(X_train, X_test, complaints_facilities, facilities_inspection_id)
print(X_test.iloc[:5,:])
if 'sanction_inspection_ratio' in exp_features:
    X_train, X_test = sanction_inspection_ratio(X_train, X_test,complaints_facilities,facilities_inspection_id,facilities_sanction_id)
    if 'facility_num_sanctions' not in exp_features:
        print('DROP sanctions')
        X_train = X_train.drop(columns=['facility_num_sanctions'])
        X_test = X_test.drop(columns=['facility_num_sanctions'])
    if 'facility_num_inspections' not in exp_features:
        print('DROP inspections')
        X_train = X_train.drop(columns=['facility_num_inspections'])
        X_test = X_test.drop(columns=['facility_num_inspections'])
print(X_test.iloc[:5,:])

if 'FacilityId' in X_train.columns:
    X_train = X_train.drop(columns=['FacilityId'])
if 'FacilityId' in X_test.columns:
    X_test = X_test.drop(columns=['FacilityId'])

print('X.COLUMNS', X_train.columns)

# -- Collins example code
mlflow.set_experiment("Binary Relevance Model")
with mlflow.start_run(nested=True):
    m_type = 'rf'
    m = TrainModel(X_train=X_train, y_train=y_train, model_type=m_type)
    model = m.train()

    # track model metrics and params
    predictions = model.predict(X_test)
    # print(predictions)


    ##### MY CODE START #####

    # Make predictions on the test set
    y_pred = predictions

    # Find indices of relevant complaints
    ind_relevant = np.where(y_test==class_to_numeric(['Relevant']))[0]
    pred_relevant = y_pred[ind_relevant]
    relevant_missed = pred_relevant!=class_to_numeric(['Relevant'])
    prop_relevant_missed = sum(relevant_missed) / len(ind_relevant)

    # Other metrics
    classif_report = classification_report(y_test, y_pred, output_dict=True, 
                                        target_names=numeric_to_class([0,1]))
    precisions = (round(classif_report['Relevant']['precision'], 5), 
                  round(classif_report['Irrelevant']['precision'], 5))
    recalls = (round(classif_report['Relevant']['recall'], 5), 
               round(classif_report['Irrelevant']['recall'], 5))
    accuracy = model.score(X_test, y_test)
    confusion_matrix = confusion_matrix(y_test, y_pred)
    feat_importance = model.feature_importances_
    feature_importance = sorted(zip(feat_importance, X_test.columns), reverse=True)
    
    if m_type == 'rf':
        algorithm = 'Random Forest'
    elif m_type == 'svm':
        algorithm = 'SVM'
    elif m_type == 'lr':
        algorithm = 'Logistic Regression'
    elif m_type == 'nb':
        algorithm = 'Naive Bayes'
    else:
        algorithm = m_type

    # Save the metrics
    metrics_df = pd.read_csv('metrics_binary_new.csv')
    #metrics_df = pd.DataFrame()
    new_row = pd.DataFrame({'ExperimentId': [len(metrics_df)],
                            'Proportion of Relevant complaints missed': [prop_relevant_missed],
                            'Accuracy': [accuracy],
                            'Precision (R, NR)': [precisions], 
                            'Recall (R,NR)': [recalls], 
                            'Features': [exp_features],
                            '5 most important features': [feature_importance[:5]],
                            'Algorithm': [algorithm]})
    print(new_row)
    metrics_df = metrics_df.append([new_row])
    print(metrics_df)
    if 'Unnamed: 0' in metrics_df.columns:
        metrics_df.drop(columns='Unnamed: 0')
    # metrics_df.iloc[len(metrics_df),:] = [len(metrics_df), len(metrics_df), prop_relevant_missed, accuracy, precisions,
    #                                       recalls, exp_features]
    metrics_df.to_csv('metrics_binary_new.csv', index=False)

    print('Feature Importance:')
    print(feature_importance[:25])
    print()
    print(accuracy)
    print(classification_report(y_test, y_pred,
                                target_names=numeric_to_class([0,1])))
    print('\nProportion of Relevant complaints missed:', prop_relevant_missed)
    print('\nConfusion matrix:')
    print(confusion_matrix)

    ##### MY CODE END #####


    model_metrics = ModelMetrics(y_true=y_test, y_pred=predictions)
    lr_metrics = model_metrics.regression_metrics(include_report=True, classification_metric=True)
    lr_metrics['proportion_relevant_missed'] = prop_relevant_missed
    print(lr_metrics)
    model_metrics.log_metric_to_mlflow(lr_metrics)


