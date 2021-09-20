# Define the experiment features
# exp_features = ['ComplaintType', 'num_words', 'EnvironmentalTopic', 'FacilityRegion', 
#                 'FacilityEconomicSubSector', 'month', 'sanction_inspection_ratio', 
#                 'facility_num_inspections', 'facility_num_sanctions', 'LDA', 'RAKE', 
#                 'facility_mentioned','TF-IDF'] # Old Best for RF
# exp_features = ['EnvironmentalTopic','sanction_inspection_ratio','FacilityRegion','LDA','RAKE',
#                 'FacilityEconomicSector','quarter','facility_mentioned'] # Best for SVM
# exp_features = ['ComplaintType','ComplaintType_archivo1','num_words']
# exp_features = ['ComplaintType', 'num_words', 'Effect_on_Environment']
# exp_features = ['ComplaintType', 'num_words', 'FacilityRegion', 'FacilityEconomicSubSector',
#                 'month', 'LDA', 'RAKE', 'TF-IDF', 'facility_mentioned', 
#                 'proportion_urban', 'proportion_protected', 'EnvironmentalTopic',
#                 'num_past_sanctions'] # Best run (no PCA)
exp_features = [
                'LDA', 'RAKE', 'TF-IDF', 
                'num_words',
                # 'ComplaintType',  
                'FacilityRegion', 
                'natural_region',
                # 'FacilityEconomicSector',
                'facility_mentioned', 
                'proportion_urban', 
                'proportion_protected', 
                # 'proportion_poor_air',
                'EnvironmentalTopic', 
                # 'month', 
                # 'weekday',
                'num_past_sanctions',
                # 'total_past_fines'
                # 'Effect_on_Environment', 
                # 'Health_Impact', 
                # 'Affected_population',
                # 'Distance_from_event', 'Time_of_event',  
                # 'Frequency_of_event', 'Day_of_event', 
                ]
# exp_features = ['LDA', 'TF-IDF', 'RAKE']
# exp_features = ['RAKE']
return_goodness_fit = False


m_type = 'rf'
tfidf_num=100; lda_num=50; rake_num=15000
pca_components = 8
apply_pca = False

# Model parameters
h_params = {'max_depth': 7, 'max_features': 20} #Probable best params for RF
# h_params = {'max_depth': 3, 'max_features': 10} #Best params for GB
# h_params = {'max_depth': 3, 'max_features': 10, 'min_samples_leaf':3}
# h_params = {'max_depth': 3, 'max_features': 10}
h_params = {}

random_state = 42
h_params['random_state'] = random_state
print(h_params)

# Hyperparameters to test
import itertools
# First run
tuning_params = [[2,6,10], # max_depth
                 [2,20,40,70,100], # max_features
                 [2,20,50,100,150], # n_estimators (number of trees)
                 [2,10,20,40], # max_leaf_nodes
                ]
# Second run
tuning_params = [[3,6,9], # max_depth
                 [20,40,70,100], # max_features
                 [20,50,100,150], # n_estimators (number of trees)
                 [20,40], # max_leaf_nodes
                ]
# Third run
tuning_params = [[7,8,9,10], # max_depth
                 [20,40,70,100], # max_features
                 [20,50,100,150], # n_estimators (number of trees)
                 [30,40,50], # max_leaf_nodes
                ]
# Fourth Run
tuning_params = [[7,8,9,10], # max_depth
                 [20,40,70], # max_features
                 [20,50,70,100], # n_estimators (number of trees)
                 [30,40,50], # max_leaf_nodes
                ]
# Fifth Run
tuning_params = [[7,8,9,10], # max_depth
                 [20,35,60], # max_features
                 [20,50,70,100], # n_estimators (number of trees)
                 [35,40,45], # max_leaf_nodes
                ]
# Sixth Run
tuning_params = [[7,8,9,10], # max_depth
                 [20,30,40], # max_features
                 [20,50,100], # n_estimators (number of trees)
                 [36,40,44], # max_leaf_nodes
                ]
# Seventh Run
tuning_params = [[7,8,9,10], # max_depth
                 [15,20,25,30], # max_features
                 [20,50,100], # n_estimators (number of trees)
                 [36,39,42], # max_leaf_nodes
                ]
# Eighth Run
tuning_params = [[7,8], # max_depth
                 [15,20,25], # max_features
                 [50,80,100], # n_estimators (number of trees)
                 [36,39,42], # max_leaf_nodes
                ]
# Ninth Run
tuning_params = [[7], # max_depth
                 [15,20], # max_features
                 [50,55,60,65,70], # n_estimators (number of trees)
                 [39,40,41,42], # max_leaf_nodes
                ]
# Final Parameters
tuning_params = [[7], # max_depth
                 [20], # max_features
                 [70], # n_estimators (number of trees)
                 [42], # max_leaf_nodes
                ]
# Min_samples_split
use_min_samples_split = False
if use_min_samples_split == True:
    tuning_params = [[7], # max_depth
                    [20], # max_features
                    [70], # n_estimators (number of trees)
                    [42], # max_leaf_nodes
                    [2,3,4,5,6,7] # min_samples_split
                    ]
# Min_samples_leaf
use_min_samples_leaf = False
if use_min_samples_leaf == True:
    tuning_params = [[7], # max_depth
                    [20], # max_features
                    [70], # n_estimators (number of trees)
                    [42], # max_leaf_nodes
                    [1,2,3,4,5,6,7] # min_samples_leaf
                    ]
# tuning_params = [[None], # max_depth
#                  ['auto'], # max_features
#                  [100], # n_estimators (number of trees)
#                  [None], # max_leaf_nodes
#                 ]
tuning_hyperparams = list(itertools.product(*tuning_params))
hyperparameter_tuning = True
mlflow_exp_name = "Relevance Model"


# Run information
compute_text_features = False


numeric_cols = ['rake_feature', 'num_words', 'num_details', 'min_num_words', 'max_num_words', 
                'month', 'quarter', 'proportion_urban', 'proportion_protected', 'proportion_poor_air', 
                'income_poverty_percentage','num_past_sanctions']

columns_to_drop = ['ComplaintStatus', 'ComplaintType', 'DateComplaint',
                   'DateResponse', 'EndType', 'DigitalComplaintId', 'PreClasification',
                   'Clasification', 'FacilityId', 'FacilityRegion', 'FacilityDistrict',
                   'FacilityEconomicSector', 'FacilityEconomicSubSector', 'concat_text',
                    'District', 'CutId', 'income_poverty_percentage', 
                    'multidimensional_poverty_percentage', 'surface_km2', 'tot_population',
                    'urban_zones_km2', 'protected_areas_km2', 'urban_pop_0_5',
                    'urban_pop_6_14', 'urban_pop_15_64', 'urban_pop_M65', 'rural_pop_0_5',
                    'rural_pop_6_14', 'rural_pop_15_64', 'rural_pop_M65',
                    'declared_area_poor_air_quality_km2', 'Unnamed: 0', 'Unnamed: 0.1',
                #    'Effect_on_Environment'
                   ]
if ('month' in exp_features) or ('quarter' in exp_features):
    columns_to_drop.remove('DateComplaint')
if 'proportion_urban' in exp_features:
    columns_to_drop.remove('urban_zones_km2')
    columns_to_drop.remove('surface_km2')
if 'proportion_protected' in exp_features:
    columns_to_drop.remove('protected_areas_km2')
    if 'surface_km2' in columns_to_drop:
        columns_to_drop.remove('surface_km2')
if 'proportion_poor_air' in exp_features:
    columns_to_drop.remove('declared_area_poor_air_quality_km2')
    if 'surface_km2' in columns_to_drop:
        columns_to_drop.remove('surface_km2')
for col in columns_to_drop:
    if col in exp_features:
        columns_to_drop.remove(col)


import os
import sys
import pandas as pd
import numpy as np
from unidecode import unidecode
from collections import Counter
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler, SMOTE
import itertools

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

helpers_dir = (os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) + '/helpers/')
sys.path.append(helpers_dir)
sma_dir = (os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')) + '/sma/')
sys.path.append(sma_dir)
sma_project_dir = (os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')) + '/sma-project/')
sys.path.append(sma_project_dir)
from feature_extraction import concat_complaint_details, pivot_env_topics, \
                                num_words_all, num_details_all
from feature_transformation import *
from parse_csv import normalize_ascii
# from tfidf_pipeline import DenseTFIDFVectorization

from feature_union import FeatureUnification
from model_metrics import ModelMetrics
from rake_extraction import DenseRakeFeatures
from tfidf_transform import DenseTFIDFVectorization
from feature_selection import FeatureSelection
from train_model import TrainModel
from topic_models import TopicScores
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
import mlflow

# Set the seed
np.random.seed(0)

# Functions

def three_class_target(endtype):
    # Check if the EndType is NaN
    if isinstance(endtype, float):
        return endtype
    elif endtype in ['Archivo II','Formulación de Cargos','Formulacion de Cargos']:
        return 'Relevant'
    else:
        return endtype

def class_to_numeric(classes):
    class_to_num = {'Archivo I': 0, 'Derivacion Total a Organismo Competente': 1, 
                    'Derivación Total a Organismo Competente': 1, 'Relevant': 2}
    return [class_to_num[c] for c in classes]
    
def numeric_to_class(numbers):
    num_to_class = {0: 'Archivo I', 1: 'Derivacion', 2: 'Relevant'}
    return [num_to_class[n] for n in numbers]

def balanced_sampling(X, y, sampling_type, strategy = 'default', target_col: str = 'Target'):
    """
    Performs undersampling or oversampling according to the specified strategy.
    Allowed samplers are undersampling or oversampling.
    Returns the specified number of randomly-sampled data points for each class.
    """
    ascending_counts = sorted(Counter(y).items(), key = lambda tup: tup[1])

    if sampling_type == 'oversample':
        if strategy == 'default':
            # Oversample the minimum class to the middle-sized class
            strategy = {ascending_counts[0][0]: ascending_counts[1][1],
                        ascending_counts[1][0]: ascending_counts[1][1],
                        ascending_counts[2][0]: ascending_counts[2][1]}
        ros = RandomOverSampler(sampling_strategy=strategy, random_state=random_state)
        X_resampled, y_resampled = ros.fit_resample(X, y)
        
    elif sampling_type == 'undersample':
        if strategy == 'default':
            # Undersample the maximum class to the middle-sized class
            strategy = {ascending_counts[0][0]: ascending_counts[0][1],
                        ascending_counts[1][0]: ascending_counts[1][1],
                        ascending_counts[2][0]: ascending_counts[1][1]}
        rus = RandomUnderSampler(sampling_strategy=strategy, random_state=random_state)
        X_resampled, y_resampled = rus.fit_resample(X, y)

    elif sampling_type == 'smote':
        strategy = {ascending_counts[0][0]: ascending_counts[1][1],
                        ascending_counts[1][0]: ascending_counts[1][1],
                        ascending_counts[2][0]: ascending_counts[2][1]}
        sm = SMOTE(sampling_strategy=strategy, random_state=random_state)
        X_resampled, y_resampled = sm.fit_resample(X, y)
    else:
        print('Input error: sampling_type must be one of [oversample,undersample,smote]')
        
    return X_resampled, y_resampled
    
    
# Load the data
complaints = pd.read_csv('../../../../files/data_merge/complaint_registry.csv')
complaints_facilities = pd.read_csv('../../../../files/data_merge/complaints_facilities_registry.csv')
## complaints_sanctions = pd.read_csv('../../../../files/data_merge/complaints_sanctions_registry.csv')
facilities_sanction_id = pd.read_csv('../../../../files/data_merge/facilities_sanction_id.csv')
facilities_inspection_id = pd.read_csv('../../../../files/data_merge/facilities_inspection_id.csv')
complaints_inspections = pd.read_csv('../../../../files/data_merge/complaints_inspections_registry.csv')
sanctions = pd.read_csv('../../../../files/data_merge/sanctions_registry.csv')

# variables_territoriales = pd.read_excel('../../../../files/20210729_SMAData.xlsx',
#                                 sheet_name='Variables_territoriales')
# complaints_facilities = complaints_facilities.merge(variables_territoriales, left_on='FacilityDistrict', 
#                                                     right_on='District', how='left')

# print(complaints_facilities.shape)
# complaints_facilities = complaints_facilities[complaints_facilities['FacilityId'].notna()==False]
# complaints_facilities = complaints_facilities.iloc[8200:,:]
# print(complaints_facilities.shape)
# print(complaints_facilities['FacilityId'].value_counts(dropna=False))

complaints = normalize_ascii(complaints)
complaints_facilities = normalize_ascii(complaints_facilities)
facilities_sanction_id = normalize_ascii(facilities_sanction_id)
facilities_inspection_id = normalize_ascii(facilities_inspection_id)
complaints_inspections = normalize_ascii(complaints_inspections)
sanctions = normalize_ascii(sanctions)

print(complaints.iloc[:5,:])

# Need to groupby for the geographical indicators
print(complaints_facilities.columns)

# Create the target variable
complaints_facilities['Target'] = complaints_facilities['EndType'].apply(three_class_target)
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
print("df['Target']",df['Target'])

print(df['Target'].value_counts(dropna=False))
# print(df[df['FacilityId'].isna()]['Target'].value_counts(dropna=False))
# print(df.columns)

# Set the list of features to unify
features_to_unify = []

# initialize pipeline steps
X = df['concat_text'].fillna(' ')


if compute_text_features == True:
    print('TEXT PROCESSING')
    print(X[:5])
    print(X.dtype)
    X = X.apply(normalize_text).apply(lemmatize_and_stem_text)
    print(X[:5])

    if 'TF-IDF' in exp_features:
        dense_tfidf = DenseTFIDFVectorization(ngram_range=(1, 3), max_features=tfidf_num)
        # dtidf_archivo = dense_tfidf.fit_transform(X_archivo)
        # dtidf_derivacion = dense_tfidf.fit_transform(X_derivacion)
        # dtidf_relevant = dense_tfidf.fit_transform(X_relevant)
        # dtidf = pd.concat([dtidf_archivo, dtidf_derivacion, dtidf_relevant], axis=1)
        # print(dtidf.columns)
        dtidf = dense_tfidf.fit_transform(X)
        features_to_unify.append(dtidf)
    if 'LDA' in exp_features:
        topic_scores = TopicScores(data=X, num_topics = lda_num)
        topic_scores = topic_scores.drop(columns='row_number')
        # Change the topic model column names to strings
        for topic in topic_scores.columns:
            topic_scores = topic_scores.rename(columns={topic:'LDA Topic ' + str(topic)[:-2]})
        # topic_scores = topic_scores.fillna(0)
        features_to_unify.append(topic_scores)
        print('LDA done.')
    if 'RAKE' in exp_features:
        rake_features = DenseRakeFeatures(num_phrases=rake_num)
        rakefeatures = rake_features.fit_transform(X)
        features_to_unify.append(rakefeatures)
        print('RAKE done.')
    # text_df = FeatureUnification().unify(features_to_unify)
    # text_df.to_csv('text_features.csv', index=True)
else:
    if 'LDA' in exp_features:
        print('START:')
        text_df = pd.read_csv('text_features.csv')
        print(text_df.iloc[:2,:])
        print(text_df.columns)
        text_df = text_df.drop(columns=['Unnamed: 0'])
        print(text_df.iloc[:2,:])
        features_to_unify.append(text_df)


if 'EnvironmentalTopic' in exp_features:
    # print('Environmental Topic:')
    # print(df.shape)
    # print(complaints.shape)
    env_topics = complaints["EnvironmentalTopic"].values.reshape(-1,1)
    onehot_encoder = OneHotEncoder()
    encoder_features = onehot_encoder.fit_transform(env_topics)
    encoder_features_df = pd.DataFrame(encoder_features.toarray(), columns=onehot_encoder.get_feature_names())
    encoder_features_df['ComplaintId'] = complaints['ComplaintId']
    encoder_features_df = encoder_features_df.groupby('ComplaintId').apply(sum)
    encoder_features_df = encoder_features_df.drop(columns=['ComplaintId'])
    for col in encoder_features_df.columns:
        encoder_features_df[col][encoder_features_df[col] > 1] = 1
        encoder_features_df = encoder_features_df.rename(columns={col: 'EnvTopic' + col[2:]})
    # print(encoder_features_df.max(axis=0))
    df = df.join(encoder_features_df, on='ComplaintId', rsuffix='_r', how='left')
    # print(df.shape)
    # print(encoder_features_df.iloc[:5,:])
    # features_to_unify.append(encoder_features_df)
# if 'EnvironmentalTopic' in exp_features:
#     pivot_topics = pivot_env_topics(complaints)
#     df = df.join(pivot_topics, on='ComplaintId', rsuffix='_r', how='left')
if 'num_details' in exp_features:
    df['num_details'] = num_details_all(complaints)

# New data from the complaints registry
def list_to_string(row):
    row_str = []
    for list_item in row:
        if isinstance(list_item, float):
            list_item = str(list_item)
        row_str.append(list_item)
    row = '|'.join(row_str)
    return row
if 'Distance_from_event' in exp_features:
    grouped = complaints.groupby('ComplaintId')['Distance_from_event'].apply(set).apply(list)
    grouped_str = grouped.apply(list_to_string)
    df = df.merge(grouped_str.str.get_dummies(), how='left', on='ComplaintId')
if 'Frequency_of_event' in exp_features:
    grouped = complaints.groupby('ComplaintId')['Frequency_of_event'].apply(set).apply(list)
    grouped_str = grouped.apply(list_to_string)
    df = df.merge(grouped_str.str.get_dummies(), how='left', on='ComplaintId')
if 'Day_of_event' in exp_features:
    grouped = complaints.groupby('ComplaintId')['Day_of_event'].apply(set).apply(list)
    grouped_str = grouped.apply(list_to_string)
    df = df.merge(grouped_str.str.get_dummies(), how='left', on='ComplaintId')
if 'Time_of_event' in exp_features:
    grouped = complaints.groupby('ComplaintId')['Time_of_event'].apply(set).apply(list)
    grouped_str = grouped.apply(list_to_string)
    df = df.merge(grouped_str.str.get_dummies(), how='left', on='ComplaintId')
if 'Affected_population' in exp_features:
    grouped = complaints.groupby('ComplaintId')['Affected_population'].apply(set).apply(list)
    grouped_str = grouped.apply(list_to_string)
    df = df.merge(grouped_str.str.get_dummies(), how='left', on='ComplaintId')
if 'Health_Impact' in exp_features:
    grouped = complaints.groupby('ComplaintId')['Health_Impact'].apply(set).apply(list)
    grouped_str = grouped.apply(list_to_string)
    df = df.merge(grouped_str.str.get_dummies(), how='left', on='ComplaintId')
if 'Effect_on_Environment' in exp_features:
    grouped = complaints.groupby('ComplaintId')['Effect_on_Environment'].apply(set).apply(list)
    grouped_str = grouped.apply(list_to_string)
    df = df.merge(grouped_str.str.get_dummies(), how='left', on='ComplaintId')

### For the TEST set without missing facilities
# if 'FacilityId' in columns_to_drop:
#         columns_to_drop.remove('FacilityId')
### 

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
if 'weekday' in exp_features:
    df = weekday(df)
if 'facility_mentioned' in exp_features:
    df = facility_mentioned(df)
if 'ComplaintType_archivo1' in exp_features:
    df = ComplaintType_archivo1(df)
if 'proportion_urban' in exp_features:
    df = proportion_urban(df)
if 'proportion_protected' in exp_features:
    df = proportion_protected(df)
if 'num_past_sanctions' in exp_features:
    df['num_past_sanctions'] = num_past_sanctions(df, complaints_facilities, facilities_sanction_id, sanctions)
    df = df[df['Target'].notna()]
    # df['Target'] = class_to_numeric(df['Target'].tolist())
if 'total_past_fines' in exp_features:
    print('BEFORE:', df.columns)
    df['total_past_fines'] = total_past_fines(df, complaints_facilities, facilities_sanction_id, sanctions)
    print('AFTER:', df.columns)
    df = df[df['Target'].notna()]
    # df['Target'] = class_to_numeric(df['Target'].tolist())


# Remove the DateComplaint field from the features if it is still there
if 'DateComplaint' in df.columns:
    df = df.drop(columns=['DateComplaint'])
    if 'DateComplaint' in columns_to_drop:
        columns_to_drop.remove('DateComplaint')

if 'proportion_urban' in exp_features:
    if 'surface_km2' in df.columns and 'surface_km2' not in exp_features:
        columns_to_drop.append('surface_km2')
    if 'urban_zones_km2' in df.columns and 'urban_zones_km2' not in exp_features:
        columns_to_drop.append('urban_zones_km2')
if 'proportion_protected' in exp_features:
    if 'surface_km2' in df.columns and 'surface_km2' not in exp_features:
        columns_to_drop.append('surface_km2')
    if 'protected_areas_km2' in df.columns and 'protected_areas_km2' not in exp_features:
        columns_to_drop.append('protected_areas_km2')
if 'proportion_poor_air' in exp_features:
    if 'surface_km2' in df.columns and 'surface_km2' not in exp_features:
        columns_to_drop.append('surface_km2')
    if 'declared_area_poor_air_quality_km2' in df.columns and 'declared_area_poor_air_quality_km2' not in exp_features:
        columns_to_drop.append('declared_area_poor_air_quality_km2')

    

df = df.reset_index()
df = df.drop(columns=['index'])
# print(df.columns)
# print(columns_to_drop)
for col in columns_to_drop:
    if col in df.columns:
        df = df.drop(columns=col)
# print(df.columns)
# print('df Target: ', df['Target'])

# Drop any rows where there are NaNs (the NaNs are introduced by the rows which are
# present in the complaints_details but not in the complaints_facilities)
# print('--->', df.columns.tolist())
# columns_to_drop_na = df.columns.tolist()
# columns_to_drop_na.remove('ComplaintId')
# columns_to_drop_na.remove('Target')
features_to_unify_temp = features_to_unify.copy()
features_to_unify_temp.append(df)
# print([f.shape for f in features_to_unify_temp])
df = FeatureUnification().unify(features_to_unify_temp)

# print(columns_to_drop_na)
# for c in columns_to_drop_na:
#     if df[c].isna().sum() != 0:
#         df = df[df[c].notna()]

# Drop rows where the Target is NaN
df = df[df['Target'].isna()==False]
# Convert the classes to numbers
# df['Target'] = class_to_numeric(df['Target'].tolist())

# Define the inputs and target
# X = df.drop(columns=['ComplaintId', 'Target'])
X = df.drop(columns=['Target'])
y = df['Target']
# print('Y', y)

# One-hot-encode the categorical features
# X = pd.get_dummies(X, columns=one_hot_required)
# X = pd.get_dummies(X, dummy_na=True)
X = pd.get_dummies(X)
# Remove accents from the column names (causes problems with MLflow logging)
for col_name in X.columns:
        u = unidecode(str(col_name), "utf-8")
        new_col_name = unidecode(u)
        new_col_name = new_col_name.replace("'", " ")
        new_col_name = new_col_name.replace("\n", "")
        new_col_name = new_col_name.replace("(", "")
        new_col_name = new_col_name.replace(")", "")
        new_col_name = new_col_name.replace(",", "")
        new_col_name = new_col_name.replace("*", "")
        X = X.rename(columns={col_name: new_col_name})
# print('COLUMN NAMES:', X.columns)
# Convert the target into the right shape
y = y.values.reshape(len(y),)

# print('X.COLUMNS', X.columns)

# Add y back to X, to do FeatureUnification together
X['Target'] = y

# # Drop the concat_text column (we cannot pass this to the Random Forest)
# X = X.drop(columns='concat_text')

# # The final df to unify
# features_to_unify.append(X)
# print([f.shape for f in features_to_unify])

# #-- Collins example code
# # unify all features
# X = FeatureUnification().unify(features_to_unify)

# Define the inputs and target
y = X['Target'].values
X = X.drop(columns=['Target'])

# Split the data into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=random_state)
# Add a validation set 
X_train, X_validate, y_train, y_validate = train_test_split(X_train, y_train, train_size=0.75, 
                                                            random_state=random_state)


# Fill NaN for numeric features
numeric_geo_cols = ['income_poverty_percentage', 
                    'multidimensional_poverty_percentage', 'surface_km2', 'tot_population',
                    'urban_zones_km2', 'protected_areas_km2', 'urban_pop_0_5',
                    'urban_pop_6_14', 'urban_pop_15_64', 'urban_pop_M65', 'rural_pop_0_5',
                    'rural_pop_6_14', 'rural_pop_15_64', 'rural_pop_M65',
                    'declared_area_poor_air_quality_km2',
                    'proportion_urban', 'proportion_protected'
                    ]
for col in numeric_geo_cols:
    if col in exp_features:
        # print(X_train[col].isna().sum(), X_test[col].isna().sum(), X_train[col].mean(), X_test[col].mean())
        X_train[col] = X_train[col].fillna(X_train[col].mean())
        X_test[col] = X_test[col].fillna(X_test[col].mean())
        X_validate[col] = X_validate[col].fillna(X_validate[col].mean())

# For PCA, we need to standardise the numerical values
# if apply_pca == True:


# # Perform dimensionality reduction using PCA
# if apply_pca == True:
#     # Save the ComplaintId column for later
#     complaint_ids_train = X_train['ComplaintId']
#     complaint_ids_test = X_test['ComplaintId']
#     X_train = X_train.drop(columns=['ComplaintId'])
#     X_test = X_test.drop(columns=['ComplaintId'])
#     # X_train = X_train.drop(columns=['Topic 49'])
#     # X_test = X_test.drop(columns=['Topic 49'])
#     # Standardise numerical columns so that no feature dominates the others
#     for numer_col in numeric_cols:
#         if numer_col in exp_features or numer_col in X_train.columns:
#             # print(numer_col)
#             for X in [X_train, X_test]:
#                 # print('PCA X.shape =', X.shape)
#                 col = X[numer_col]
#                 col = col.values.reshape(-1,1)
#                 scaler = StandardScaler().fit(col)
#                 scaled_col = scaler.transform(col)
#                 X[numer_col] = scaled_col
#     # print('PCA suspicious columns:', X_train.columns[X_train.mean(axis=0) > 0.2])
#     # print(X_train['rake_feature'].mean(axis=0))
#     # print(X_test['rake_feature'].mean(axis=0))
#     # print(X_train.iloc[:5,:])
#     # print(X_test.iloc[:5,:])
#     # Apply PCA
#     pca = PCA(n_components=pca_components)
#     X_train = pca.fit_transform(X_train)
#     # print('PCA train')
#     # print(pca.n_components_)
#     # print(pca.explained_variance_ratio_)
#     X_test = pca.fit_transform(X_test)
#     # print('PCA test')
#     # print(pca.n_components_)
#     # print(pca.explained_variance_ratio_)
#     # print('\nPCA')
#     # print(X_train.shape)
#     pca_cols = []
#     for i in range(pca_components):
#         pca_cols.append('PC' + str(i+1))
#     X_train = pd.DataFrame(X_train, columns=pca_cols)
#     X_test = pd.DataFrame(X_test, columns=pca_cols)
#     # Add the ComplaintId back into the dataframes
#     X_train['ComplaintId'] = complaint_ids_train
#     X_test['ComplaintId'] = complaint_ids_test
#     # print(X_train.iloc[:3,:])
#     # print(X_test.iloc[:3,:])
#     # print('\nColumns after PCA:', X_train.columns)


# print(X_train.shape, y_train.shape)
# print('columns with NaNs:', X_train.columns[X_train.isna().sum(axis=0) > 0])
# print(X_train.dtypes)
# print(X_train.iloc[:5,:])
# print(y_train.dtypes)
# print(y_train)
# Address class imbalance
oversampling_type = 'oversample'
if oversampling_type == 'smote':
    if 'FacilityId' in X_train.columns:
        X_train['FacilityId'] = X_train['FacilityId'].fillna(0)
X_train, y_train = balanced_sampling(X_train, y_train, oversampling_type)
X_train, y_train = balanced_sampling(X_train, y_train, 'undersample')
# X_train, y_train = balanced_sampling(X_train, y_train, 'undersample', 'not minority')
# print('BALANCE->',np.unique(y_train, return_counts=True))

# X_test, y_test = balanced_sampling(X_test, y_test, 'oversample')
# X_test, y_test = balanced_sampling(X_test, y_test, 'undersample')

# # Apply the feature transformations which depend on the train-test split
# if 'facility_num_sanctions' in exp_features:
#     X_train, X_test = facility_num_sanctions(X_train, X_test, complaints_facilities, facilities_sanction_id)
# if 'facility_has_sanctions' in exp_features:
#     X_train, X_test = facility_has_sanctions(X_train, X_test, complaints_facilities, facilities_inspection_id)
# if 'facility_num_inspections' in exp_features:
#     X_train, X_test = facility_num_inspections(X_train, X_test, complaints_facilities, facilities_inspection_id)
# if 'facility_has_inspections' in exp_features:
#     X_train, X_test = facility_has_inspections(X_train, X_test, complaints_facilities, facilities_inspection_id)
# if 'has_many_inspections' in exp_features:
#     X_train, X_test = has_many_inspections(X_train, X_test, complaints_facilities, facilities_inspection_id)
# print(X_test.iloc[:5,:])
# if 'sanction_inspection_ratio' in exp_features:
#     X_train, X_test = sanction_inspection_ratio(X_train, X_test,complaints_facilities,facilities_inspection_id,facilities_sanction_id)
#     if 'facility_num_sanctions' not in exp_features:
#         print('DROP sanctions')
#         X_train = X_train.drop(columns=['facility_num_sanctions'])
#         X_test = X_test.drop(columns=['facility_num_sanctions'])
#     if 'facility_num_inspections' not in exp_features:
#         print('DROP inspections')
#         X_train = X_train.drop(columns=['facility_num_inspections'])
#         X_test = X_test.drop(columns=['facility_num_inspections'])
# print(X_test.iloc[:5,:])


### Stress test: keep only complaints where the FacilityId is NaN
# X_test['Target'] = y_test
# X_test = X_test[X_test['FacilityId'].isna()==False]
# y_test = X_test['Target'].values
# X_test = X_test.drop(columns=['Target'])
###

print('Y-TEST', y_test)

if 'FacilityId' in X_train.columns:
    X_train = X_train.drop(columns=['FacilityId'])
if 'FacilityId' in X_test.columns:
    X_test = X_test.drop(columns=['FacilityId'])
if 'FacilityId' in X_validate.columns:
    X_validate = X_validate.drop(columns=['FacilityId'])

complaint_ids_test = X_test['ComplaintId']
X_train = X_train.drop(columns=['ComplaintId'])
X_test = X_test.drop(columns=['ComplaintId'])
X_validate = X_validate.drop(columns=['ComplaintId'])

# print('X.COLUMNS', X_train.columns)

# Fill NaN for numeric features
numeric_geo_cols = ['income_poverty_percentage', 
                    'multidimensional_poverty_percentage', 'surface_km2', 'tot_population',
                    'urban_zones_km2', 'protected_areas_km2', 'urban_pop_0_5',
                    'urban_pop_6_14', 'urban_pop_15_64', 'urban_pop_M65', 'rural_pop_0_5',
                    'rural_pop_6_14', 'rural_pop_15_64', 'rural_pop_M65',
                    'declared_area_poor_air_quality_km2',
                    'proportion_urban', 'proportion_protected'
                    ]
# if apply_pca == False:
#     for col in numeric_geo_cols:
#         if col in exp_features:
#             # print(X_train[col].isna().sum(), X_test[col].isna().sum(), X_train[col].mean(), X_test[col].mean())
#             X_train[col] = X_train[col].fillna(X_train[col].mean())
#             X_test[col] = X_test[col].fillna(X_test[col].mean())

# # Train-validation-test split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1)
# Split the data to training and testing
# X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=17)

# Metrics which depend on the train-test split
# facility_has_sanction
# facility_num_sanctions

# # Fit the Random Forest
# rf = RandomForestClassifier()
# rf.fit(X_train, y_train)

print(X_validate.iloc[:5,:])
print(X_validate.columns[X_test.isna().sum(axis=0) > 0])
print(X_test.shape, X_validate.shape)


print('--HYPERPARAMETER TUNING--')
#### Loop through hyperparameter space for tuning ####
df_base = df.copy()
if hyperparameter_tuning == True:
    X_test_actual = X_test.copy()
    y_test_actual = y_test.copy()
    X_test = X_validate.copy()
    y_test = y_validate.copy()
best_prop_rel_missed = 1.0
best_accuracy = None
best_prop_rel_missed_train = None
best_accuracy_train = None
best_h_params = {'max_depth':None, 'max_features':None, 'n_estimators':None, 'max_leaf_nodes':None}
for t_param in tuning_hyperparams:
    print()
    print()
    print(t_param)
    # Extract the hyperparameters to try
    h_params['max_depth'] = t_param[0]
    h_params['max_features'] = t_param[1]
    h_params['n_estimators'] = t_param[2]
    h_params['max_leaf_nodes'] = t_param[3]
    if use_min_samples_split == True:
        h_params['min_samples_split'] = t_param[4]
    if use_min_samples_leaf == True:
        h_params['min_samples_leaf'] = t_param[4]
    print(h_params)


    mlflow.set_experiment(mlflow_exp_name)
    with mlflow.start_run(nested=True):
        if m_type == 'gb':
            gb = GradientBoostingClassifier().set_params(**h_params)
            model = gb.fit(X_train, y_train)
        else:
            m = TrainModel(X_train=X_train, y_train=y_train, model_type=m_type, hyper_params=h_params)
            model = m.train()
            # model, kfold_scores = m.train()
        model_params = model.get_params()
        # kfold_mean = kfold_scores.mean()
        # kfold_std = kfold_scores.std()

        # track model metrics and params
        predictions = model.predict(X_test)


        # Make predictions on the test set
        y_pred = predictions

        # Find indices of relevant complaints
        ind_relevant = np.where(y_test==class_to_numeric(['Relevant']))[0]
        pred_relevant = y_pred[ind_relevant]
        relevant_missed = pred_relevant!=class_to_numeric(['Relevant'])
        prop_relevant_missed = sum(relevant_missed) / max(len(ind_relevant), 1)

        # Other metrics
        classif_report = classification_report(y_test, y_pred, output_dict=True, 
                                                target_names=numeric_to_class([0,1,2]))
        precisions = (round(classif_report['Archivo I']['precision'], 5), 
                    round(classif_report['Derivacion']['precision'], 5), 
                    round(classif_report['Relevant']['precision'], 5))
        recalls = (round(classif_report['Archivo I']['recall'], 5), 
                round(classif_report['Derivacion']['recall'], 5), 
                round(classif_report['Relevant']['recall'], 5))
        accuracy = model.score(X_test, y_test)
        confusion_matrix_test = confusion_matrix(y_test, y_pred)
        if m_type == 'svm':
            feature_importance = ''
        else:
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
        elif m_type == 'gb':
            algorithm = 'Gradient Boosting'
        else:
            algorithm = m_type

        # Save the metrics
        metrics_df = pd.read_csv('metrics_3class_new2.csv')
        exp_id = len(metrics_df)
        new_row = pd.DataFrame({'ExperimentId': [exp_id],
                                'Proportion of Relevant complaints missed': [prop_relevant_missed],
                                'Accuracy': [accuracy],
                                'Precision (Ar I, Der, Rel)': [precisions], 
                                'Recall (Ar I, Der, Rel)': [recalls], 
                                'Features': [exp_features],
                                '5 most important features': [feature_importance[:5]],
                                'Algorithm': [algorithm]})
        # metrics_df = pd.concat([metrics_df,new_row])
        metrics_df = metrics_df.append([new_row])
        if 'Unnamed: 0' in metrics_df.columns:
            metrics_df.drop(columns='Unnamed: 0')
        # metrics_df.iloc[len(metrics_df),:] = [len(metrics_df), len(metrics_df), prop_relevant_missed, accuracy, precisions,
        #                                       recalls, exp_features]
        metrics_df.to_csv('metrics_3class_new2.csv', index=False)

        # print('Feature Importance:')
        # print(feature_importance[:25])
        # print()
        # print(accuracy)
        # print(classification_report(y_test, y_pred,
        #                             target_names=numeric_to_class([0,1,2])))
        print('Test accuracy:', accuracy)
        print('\nProportion of Relevant complaints missed:', prop_relevant_missed)
        print('\nConfusion matrix:')
        print(confusion_matrix_test)

        # Use this once ModelMetrics is updated to deal with 3-class scenario
        model_metrics = ModelMetrics(y_true=y_test, y_pred=predictions)
        lr_metrics, _ = model_metrics.classification_metrics(include_report=True)
        lr_metrics['proportion_relevant_missed'] = prop_relevant_missed


        ## TRAINING SET metrics
        predictions_train = model.predict(X_train)
        y_train_pred = predictions_train

        # Find indices of relevant complaints
        ind_relevant_train = np.where(y_train==class_to_numeric(['Relevant']))[0]
        pred_relevant_train = y_train_pred[ind_relevant_train]
        relevant_missed_train = pred_relevant_train!=class_to_numeric(['Relevant'])
        prop_relevant_missed_train = sum(relevant_missed_train) / max(len(ind_relevant_train), 1)

        # Other metrics
        # classif_report_train = classification_report(y_train, y_train_pred, output_dict=True, 
        #                                              target_names=numeric_to_class([0,1,2]))
        # precisions = (round(classif_report['Archivo I']['precision'], 5), 
        #               round(classif_report['Derivacion']['precision'], 5), 
        #               round(classif_report['Relevant']['precision'], 5))
        # recalls = (round(classif_report['Archivo I']['recall'], 5), 
        #            round(classif_report['Derivacion']['recall'], 5), 
        #            round(classif_report['Relevant']['recall'], 5))
        accuracy_train = model.score(X_train, y_train)
        confusion_matrix_train = confusion_matrix(y_train, y_train_pred)
        print('\nTrain accuracy:', accuracy_train)
        # print(classification_report(y_test, y_pred,
        #                             target_names=numeric_to_class([0,1,2])))
        print('TRAIN Proportion of Relevant complaints missed:', prop_relevant_missed_train)
        print('\nTRAIN Confusion matrix:')
        print(confusion_matrix_train)
        # print('\nModel params:', model.get_params())
        print()

        # # Save the experiment outputs for random forest experiments
        # if m_type == 'rf':
        #     rf_exp = pd.read_csv('rf_experiments.csv')
        #     new_row = pd.DataFrame({'ExperimentId': [exp_id],
        #                             'max_depth': [model_params['max_depth']],
        #                             'max_features': [model_params['max_features']],
        #                             'Train accuracy': [accuracy_train],
        #                             'Test accuracy': [accuracy],
        #                             'Train Relevants missed': [prop_relevant_missed],
        #                             'Test Relevants missed': [prop_relevant_missed_train],
        #                             'Train confusion matrix': [confusion_matrix_train],
        #                             'Test confusion matrix': [confusion_matrix_test],
        #                             'model_params': [model_params],
        #                             'Features': [exp_features]})
        #     print(new_row)
        #     rf_exp = rf_exp.append([new_row])
        #     if 'Unnamed: 0' in rf_exp.columns:
        #         rf_exp.drop(columns='Unnamed: 0')
        #     rf_exp.to_csv('rf_experiments.csv', index=False)


        # if return_goodness_fit == True:
        #     goodness_fit = X_test.copy()
        #     goodness_fit['ComplaintId'] = complaint_ids_test
        #     goodness_fit['Target'] = y_test
        #     goodness_fit['Prediction'] = predictions
        #     goodness_fit.to_csv('goodness_of_fit.csv', index=False)

        # encode labels to prevent misrepresentation in categorical labels
        label_encoder = LabelEncoder()
        predicted_labels = label_encoder.fit_transform(predictions)
        actual_labels = label_encoder.fit_transform(y_test)

        # Log the training set metrics to MLflow
        lr_metrics['model_accuracy_train'] = accuracy_train
        lr_metrics['proportion_relevant_missed_train'] = prop_relevant_missed_train
        # lr_metrics['10-fold accuracy mean'] = kfold_mean
        # lr_metrics['10-fold accuracy st.d.'] = kfold_std
        # Log the hyperparameters to MLflow

        h_params1 = h_params.copy()
        del h_params1['random_state']
        model_metrics.log_hyperparameters_to_mlflow(hyper_params=h_params1)
        # for h in h_params.keys():
        #     lr_metrics['rf_' + h] = h_params[h]
        # lr_metrics['set'] = 1
        # del lr_metrics['rf_random_state']


        # Log the metrics for hyperparameter tuning to a csv file
        hp_exp = pd.read_csv('hparam_experiments.csv')
        new_row = pd.DataFrame({'HparamExpId': [len(hp_exp)],
                                'Set': ['Validation'],
                                'max_depth': [model_params['max_depth']],
                                'max_features': [model_params['max_features']],
                                'n_estimators': [model_params['n_estimators']],
                                'max_leaf_nodes': [model_params['max_leaf_nodes']],
                                'Train accuracy': [accuracy_train],
                                'Test accuracy': [accuracy],
                                # '10-fold accuracy mean': [kfold_mean],
                                # '10-fold accuracy st.d.': [kfold_std],
                                'Train Relevants missed': [prop_relevant_missed],
                                'Test Relevants missed': [prop_relevant_missed_train],
                                'Features': [exp_features]})
        hp_exp = hp_exp.append([new_row])
        if 'Unnamed: 0' in hp_exp.columns:
            hp_exp.drop(columns='Unnamed: 0')
        hp_exp.to_csv('hparam_experiments.csv', index=False)


        # Replace the best hyperparams if this exceeds the current best score
        if prop_relevant_missed < best_prop_rel_missed:
            best_prop_rel_missed = prop_relevant_missed
            best_accuracy = accuracy
            best_prop_rel_missed_train = prop_relevant_missed_train
            best_accuracy_train = accuracy_train
            best_h_params = h_params


        # track feature importance
        if m_type != 'gb':
            model_metrics.log_feature_importance(m)

        # track the metrics in mlflow
        model_metrics.log_metric_to_mlflow(lr_metrics)
        # Log the features used for this model run to MLflow
        model_metrics.log_model_features(exp_features)
        # track the model in mlflow
        mlflow.sklearn.log_model(model, "model")

    mlflow.end_run()
    # print("Experiment Done. Run $ `mlflow ui --backend-store-uri file:///files/sma_experiments/mlruns` to view results in mlflow dashboard")
    # print(f"Run $ `python make_prediction --run_id ${run_id} --data prediction_sample_data.csv` to use model for predictions")


    # if apply_pca == True:
    #     print(pca.n_components_)
    #     print(pca.explained_variance_ratio_)


### Test set

mlflow.set_experiment(mlflow_exp_name)
with mlflow.start_run(nested=True):

    X_test = X_test_actual
    y_test = y_test_actual
    y_pred = model.predict(X_test)

    m = TrainModel(X_train=X_train, y_train=y_train, model_type=m_type, hyper_params=best_h_params)
    model = m.train()
    # model, kfold_scores = m.train()
    model_params = model.get_params()
    # kfold_mean = kfold_scores.mean()
    # kfold_std = kfold_scores.std()

    # Find indices of relevant complaints
    ind_relevant = np.where(y_test==class_to_numeric(['Relevant']))[0]
    pred_relevant = y_pred[ind_relevant]
    relevant_missed = pred_relevant!=class_to_numeric(['Relevant'])
    prop_relevant_missed = sum(relevant_missed) / max(len(ind_relevant), 1)

    # Other metrics
    classif_report = classification_report(y_test, y_pred, output_dict=True, 
                                            target_names=numeric_to_class([0,1,2]))
    precisions = (round(classif_report['Archivo I']['precision'], 5), 
                round(classif_report['Derivacion']['precision'], 5), 
                round(classif_report['Relevant']['precision'], 5))
    recalls = (round(classif_report['Archivo I']['recall'], 5), 
            round(classif_report['Derivacion']['recall'], 5), 
            round(classif_report['Relevant']['recall'], 5))
    accuracy = model.score(X_test, y_test)
    confusion_matrix_test = confusion_matrix(y_test, y_pred)
    print('TEST cofusion matrix:')
    print(confusion_matrix_test)

    hp_exp = pd.read_csv('hparam_experiments.csv')
    new_row = pd.DataFrame({'HparamExpId': [len(hp_exp)],
                            'Set': ['Test'],
                            'max_depth': [model_params['max_depth']],
                            'max_features': [model_params['max_features']],
                            'n_estimators': [model_params['n_estimators']],
                            'max_leaf_nodes': [model_params['max_leaf_nodes']],
                            'Train accuracy': [accuracy_train],
                            'Test accuracy': [accuracy],
                            # '10-fold accuracy mean': [kfold_mean],
                            # '10-fold accuracy st.d.': [kfold_std],
                            'Train Relevants missed': [prop_relevant_missed],
                            'Test Relevants missed': [prop_relevant_missed_train],
                            'Features': [exp_features]})
    # print(new_row)
    hp_exp = hp_exp.append([new_row])
    if 'Unnamed: 0' in hp_exp.columns:
        hp_exp.drop(columns='Unnamed: 0')
    hp_exp.to_csv('hparam_experiments.csv', index=False)


    # Use this once ModelMetrics is updated to deal with 3-class scenario
    model_metrics = ModelMetrics(y_true=y_test, y_pred=y_pred)
    lr_metrics, _ = model_metrics.classification_metrics(include_report=True)
    lr_metrics['proportion_relevant_missed'] = prop_relevant_missed
    lr_metrics['model_accuracy_train'] = accuracy_train
    lr_metrics['proportion_relevant_missed_train'] = prop_relevant_missed_train
    # lr_metrics['10-fold accuracy mean'] = kfold_mean
    # lr_metrics['10-fold accuracy st.d.'] = kfold_std
    h_params1 = h_params.copy()
    del h_params1['random_state']
    model_metrics.log_hyperparameters_to_mlflow(hyper_params=h_params1)
    # for h in h_params.keys():
    #     lr_metrics['rf_' + h] = h_params[h]
    # lr_metrics['set'] = 2
    # del lr_metrics['rf_random_state']

    # track the metrics in mlflow
    model_metrics.log_metric_to_mlflow(lr_metrics)
    # Log the features used for this model run to MLflow
    model_metrics.log_model_features(exp_features)
    # track the model in mlflow
    mlflow.sklearn.log_model(model, "model")
    # track feature importance
    if m_type != 'gb':
        model_metrics.log_feature_importance(m)

mlflow.end_run()
print('\nHyperparameter Tuning Done.')