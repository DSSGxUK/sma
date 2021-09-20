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
import csv
import pickle
from datetime import datetime

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
                                    proportion_urban, proportion_protected, num_past_sanctions, \
                                    min_num_words, max_num_words, populated_districts, quarter, \
                                    month, weekday, proportion_poor_air, ComplaintType_archivo1
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

# data_preprocess = (os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) + '/data/')
# sys.path.append(data_preprocess)
# from data_preprocess import merge_etl_data

## RUN SETUP ##

# Set the seed
np.random.seed(0)

# Set required parameters and the features to use
NUMBER_OF_LDA_TOPICS = 50
NUMBER_OF_TFIDF_FEATURES = 100
NUMBER_OF_RAKE_PHRASES = 15000
FEATURE_NAMES = ['TF-IDF', 'LDA', 'RAKE', 'num_words', 'FacilityRegion', 'EnvironmentalTopic', 
                'natural_region', 'facility_mentioned', 'proportion_urban', 'proportion_protected',
                'num_past_sanctions'
                ]


## FUNCTIONS ##

def get_columns_to_drop(FEATURE_NAMES):
    """
    Returns the list of column names which should be dropped from the imported data.
    """
    columns_to_drop = ['ComplaintStatus', 'ComplaintType', 'DateComplaint',
                        'DateResponse', 'EndType', 'DigitalComplaintId', 'PreClasification',
                        'Clasification', 'FacilityId', 'FacilityRegion', 'FacilityDistrict',
                        'FacilityEconomicSector', 'FacilityEconomicSubSector', 'concat_text', 
                        'District', 'CutId', 'income_poverty_percentage', 
                        'multidimensional_poverty_percentage', 'surface_km2', 'tot_population',
                        'urban_zones_km2', 'protected_areas_km2', 'urban_pop_0_5',
                        'urban_pop_6_14', 'urban_pop_15_64', 'urban_pop_M65', 'rural_pop_0_5',
                        'rural_pop_6_14', 'rural_pop_15_64', 'rural_pop_M65',
                        'declared_area_poor_air_quality_km2', 'Unnamed: 0', 'Unnamed: 0.1']
    if ('month' in FEATURE_NAMES) or ('quarter' in FEATURE_NAMES):
        columns_to_drop.remove('DateComplaint')
    if 'proportion_urban' in FEATURE_NAMES:
        columns_to_drop.remove('urban_zones_km2')
        columns_to_drop.remove('surface_km2')
    if 'proportion_protected' in FEATURE_NAMES:
        columns_to_drop.remove('protected_areas_km2')
        if 'surface_km2' in columns_to_drop:
            columns_to_drop.remove('surface_km2')
    if 'proportion_poor_air' in FEATURE_NAMES:
        columns_to_drop.remove('declared_area_poor_air_quality_km2')
        if 'surface_km2' in columns_to_drop:
            columns_to_drop.remove('surface_km2')
    for col in columns_to_drop:
        if col in FEATURE_NAMES:
            columns_to_drop.remove(col)

    return columns_to_drop

def class_to_numeric(classes):
    """
    Convert each class category to a numerical value (input should be a list).
    """
    class_to_num = {'Archivo I': 0, 'Derivacion Total a Organismo Competente': 1, 
                    'Derivación Total a Organismo Competente': 1, 'Relevant': 2}
    return [class_to_num[c] for c in classes]
    
def numeric_to_class(numbers):
    """
    Convert the numerical representation of each class back to the class category.
    """
    num_to_class = {0: 'Archivo I', 1: 'Derivacion', 2: 'Relevant'}
    return [num_to_class[n] for n in numbers]
    
def concatenate_text(df, complaints):
    """
    Takes in the dataframe and the complaints data, returns the dataframe with a columns 
    containing the concatenated string of complaint text for each row.
    """
    # Get the concatenated complaint details text
    concat_text = concat_complaint_details(complaints) 
    def get_concat_text(complaint_id):
        if complaint_id in concat_text.keys():
            return concat_text[complaint_id]
        else:
            return ' '
    df['concat_text'] = df['ComplaintId'].apply(get_concat_text)
    df['concat_text'] = df['concat_text'].fillna(' ')
    return df

def clean_column_names(X):
    """
    Remove accents and symbols from the feature names, as this causes issue when logging to MLflow.
    """
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
    return X

def environmental_topic(df, complaints):
    """
    One-hot encode the environmental topics of the complaints.
    """
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
    df = df.join(encoder_features_df, on='ComplaintId', rsuffix='_r', how='left')

    return df

def list_to_string(row):
        row_str = []
        for list_item in row:
            if isinstance(list_item, float):
                list_item = str(list_item)
            row_str.append(list_item)
        row = '|'.join(row_str)
        return row

def tfidf_word_counts(df, tfidf_words):
    """
    Return the number of times each of the top TF-IDF words occur in each complaint.

    Call this on the output of the top_tfidf_words(...) function.
    
    Args:
            df (pd.DataFrame): The training ot test dataframe (containing the complaint text)
            tfidf_words (list): The top TF-IDF words to count

    Example:
        >>> X_train = pd.DataFrame({"ComplaintId": [1, 2, 3], 
                                    "concat_text": ["Text for first complaint", 
                                                    "Text for second complaint",
                                                    "Text for third complaint"]})
        >>> X_test = pd.DataFrame({"ComplaintId": [132, 134, 135], 
                                    "concat_text": ["Text for first complaint", 
                                                    "Text for second complaint",
                                                    "Text for third complaint"]})
        >>> tfidf_words = top_tfidf_words(X_train)
        >>> X_train = tfidf_word_counts(X_train, tfidf_words)
        >>> X_test = tfidf_word_counts(X_test, tfidf_words)
    """
    # Count how many times each word occurs for each row (class)
    def count_words(row): 
        for word in tfidf_words:
            row[word] = row['words'].count(word)
        return row

    df.loc[:,'words'] = df['concat_text'].apply(lambda x : x.split(' '))
    df = df.apply(count_words, axis=1)
    return df.drop(columns=['words'])

def fill_na_numeric_geo_columns_pred(X, FEATURE_NAMES):
    """
    Fill the NaN values in the geographical variables which are numerical, so that the random
    forest is able to deal with them.
    """
    numeric_geo_cols = ['income_poverty_percentage', 
                        'multidimensional_poverty_percentage', 'surface_km2', 'tot_population',
                        'urban_zones_km2', 'protected_areas_km2', 'urban_pop_0_5',
                        'urban_pop_6_14', 'urban_pop_15_64', 'urban_pop_M65', 'rural_pop_0_5',
                        'rural_pop_6_14', 'rural_pop_15_64', 'rural_pop_M65',
                        'declared_area_poor_air_quality_km2',
                        'proportion_urban', 'proportion_protected']
    for col in numeric_geo_cols:
        if col in FEATURE_NAMES:
            # Fill NaN values with the mean of the column
            X.loc[:,col] = X[col].fillna(X[col].mean())
    
    return X

def bigrams_pred(words, bi_min=15, tri_min=10):
        bigram = gensim.models.Phrases(words, min_count = bi_min)
        bigram_mod = gensim.models.phrases.Phraser(bigram)
        return bigram_mod

def get_bigram_pred(df):
        """
        For the test data we only need the bigram data built on 2017 reviews,
        as we'll use the 2016 id2word mappings. This is a requirement due to 
        the shapes Gensim functions expect in the test-vector transformation below.
        With both these in hand, we can make the test corpus.
        """
        tokenized_list = [simple_preprocess(doc) for doc in df['concat_text']]
        bigram = bigrams_pred(tokenized_list)
        bigram = [bigram[review] for review in tokenized_list]
        return bigram


## DATA PREPARATION ##

def transform_raw_data_to_features_pred(FEATURE_NAMES, data):
 
    # Load the data
    complaints = pd.read_csv(data['complaints'], encoding='latin-1')
    complaints_facilities = pd.read_csv(data['complaints_facilities'], encoding='latin-1')
    facilities_sanction_id = pd.read_csv(data['facilities_sanction_id'], encoding='latin-1')
    sanctions = pd.read_csv(data['sanctions'], encoding='latin-1')

    # Convert the text to ascii
    complaints = normalize_ascii(complaints)
    complaints_facilities = normalize_ascii(complaints_facilities)
    facilities_sanction_id = normalize_ascii(facilities_sanction_id)
    sanctions = normalize_ascii(sanctions)
    print('\nData read in and normalized to ascii.')
    print('Number of complaints in the dataset:', complaints_facilities.shape[0])

    # Drop the EndType column
    df = complaints_facilities.copy()
    print('Shape before dropping the EndType column:', df.shape)
    if 'EndType' in df.columns:
        df = df.drop(columns=['EndType'])
    print('Shape after dropping the EndType column:', df.shape)

    # Concatenate the text by complaint
    df = concatenate_text(df, complaints)
    print('\nAdded the concatenated text of the complaint details for each complaint.')
    print('Number of complaints after adding in the complaint text:', df.shape[0])

    # Get the list of columns to discard later
    columns_to_drop = get_columns_to_drop(FEATURE_NAMES)

    # # Set the list of features to unify
    # features_to_unify = []

    # One-hot encode the Environmanetal topics of the complaints
    if 'EnvironmentalTopic' in FEATURE_NAMES:
        print('\nAdding in the EnvironmentalTopic...')
        df = environmental_topic(df, complaints)
        print('Added the EnvironmentalTopic.')
        print('Number of complaints after adding the EnvironmentalTopic:', df.shape[0])

    # One-hot encode additional descriptor data
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
    print('\nAdded the additional descriptor data (Distance from event, Affected Population, etc).')
    print('Number of complaints:', df.shape[0])

    # Select the feature transformations to include from the feature_transformations.py file
    if 'num_words' in FEATURE_NAMES:
        df = num_words(df)
    if 'min_num_words' in FEATURE_NAMES:
        df = min_num_words(df)
    if 'max_num_words' in FEATURE_NAMES:
        df = max_num_words(df)
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
        # df = df[df['Target'].notna()]
    if 'total_past_fines' in FEATURE_NAMES:
        df['total_past_fines'] = total_past_fines(df, complaints_facilities, facilities_sanction_id, sanctions)
        # df = df[df['Target'].notna()]
    print('\nAdded the feature transformations.')
    print('Number of complaints:', df.shape[0])

    # Remove the FacilityId if it is there
    if 'FacilityId' in df.columns:
        df = df.drop(columns=['FacilityId'])

    # Remove the DateComplaint field from the features if it is still there
    if 'DateComplaint' in df.columns:
        df = df.drop(columns=['DateComplaint'])
        if 'DateComplaint' in columns_to_drop:
            columns_to_drop.remove('DateComplaint')

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
    print('\nAdded the feature transformations based on geographical data.')
    print('Number of complaints:', df.shape[0])

    # Keep a copy of the concat_text column for later
    concat_text = df['concat_text']
    # Drop the columns we don't need
    df = df.reset_index()
    df = df.drop(columns=['index'])
    for col in columns_to_drop:
        if col in df.columns:
            df = df.drop(columns=col)

    # # Unify the Environmentl Topic with the rest of the dataframe
    # features_to_unify.append(df)
    # df = FeatureUnification().unify(features_to_unify)

    X = df.copy()
    print('\nShape of X:', X.shape)
    for x in X.columns:
        print(x)

    # One-hot encode the necessary columns
    X = pd.get_dummies(X)
    # Import the names of all the columns expected by the model, saved during the model training stage
    #with open('relevance_column_names.csv') as f:
    #    reader = csv.reader(f)
    #    for row in reader:
    #        full_column_names = row
    full_column_names = pd.read_csv('relevance_column_names.csv', encoding='latin-1').columns
    
    # Remove any categories not present in the trained model, as the model will not know how to 
    # deal with these
    for col in X.columns:
        if col not in full_column_names:
            X.drop(col, axis=1, inplace=True)
    # Add in the missing columns which the model expects (the reason columns may be missing is 
    # because the dataset to predict on may have fewer categories for some of the features, so 
    # the one-hot encoding from the pd.get_dummies() step results in fewer columns)
    for name in full_column_names:
        if name not in X.columns:
            # Add the column, with zeros in all the rows
            X[name] = pd.Series([0] * X.shape[0])
    # Specify the order of the columns to be the same as in the model training stage
    X = X[full_column_names]
    # Remove accents and symbols from the column names (causes problems with MLflow logging)
    X = clean_column_names(X)
    # Add the concat_text column back in, as we need this for the text features later
    X['concat_text'] = concat_text
    X['concat_text'] = X['concat_text'].fillna(' ')
    print('\nOne-hot-encoded the necessary columns.')
    print('Shape of X:', X.shape)

    # Fill NaN values for the numerical geographical features
    X = fill_na_numeric_geo_columns_pred(X, FEATURE_NAMES)

    # Preprocess the text in the concat_text field to prepare for the text feature extraction
    print('\nPreprocessing the text data (stopword removal, lemmatization, stemming)...')
    X.loc[:,'concat_text'] = X['concat_text'].apply(normalize_text).apply(lemmatize_and_stem_text)
    print('Preprocessed the text data.')
    print('Shape of X:', X.shape)

    # Extract the text features
    if 'TF-IDF' in FEATURE_NAMES:
        print('\nAdding top TF-IDF words for each class (learned during the model training stage)...')
        # Import the saved top TF-IDF words learned from the model training stage
        #with open('relevance_tfidf_words.csv') as f:
        #    reader = csv.reader(f)
        #    for row in reader:
        #        tfidf_words = row
        tfidf_words = pd.read_csv('relevance_tfidf_words.csv', encoding='latin-1' ).columns
        X = tfidf_word_counts(X, tfidf_words)
        print('Added top TF-IDF words to the dataframe.')
        print('Shape of X:', X.shape)

    if 'RAKE' in FEATURE_NAMES:
        print('\nAdding the RAKE feature (learned during the model training stage)...')
        # Load the fitted RAKE model from the .pkl file
        rake_features = pickle.load(open('relevance_rake.pkl', 'rb'))
        # Apply the fitted RAKE model to the training and test sets separately
        X.loc[:,'rake_feature'] = rake_features.transform(X['concat_text'])
        print('Added the RAKE feature to the dataframe.')
        print('Shape of X:', X.shape)

    if 'LDA' in FEATURE_NAMES:
        print('\nAdding the most important topics from LDA (learned during the model training stage)')
        # Load the fitted LDA model from the .pkl file
        lda_model = pickle.load(open('relevance_lda_model.pkl', 'rb'))
        # Load the id2word dictionary saved at the model training stage
        id2word = gensim.corpora.dictionary.Dictionary.load('relevance_id2word.dict')
        # Get the corpus for LDA (including bigrams)
        bigram_test = get_bigram_pred(X)
        corpus = [id2word.doc2bow(text) for text in bigram_test]
        # Get the LDA vectors
        vecs = []
        for i in range(len(X)):
            top_topics = lda_model.get_document_topics(corpus[i], minimum_probability=0.0)
            topic_vec = [top_topics[i][1] for i in range(NUMBER_OF_LDA_TOPICS)]
            vecs.append(topic_vec)
        # Create a dataframe for the LDA information
        topics_df = pd.DataFrame(vecs, dtype = float).reset_index(drop=True)
        topics_df.columns = pd.Index(np.arange(1,len(topics_df.columns)+1).astype(str))
        for topic in topics_df.columns:
            topics_df = topics_df.rename(columns={topic:'LDA_Topic_' + str(topic)})
        # Concatenate the X dataframe with the LDA dataframe
        X = X.reset_index(drop=True)
        X = pd.concat([X, topics_df], axis = 1)
        if 'Number' in X.columns:
            X = X.drop(['Number'], axis=1)
        print('Added the most important topics to the dataframe.')
        print('Shape of X:', X.shape)

    # Remove the columns we no longer need from X_train and X_test
    X = X.drop(columns=['concat_text'])
    complaint_ids = X['ComplaintId']
    X = X.drop(columns=['ComplaintId'])
    print('\nFinal shape of the dataframe:')
    print('Shape of X:', X.shape)

    return X, complaint_ids


## RUN EXPERIMENT ##

def run_predictions(X, complaint_ids, model_type="rf"):
    """Runs the predictions using a saved pre-trained model

    Args:
        X (numpy.ndarray, scipy.sparse.csr_matrix, pd.DataFrame): The dataframe to use as input
        model_type (str, optional): A model type parameter. The `model type` can be specified using the full name qualifier or the short name.
        `model_type` must be one of [lr, smv, nb, rf, rfr, xg]. Defaults to "rf". See [train_model](train_model.md) for more details
    """
    # Load the .pkl file where the model is saved
    print('\nLoading model to run the predictions...')
    model = pickle.load(open('relevance_model_saved.pkl', 'rb'))

    # Make predictions with the model
    predictions = model.predict(X)
    predictions = numeric_to_class(predictions)
    print('Predictions complete!')
    print('Number of complaints in the input dataframe', X.shape[0])
    print('Number of predicted complaints:', len(predictions))

    model_type = 'relevance'

    model_predictions_df = pd.DataFrame({
        "complaint_id": complaint_ids,
        "model_prediction": predictions,
        "model_type": model_type,
        "prediction_timestamp": datetime.now().isoformat(),
    })

    model_predictions_df.to_csv(f"{model_type}_model_predictions.csv")
    
#     ## upload to db
#     import pyodbc
#     import os

#     D = "ODBC Driver 17 for SQL Server"
#     S = "<DB_SERVER>"
#     U = "<DB_USERNAME>"
#     P = os.environ["dbPass"]
#     db = "dssg"

#     cnxn = pyodbc.connect('DRIVER={ODBC Driver 17 for SQL Server};SERVER='+S+';DATABASE='+db+';UID='+U+';PWD='+P)
#     cursor = cnxn.cursor()
    
#     insert_stmt = ("INSERT INTO Predictions (complaint_id, model_prediction, model_type, prediction_timestamp) values(%s, %s, %s, %s)")
  
#   , 
#           row.complaint_id, row.model_prediction, row.model_type, row.prediction_timestamp)
#   "VALUES (%s, %s, %s, %s)"
# )
# # data = (2, 'Jane', 'Doe', datetime.date(2012, 3, 23))
# data = (13570,"Relevant", "relevance", "2021-08-30T19:15:57.803185")
# cursor.execute(insert_stmt, data)


#     # Insert Dataframe into SQL Server:
#     for index, row in model_predictions_df.iterrows():
#         cursor.execute("INSERT INTO Predictions (complaint_id, model_prediction, model_type, prediction_timestamp) values(?,?,?,?)", 
#           row.complaint_id, row.model_prediction, row.model_type, row.prediction_timestamp)
#     cnxn.commit()
#     cursor.close()
    
    
    
    return predictions

def main(data_path):
    data = {'complaints': data_path + 'complaints_registry.csv',
              'complaints_facilities': data_path + 'complaints_facilities_registry.csv',
              'facilities_sanction_id': data_path + 'facilities_sanction_id.csv',
              'sanctions': data_path + 'sanctions_registry.csv'}
              
    X, complaint_ids = transform_raw_data_to_features_pred(FEATURE_NAMES, data)
    predictions = run_predictions(X, complaint_ids, model_type="random_forest")
    
    # print("Experiment Done. Run $ `mlflow ui` to view results in mlflow dashboard")
    # print(f"Run $ `python make_prediction.py --run_id ${experiment_id} --data ../sample_data/prediction_sample_data.csv` to use model for predictions")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", help="Data folder where the data files will be read from")

    args = parser.parse_args()
    main(args.data)

