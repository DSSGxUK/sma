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
from tqdm import tqdm

# Use TQDM to show progress bars
tqdm.pandas()

helpers_dir = (os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) + '/helpers/')
sys.path.append(helpers_dir)
sma_dir = (os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')) + '/sma/')
sys.path.append(sma_dir)
sma_project_dir = (os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')) + '/sma-project/')
sys.path.append(sma_project_dir)
data_preprocess = (os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) + '/data/')
sys.path.append(data_preprocess)

from data_preprocess import merge_etl_data
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
# Random Forest parameters
h_params = {'max_depth': 9, 'max_features': 50, 'n_estimators': 100, 'max_leaf_nodes': None}
h_params['random_state'] = 17

save_trained_text_features = True

## FUNCTIONS ##

def get_columns_to_drop(FEATURE_NAMES):
    """
    Returns the list of column names which should be dropped from the imported data.
    """
    columns_to_drop = ['ComplaintStatus', 'ComplaintType', 'DateComplaint',
                        'DateResponse', 'EndType', 'DigitalComplaintId', 'PreClasification',
                        'Clasification', 'FacilityId', 'FacilityRegion', 'FacilityDistrict',
                        'FacilityEconomicSector', 'FacilityEconomicSubSector', 'concat_text', 
                        'District', 'District_y', 'District_x', 'CutId', 'income_poverty_percentage', 
                        'multidimensional_poverty_percentage', 'surface_km2', 'tot_population',
                        'urban_zones_km2', 'protected_areas_km2', 'urban_pop_0_5',
                        'urban_pop_6_14', 'urban_pop_15_64', 'urban_pop_M65', 'rural_pop_0_5',
                        'rural_pop_6_14', 'rural_pop_15_64', 'rural_pop_M65',
                        'declared_area_poor_air_quality_km2', 'Unnamed: 0', 'Unnamed: 0.1', 'Longitude', 'Latitude', 'Region']

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

def three_class_target(endtype):
    """
    Construct the target variable for the 3-class classification problem.
    """
    # Check if the EndType is NaN
    if isinstance(endtype, float):
        return endtype
    elif endtype in ['Archivo II','Formulación de Cargos','Formulacion de Cargos']:
        return 'Relevant'
    else:
        return endtype

def class_to_numeric(classes):
    """
    Convert each class category to a numerical value.
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

def balanced_sampling(X, y, sampling_type, strategy = 'default', target_col: str = 'Target'):
    """
    Performs undersampling or oversampling according to the specified strategy.
    Allowed samplers are undersampling or oversampling.
    Returns the specified number of randomly-sampled data points for each class.
    """
    ascending_counts = sorted(Counter(y).items(), key = lambda tup: tup[1])
    X = X.loc[:, ~X.columns.duplicated()]
    if sampling_type == 'oversample':
        if strategy == 'default':
            # Oversample the minimum class to the middle-sized class
            strategy = {ascending_counts[0][0]: ascending_counts[1][1],
                        ascending_counts[1][0]: ascending_counts[1][1],
                        ascending_counts[2][0]: ascending_counts[2][1]}
        ros = RandomOverSampler(sampling_strategy=strategy, random_state=h_params['random_state'])
        X_resampled, y_resampled = ros.fit_resample(X, y)
        
    elif sampling_type == 'undersample':
        if strategy == 'default':
            # Undersample the maximum class to the middle-sized class
            strategy = {ascending_counts[0][0]: ascending_counts[0][1],
                        ascending_counts[1][0]: ascending_counts[1][1],
                        ascending_counts[2][0]: ascending_counts[1][1]}
        rus = RandomUnderSampler(sampling_strategy=strategy, random_state=h_params['random_state'])
        X_resampled, y_resampled = rus.fit_resample(X, y)

    elif sampling_type == 'smote':
        strategy = {ascending_counts[0][0]: ascending_counts[1][1],
                        ascending_counts[1][0]: ascending_counts[1][1],
                        ascending_counts[2][0]: ascending_counts[2][1]}
        sm = SMOTE(sampling_strategy=strategy, random_state=h_params['random_state'])
        X_resampled, y_resampled = sm.fit_resample(X, y)
    else:
        print('Input error: sampling_type must be one of [oversample,undersample,smote]')
        
    return X_resampled, y_resampled
    
def create_target_variable(data):
    """
    Takes in the dataframe containing the complaints and facilities data (one row per complaint) 
    and returns a copy of this dataframe with an added column for the Target variable of the 
    three-class relevance classification problem.
    """
    # Create the target variable
    df = data.copy()
    df['Target'] = df['EndType'].apply(three_class_target)
    # Drop rows where the Target is NaN
    df = df[df['Target'].notna()]
    # Convert Target classes to numbers (this is required for the Random Forest)
    df['Target'] = class_to_numeric(df['Target'].values.tolist())
    return df
    
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

def top_tfidf_words(X_train, NUMBER_OF_TFIDF_FEATURES=100):
    """
    Finds the words with the highest TF-IDF scores for each of the target classes and returns these. 
    Call the tfidf_word_counts(...) function after this one.

    Makes use of the Dense TF-IDF Transform class.
    
    Args:
            X_train (pd.DataFrame): The training set dataframe (containing the complaint text)
            NUMBER_OF_TFIDF_FEATURES (int, optional): The number of top TF-IDF words to select from each category

    Example:
        >>> X_train = pd.DataFrame({"ComplaintId": [1, 2, 3], 
                                    "concat_text": ["Text for first complaint", 
                                                    "Text for second complaint",
                                                    "Text for third complaint"]})
        >>> top_tfidf_words = top_tfidf_words(X_train)
    """
    # Implement the DenseTFIDFVectorizer for unigrams, bigrams and trigrams
    dense_tfidf = DenseTFIDFVectorization(ngram_range=(1, 3), max_features=NUMBER_OF_TFIDF_FEATURES)

    # Concatenate the text from all of the complaints in each of the three target classes
    X_archivo = X_train[X_train['Target']==0]['concat_text']
    X_derivacion = X_train[X_train['Target']==1]['concat_text']
    X_relevant = X_train[X_train['Target']==2]['concat_text']
    X_archivo = ' '.join(X_archivo)
    X_derivacion = ' '.join(X_derivacion)
    X_relevant = ' '.join(X_relevant)

    # Take a random subsample of the words, as there is a 1,000,000 character limit to TfidfVectorizer
    X_archivo_split = X_archivo.split()
    X_derivacion_split = X_derivacion.split()
    X_relevant_split = X_relevant.split()
    # Randomly select 50,000 words from the corpus for each class, if the class contains too
    # many characters
    num_random_words = 50000
    if len(X_archivo) > 1000000:
        X_archivo_split = random.sample(X_archivo_split, num_random_words)
    if len(X_derivacion) > 1000000:
        X_derivacion_split = random.sample(X_derivacion_split, num_random_words)
    if len(X_relevant) > 1000000:
        X_relevant_split = random.sample(X_relevant_split, num_random_words)

    # Make the words back into a single string per target class
    X_archivo = ' '.join(X_archivo_split)
    X_derivacion = ' '.join(X_derivacion_split)
    X_relevant = ' '.join(X_relevant_split)

    # Combine the text of each class into a DataFrame to pass to the TfidfVectorizer
    X_archivo = pd.DataFrame({'class_text': [X_archivo]})
    X_derivacion = pd.DataFrame({'class_text': [X_derivacion]})
    X_relevant = pd.DataFrame({'class_text': [X_relevant]})
    X = pd.concat([X_archivo, X_derivacion, X_relevant], axis=0)

    # Apply the TF-IDF transform
    dtfidf = dense_tfidf.fit_transform(X['class_text'])
    dtfidf['class'] = ['archivo','derivacion','relevant']
    dtfidf = dtfidf.set_index(['class'])

    # Get the columns (words) with the highest TF-IDF score for each group
    dtfidf_archivo = dtfidf.sort_values(by='archivo', axis=1, ascending=False)
    dtfidf_derivacion = dtfidf.sort_values(by='derivacion', axis=1, ascending=False)
    dtfidf_relevant = dtfidf.sort_values(by='relevant', axis=1, ascending=False)

    # Concatenate the columns into one DataFrame
    dtfidf = pd.concat([dtfidf_archivo, dtfidf_derivacion, dtfidf_relevant], axis=1)
    dtfidf = dtfidf.loc[:, ~dtfidf.columns.duplicated()]
    if '00' in dtfidf.columns:
        # '00' can appear as a word, due to times mentioned in the complaint 
        # text (for example eight o'clock written as 8.00). Let's remove this.
        tfidf = tfidf.drop(columns=['00'])
    # Get the list of columns, which gives us the top words for the three target classes
    tfidf_words = dtfidf.columns

    return tfidf_words


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

    df['words'] = df['concat_text'].apply(lambda x : x.split(' '))
    df = df.progress_apply(count_words, axis=1)
    return df.drop(columns=['words'])

def fill_na_numeric_geo_columns(X_train, X_test, FEATURE_NAMES):
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
            X_train.loc[:,col] = X_train[col].fillna(X_train[col].mean())
            X_test.loc[:,col] = X_test[col].fillna(X_test[col].mean())
    
    return X_train, X_test

def perform_topic_modelling(X_train, X_test, NUMBER_OF_LDA_TOPICS=50):
    """
    Performs Latent Dirichlet Allocation (LDA) on the training set to learn the most important topics in the
    complaint texts. Then adds a column for each topic to the training set dataframe, with each column 
    containing the probability that the complaint text mentions that topic. Finally, adds the topic columns 
    to the test set dataframe, with the topics learned from the training set.
    
    Makes use of the TopicScores function.
    
    Args:
            X_train (pd.DataFrame): The training set dataframe (containing the complaint text)
            X_test (pd.DataFrame): The test set dataframe (containing the complaint text)
            NUMBER_OF_LDA_TOPICS (int, optional): The number of topics to learn with LDA

    Example:
        >>> X_train = pd.DataFrame({"ComplaintId": [1, 2, 3], 
                                    "concat_text": ["Text for first complaint", 
                                                    "Text for second complaint",
                                                    "Text for third complaint"]})
        >>> X_test = pd.DataFrame({"ComplaintId": [132, 134, 135], 
                                    "concat_text": ["Text for first complaint", 
                                                    "Text for second complaint",
                                                    "Text for third complaint"]})
        >>> X_train, X_test, lda_model = perform_topic_modelling(X_train, X_test)
    """
    model, corpus, id2word, bigram = TopicScores(X_train, num_topics = NUMBER_OF_LDA_TOPICS)

    train_vecs = []
    for i in range(len(X_train)):
        top_topics = model.get_document_topics(corpus[i], minimum_probability=0.0)
        topic_vec = [top_topics[i][1] for i in range(NUMBER_OF_LDA_TOPICS)]
        train_vecs.append(topic_vec)

    train_topics_df = pd.DataFrame(train_vecs, dtype = float) 
    train_topics_df.columns = pd.Index(np.arange(1,len(train_topics_df.columns)+1).astype(str))
    for topic in train_topics_df.columns:
        train_topics_df = train_topics_df.rename(columns={topic:'LDA_Topic_' + str(topic)})
    X_train = pd.concat([X_train,train_topics_df],axis = 1)
    if 'Number' in X_train.columns:
        X_train = X_train.drop(['Number'], axis=1)
    def bigrams(words, bi_min=15, tri_min=10):
        bigram = gensim.models.Phrases(words, min_count = bi_min)
        bigram_mod = gensim.models.phrases.Phraser(bigram)
        return bigram_mod

    def get_bigram(df):
        """
        For the test data we only need the bigram data built on 2017 reviews,
        as we'll use the 2016 id2word mappings. This is a requirement due to 
        the shapes Gensim functions expect in the test-vector transformation below.
        With both these in hand, we can make the test corpus.
        """
        tokenized_list = [simple_preprocess(doc) for doc in df['concat_text']]
        bigram = bigrams(tokenized_list)
        bigram = [bigram[review] for review in tokenized_list]
        return bigram

    bigram_test = get_bigram(X_test)

    test_corpus = [id2word.doc2bow(text) for text in bigram_test]

    test_vecs = []
    for i in range(len(X_test)):
        top_topics = model.get_document_topics(test_corpus[i], minimum_probability=0.0)
        topic_vec = [top_topics[i][1] for i in range(NUMBER_OF_LDA_TOPICS)]
        test_vecs.append(topic_vec)

    test_topics_df = pd.DataFrame(test_vecs, dtype = float).reset_index(drop=True)
    test_topics_df.columns = pd.Index(np.arange(1,len(test_topics_df.columns)+1).astype(str))
    for topic in test_topics_df.columns:
        test_topics_df = test_topics_df.rename(columns={topic:'LDA_Topic_' + str(topic)})
    X_test = X_test.reset_index(drop=True)
    X_test = pd.concat([X_test, test_topics_df], axis = 1)
    if 'Number' in X_test.columns:
        X_test = X_test.drop(['Number'], axis=1)

    return X_train, X_test, model, id2word


## DATA PREPARATION ##
def transform_raw_data_to_features(FEATURE_NAMES, data, split_data=True):

    # Load the data
    # complaints = data.get('complaints_registry')
    # complaints_facilities = data.get('complaints_facilities_registry')
    # facilities_sanction_id = data.get('facilities_sanction_id')
    # sanctions = data.get('sanctions_registry')
    complaints = pd.read_csv(data['complaints_registry'], encoding='latin-1')
    complaints_facilities = pd.read_csv(data['complaints_facilities_registry'], encoding='latin-1')
    facilities_sanction_id = pd.read_csv(data['facilities_sanction_id'], encoding='latin-1')
    sanctions = pd.read_csv(data['sanctions_registry'], encoding='latin-1')

    # Convert the text to ascii
    complaints = normalize_ascii(complaints)
    complaints_facilities = normalize_ascii(complaints_facilities)
    facilities_sanction_id = normalize_ascii(facilities_sanction_id)
    sanctions = normalize_ascii(sanctions)
    print('\nData read in and normalized to ascii.')
    print('Number of complaints in the dataset:', complaints_facilities.shape[0])

    # Create the Target variable as a column of the dataframe
    df = create_target_variable(complaints_facilities)
    print('\nCreated the target variable.')
    print('Number of complaints after dropping complaints with no EndType:', df.shape[0])

    # Concatenate the text by complaint
    df = concatenate_text(df, complaints)
    print('\nAdded the concatenated text of the complaint details for each complaint.')
    print('Number of complaints after adding in the complaint text:', df.shape[0])

    # Get the list of columns to discard later
    columns_to_drop = get_columns_to_drop(FEATURE_NAMES)

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
        df = df[df['Target'].notna()]
    if 'total_past_fines' in FEATURE_NAMES:
        df['total_past_fines'] = total_past_fines(df, complaints_facilities, facilities_sanction_id, sanctions)
        df = df[df['Target'].notna()]
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

    X = df.drop(columns=['Target'])
    y = df['Target']
    print('\nSplit the data into X (features) and y (target variable).')
    print('Shape of X:', X.shape)
    print('Shape of y:', y.shape)

    # One-hot encode the necessary columns
    X = pd.get_dummies(X)
    if save_trained_text_features == True:
        # Save the column names for use in the prediction script
        with open('relevance_column_names.csv', 'w') as f:
            write = csv.writer(f)
            write.writerow(X.columns.to_list())
    # Remove accents and symbols from the column names (causes problems with MLflow logging)
    X = clean_column_names(X)
    # Add the concat_text column back in, as we need this for the text features later
    X['concat_text'] = concat_text
    X['concat_text'] = X['concat_text'].fillna(' ')
    print('\nOne-hot-encoded the necessary columns.')
    print('Shape of X:', X.shape)

    # Convert the target into the right shape
    y = y.values.reshape(len(y),)
    # Add the target column back into X, as this is needed for text feature extraction later
    X['Target'] = y
    print('\nAdded Target variable back into X (temporarily required for text feature extraction later) and reshaped y.')
    print('Shape of X:', X.shape)
    print('Shape of y:', y.shape)

    # Split the data to training and testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=h_params['random_state'])
    print('\nPerformed the train-test split.')
    print('Shape of X_train:', X_train.shape)
    print('Shape of X_test:', X_test.shape)
    print('Shape of y_train:', y_train.shape)
    print('Shape of y_test:', y_test.shape)

    # Fill NaN values for the numerical geographical features
    X_train, X_test = fill_na_numeric_geo_columns(X_train, X_test, FEATURE_NAMES)
    
    # Address class imbalance
    oversampling_type = 'oversample'
    X_train, y_train = balanced_sampling(X_train, y_train, oversampling_type)
    X_train, y_train = balanced_sampling(X_train, y_train, 'undersample')
    print('\nResampled the training data to balance out the classes.')
    print('Shape of X_train:', X_train.shape)
    print('Shape of y_train:', y_train.shape)

    # Preprocess the text in the concat_text field to prepare for the text feature extraction
    print('\nPreprocessing the text data (stopword removal, lemmatization, stemming)...')
    X_train.loc[:,'concat_text'] = X_train['concat_text'].apply(normalize_text).apply(lemmatize_and_stem_text)
    X_test.loc[:,'concat_text'] = X_test['concat_text'].apply(normalize_text).apply(lemmatize_and_stem_text)
    print('Preprocessed the text data.')
    print('Shape of X_train:', X_train.shape)
    print('Shape of X_test:', X_test.shape)

    # Extract the text features
    if 'TF-IDF' in FEATURE_NAMES:
        print('\nAdding top TF-IDF words for each class (learned from the training data only)...')
        tfidf_words = top_tfidf_words(X_train, NUMBER_OF_TFIDF_FEATURES)
        X_train = tfidf_word_counts(X_train, tfidf_words)
        X_test = tfidf_word_counts(X_test, tfidf_words)
        print('Added top TF-IDF words to the training and test sets.')
        print('Shape of X_train:', X_train.shape)
        print('Shape of X_test:', X_test.shape)
        if save_trained_text_features == True:
            # Save the top TF-IDF words for use in the prediction script
            with open('relevance_tfidf_words.csv', 'w') as f:
                write = csv.writer(f)
                write.writerow(tfidf_words.to_list())

    if 'RAKE' in FEATURE_NAMES:
        print('\nAdding the RAKE feature (learned from the training data only)...')
        rake_features = DenseRakeFeatures(num_phrases=NUMBER_OF_RAKE_PHRASES)
        X_test = X_test.reset_index(drop=True)
        rake_features.fit(X_train['concat_text'])
        if save_trained_text_features == True:
            # Save the fitted RAKE model as a .pkl file for use in the prediction script
            pickle.dump(rake_features, open('relevance_rake.pkl', 'wb'))
        # Apply the fitted RAKE model to the training and test sets separately
        X_train.loc[:,'rake_feature'] = rake_features.transform(X_train['concat_text'])
        X_test.loc[:,'rake_feature'] = rake_features.transform(X_test['concat_text'])
        print('Added the RAKE feature to the training and test sets.')
        print('Shape of X_train:', X_train.shape)
        print('Shape of X_test:', X_test.shape)

    if 'LDA' in FEATURE_NAMES:
        print('\nAdding the most important topics from LDA (learned from the training data only)')
        X_train, X_test, lda_model, id2word = perform_topic_modelling(X_train, X_test, NUMBER_OF_LDA_TOPICS)
        print('Added the most important topics to the training and test sets.')
        print('Shape of X_train:', X_train.shape)
        print('Shape of X_test:', X_test.shape)
        if save_trained_text_features == True:
            # Save the fitted LDA model for use in the prediction script
            pickle.dump(lda_model, open('relevance_lda_model.pkl', 'wb'))
            # Save the id2word dictionary for use in the prediction script
            id2word.save('relevance_id2word.dict')


    # Remove the columns we no longer need from X_train and X_test
    X_train = X_train.drop(columns=['Target','concat_text'])
    X_test = X_test.drop(columns=['Target','concat_text'])
    complaint_ids_test = X_test['ComplaintId']
    X_train = X_train.drop(columns=['ComplaintId'])
    X_test = X_test.drop(columns=['ComplaintId'])
    print('\nFinal shape of the training and test sets:')
    print('Shape of X_train:', X_train.shape)
    print('Shape of y_train', y_train.shape)
    print('Shape of X_test:', X_test.shape)
    print('Shape of y_test', y_test.shape)

    if(split_data):
        return X_train, X_test, y_train, y_test
    X = pd.concat([X_train, X_test], ignore_index=True)
    return X


## RUN EXPERIMENT ##

def run_experiment(X_train, X_test, y_train, y_test, model_type="rf", experiment_name="Relevance Model", model_output_path="relevance_model/"):
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
    # Set the experiment name here
    mlflow.set_experiment(experiment_name=experiment_name)

    run_id = None
    with mlflow.start_run(nested=True) as experiment_run:
        # get the experiment run id
        run_id = experiment_run.info.run_id

        # train model
        print('\nTraining the model...')
        m = TrainModel(X_train=X_train, y_train=y_train, model_type=model_type, hyper_params=h_params)
        model = m.train()
        print('Model trained!')

        # make predictions with model using test set
        predictions = model.predict(X_test)
        print('Number of complaints in X_test', X_test.shape[0])
        print('Number of complaints in y_test', y_test.shape[0])
        print('Number of predicted complaints:', len(predictions))

        # Calculate the proportion of relevant complaints missed
        ind_relevant = np.where(y_test==class_to_numeric(['Relevant']))[0]
        pred_relevant = predictions[ind_relevant]
        relevant_missed = pred_relevant!=class_to_numeric(['Relevant'])
        prop_relevant_missed = sum(relevant_missed) / max(len(ind_relevant), 1)

        # encode labels to prevent misrepresentation in categorical labels
        label_encoder = LabelEncoder()
        predicted_labels = label_encoder.fit_transform(predictions)
        actual_labels = label_encoder.fit_transform(y_test)

        # get the model metrics
        model_metrics = ModelMetrics(
            y_true=actual_labels, y_pred=predicted_labels)
        lr_metrics = model_metrics.regression_metrics(
            include_report=True, classification_metric=True)
        lr_metrics['proportion_relevant_missed'] = prop_relevant_missed

        # track feature importance
        model_metrics.log_model_features(FEATURE_NAMES)
        model_metrics.log_feature_importance(m)
        model_metrics.log_hyperparameters_to_mlflow(m._model_params)
        print(lr_metrics)

        # track the metrics in mlflow
        model_metrics.log_metric_to_mlflow(lr_metrics)
        # track the model in mlflow
        mlflow.sklearn.log_model(model, "model")

        # log the model to disk
        model_metrics.save_model_to_disk(model, file_path=model_output_path)
        mlflow.set_tag("model_type", model_type)
    mlflow.end_run()
    return run_id

def main(data_path, model_output_path):
    # data = merge_etl_data(data_folder_path=data_path, include_territories=True)
    data = {'complaints_registry': data_path + 'complaints_registry.csv',
        'complaints_facilities_registry': data_path + 'complaints_facilities_registry.csv',
        'facilities_sanction_id': data_path + 'facilities_sanction_id.csv',
        'sanctions_registry': data_path + 'sanctions_registry.csv',
        'inspections_registry': data_path + 'inspections_registry.csv',
        'facilities_registry': data_path + 'facilities_registry.csv',
        'complaints_inspections_registry': data_path + 'complaints_inspections_registry.csv',
        'complaints_sanctions_registry': data_path + 'complaints_sanctions_registry.csv'
    }
    
    X_train, X_test, y_train, y_test = transform_raw_data_to_features(FEATURE_NAMES, data=data)

    # Run the experiment
    experiment_id = run_experiment(X_train, X_test, y_train, y_test, model_type="random_forest", experiment_name="3-class Relevance Model", model_output_path=model_output_path)

    print("Experiment Done. Run $ `mlflow ui` to view results in mlflow dashboard")


# RUN:
# python three_class_relevance.py --data "/files/data_merge/prod" --output "RelevanceModel/"
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", help="Data folder where the data files will be read from")
    parser.add_argument("--output", help="Path where the model pickle file will be saved")

    args = parser.parse_args()
    main(args.data, args.output)
