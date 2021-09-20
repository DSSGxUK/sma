import pandas as pd
import os
import sys
import csv
import pickle
import pandas as pd
import numpy as np
from collections import Counter
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
import seaborn 
import sklearn 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler, SMOTE
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt  
import gensim
import gensim.corpora as corpora
from gensim.models.phrases import Phrases, Phraser
from gensim.utils import simple_preprocess
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import scale
from sklearn.feature_selection import RFE
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics.pairwise import cosine_similarity
from yellowbrick.regressor import ResidualsPlot
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

helpers_dir = (os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) + '/helpers/')
sys.path.append(helpers_dir)
data_preprocess = (os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) + '/data/')
sys.path.append(data_preprocess)
sma_dir = (os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')) + '/sma/')
sys.path.append(sma_dir)
sma_project_dir = (os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')) + '/sma-project/')
sys.path.append(sma_project_dir)

from feature_transformation import *
from parse_csv import normalize_ascii
from tf_idf import *
from document_term_matrix import *
from feature_extraction import *
from text_cleaning import *
# from tfidf_pipeline import DenseTFIDFVectorization

# -- Collins example code
from data_preprocess import merge_etl_data
from feature_union import FeatureUnification
from model_metrics import ModelMetrics
from rake_extraction import DenseRakeFeatures
from tfidf_transform import DenseTFIDFVectorization
from train_model import TrainModel
from topic_models import TopicScores
from sklearn.preprocessing import OneHotEncoder
import mlflow
from pandas.api.types import CategoricalDtype
from feature_transformation import num_words, natural_region, facility_mentioned, \
                                    proportion_urban, proportion_protected, num_past_sanctions, \
                                    min_num_words, max_num_words, populated_districts, quarter, \
                                    month, weekday, proportion_poor_air, ComplaintType_archivo1

NUMBER_OF_LDA_TOPICS = 50
NUMBER_OF_TFIDF_FEATURES = 45 
# Random Forest parameters
h_params = {'max_depth': 30, 'max_features': 'sqrt', 'n_estimators': 550, 'min_samples_split': 3, 'min_samples_leaf':1, 'bootstrap':False}
h_params['random_state'] = 17
save_trained_text_features = True

# Create Sanction Level according to the method chosen 
def add_target(df, method = 'worst'):
    level_type = CategoricalDtype(categories= ['Leves','Graves','Gravísimas'], ordered=True)
    df['SanctionLevel'] = df['InfractionCategory'].astype(level_type)
    if method == 'worst':
        target_df = (
            df.groupby('ComplaintId', as_index=False)
            .SanctionLevel
            .max()
            .merge(df[['ComplaintId','MonetaryPenalty']].dropna().drop_duplicates(),on='ComplaintId',how='left')
            .drop_duplicates(subset = ['ComplaintId'])
            #.dropna(subset=['SanctionLevel', 'MonetaryPenalty'], how='all')
        )
        return target_df
    if method == 'most':
        target_df = (
            df.groupby(['ComplaintId','SanctionLevel'], as_index=False)['SanctionInfractionId']
            .count()
            .sort_values(['ComplaintId','SanctionInfractionId','SanctionLevel'], ascending=[True,False,False]))
        target_df = (
            target_df[target_df['SanctionInfractionId'] != 0].groupby(['ComplaintId'])
            .head(1)
            .reset_index(drop=True)
            .loc[:,['ComplaintId','SanctionLevel']]
            .merge(df[['ComplaintId','MonetaryPenalty']].dropna().drop_duplicates(),on='ComplaintId',how='left')
            .drop_duplicates())
            #.dropna(subset=['SanctionLevel', 'MonetaryPenalty'], how='all'))
        return target_df


# Create Target variable according to SanctionLevel and MonetaryPenalty
def add_label (row):
    if pd.isnull(row['SanctionLevel']):
        if row['MonetaryPenalty'] > 5000:
            return 1
        else: 
            return 0
    elif row['SanctionLevel'] == 'Leves' :
        return 0
    else:
        return 1


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
    # Concatenate the text from all of the complaints in low/high
    X_low = X_train[X_train['Target']==0]['concat_text']
    X_high = X_train[X_train['Target']==1]['concat_text']

    X_low = ' '.join(X_low)
    X_high = ' '.join(X_high)


    # Take a random subsample of the words, as there is a 1,000,000 character limit to TfidfVectorizer
    X_low_split = X_low.split()
    X_high_split = X_high.split()
    # Randomly select 50,000 words from the corpus for each class, if the class contains too
    # many characters
    num_random_words = 50000
    if len(X_low) > 1000000:
        X_low_split = random.sample(X_low_split, num_random_words)
    if len(X_high) > 1000000:
        X_high_split = random.sample(X_gigh_split, num_random_words)


    # Make the words back into a single string per target class
    X_low = ' '.join(X_low_split)
    X_high = ' '.join(X_high_split)

    # Combine the text of each class into a DataFrame to pass to the TfidfVectorizer
    X_low = pd.DataFrame({'class_text': [X_low]})
    X_high = pd.DataFrame({'class_text': [X_high]})
 
    X = pd.concat([X_low, X_high], axis=0)

    # Apply the TF-IDF transform
    dtfidf = dense_tfidf.fit_transform(X['class_text'])
    dtfidf['class'] = ['low','high']
    dtfidf = dtfidf.set_index(['class'])

    # Get the columns (words) with the highest TF-IDF score for each group
    dtfidf_low = dtfidf.sort_values(by='low', axis=1, ascending=False)
    dtfidf_high = dtfidf.sort_values(by='high', axis=1, ascending=False)


    # Concatenate the columns into one DataFrame
    dtfidf = pd.concat([dtfidf_low, dtfidf_high], axis=1)
    dtfidf = dtfidf.loc[:, ~dtfidf.columns.duplicated()]
    if '00' in dtfidf.columns:
        # '00' can appear as a word, due to times mentioned in the complaint 
        # text (for example eight o'clock written as 8.00). Let's remove this.
        dtfidf = dtfidf.drop(columns=['00'])
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
    df = df.apply(count_words, axis=1)
    return df.drop(columns=['words'])


# Set the seed
np.random.seed(0)


# Define the experiment feature
exp_features = ['ComplaintType','TF-IDF','LDA','EnvironmentalTopic','num_words','num_details',
                'facility_num_infractions',
                'natural_region','populated_district','FacilityEconomicSector',
                'month']


def transform_raw_data_to_features(exp_features, data):
    # Import and create target datasets

    # Load the data
    complaints = pd.read_csv(data['complaints_registry'], encoding='latin-1')
    complaints_facilities = pd.read_csv(data['complaints_facilities_registry'], encoding='latin-1')
    facilities_sanction_id = pd.read_csv(data['facilities_sanction_id'], encoding='latin-1')
    complaints_sanctions = pd.read_csv(data["complaints_sanctions_registry"], encoding='latin-1')
    sanctions = pd.read_csv(data['sanctions_registry'], encoding='latin-1')

    complaints = normalize_ascii(complaints)
    complaints_facilities= normalize_ascii(complaints_facilities)
    complaints_sanctions = normalize_ascii(complaints_sanctions)
    facilities_sanctions = normalize_ascii(facilities_sanction_id)
    sanctions = normalize_ascii(sanctions)

    # Merge datasets
    #sanction_df = pd.merge(sanction_df[['SanctionId','MonetaryPenalty']],infraction_df[['SanctionId','SanctionInfractionId','InfractionCategory']],on = 'SanctionId',how = 'left').dropna(subset = ['InfractionCategory','MonetaryPenalty'], how = 'all')
    com_sac_df = (
        complaints_sanctions.dropna(subset = ['SanctionId','ComplaintId'])
        .merge(sanctions[['SanctionId', 'InfractionCategory']], on = 'SanctionId')
        .drop_duplicates())
    #print(com_sac_df.info())
    com_sac_df_worst = add_target(com_sac_df, method = 'worst').dropna(subset=['SanctionLevel', 'MonetaryPenalty'], how='all')

    #print(com_sac_df_worst.info())


    com_sac_df_worst['Target'] = com_sac_df_worst.apply (lambda row: add_label(row), axis=1)

    df = com_sac_df_worst
    com_df = complaints[['ComplaintId','ComplaintDetail','Number','ComplaintType','EndType','EnvironmentalTopic']]
    com_topic = complaints[['ComplaintId','EnvironmentalTopic']]
    if("ComplaintId_x" in complaints_facilities.columns):
        complaints_facilities.rename({"ComplaintId_x": "ComplaintId"}, inplace=True)
        if("Complaint_y" in complaints_facilities.columns):
            complaints_facilities.drop("Complaint_y", inplace=True)

    com_df = com_df.merge(complaints_facilities[['ComplaintId','FacilityId','FacilityRegion','FacilityDistrict','FacilityEconomicSector','FacilityEconomicSubSector'
    ]], how = 'left')
    com_df['ComplaintDetail'] = com_df['ComplaintDetail'].astype(str)


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

    print("Cleaning finished!")
    print(df.info())
    #initialize pipeline steps
    # Extract the text features from complaints_registry (if required)
    if 'EnvironmentalTopic' in exp_features:
        env_topic = pivot_env_topics(com_topic)
        df = pd.merge(df, env_topic, on = 'ComplaintId', how = 'left').fillna(0)

    print("Environmental Topics finished!")
    print(df.info())
    print(df.shape)

    if 'populated_districts' in exp_features:
        pop_df = populated_districts(complaints_facilities)
        pop_df = pop_df[['FacilityId','populated_districts']]
        complaints_facilities_df = pd.merge(complaints_facilities[['ComplaintId','FacilityId']], pop_df, on = 'FacilityId', how = 'left')
        complaints_facilities_df['ComplaintId'] = complaints_facilities_df['ComplaintId'].astype(str)
        complaints_facilities_df['FacilityId'] = complaints_facilities_df['ComplaintId'].astype(str)
        complaints_facilities_df = complaints_facilities_df.groupby('ComplaintId').sum().reset_index()
        complaints_facilities_df['ComplaintId'] = complaints_facilities_df['ComplaintId'].astype(int)
        complaints_facilities_df['FacilityId'] = complaints_facilities_df['ComplaintId'].astype(int)
        df = df.merge(complaints_facilities_df[['ComplaintId','populated_districts']], on = 'ComplaintId', how = 'left').drop_duplicates(subset = ['ComplaintId'])
        #df = feature_df.drop(['FacilityId'], axis = 1).fillna(0)

    print("populated_districts finished!")
    print(df.info())
    print(df.shape)

    if 'month' in exp_features:
        complaints = month(complaints)
        df = df.merge(complaints[['ComplaintId','month']], on = 'ComplaintId', how = 'left').drop_duplicates(subset = ['ComplaintId'])

    print("month finished!")
    print(df.info())
    print(df.shape)

    if 'quarter' in exp_features:
        complaints = quarter(complaints)
        df = df.merge(complaints[['ComplaintId','quater']], on = 'ComplaintId', how = 'left')
    
    print("quater finished!")
    print(df.info())
    print(df.shape)

    if 'FacilityEconomicSector' in exp_features:
        eco_df = pd.concat([complaints_facilities[['FacilityId']],pd.get_dummies(complaints_facilities['FacilityEconomicSector'])], axis=1)
        complaints_facilities_eco = pd.merge(complaints_facilities[['ComplaintId','FacilityId']], eco_df, on = 'FacilityId', how = 'left')
        complaints_facilities_eco['ComplaintId'] = complaints_facilities_eco['ComplaintId'].astype(str)
        complaints_facilities_eco['FacilityId'] = complaints_facilities_eco['ComplaintId'].astype(str)
        complaints_facilities_eco = complaints_facilities_eco.groupby('ComplaintId').sum().reset_index()
        complaints_facilities_eco['ComplaintId'] = complaints_facilities_eco['ComplaintId'].astype(int)
        complaints_facilities_eco['FacilityId'] = complaints_facilities_eco['ComplaintId'].astype(int)
        df = df.merge(complaints_facilities_eco, on = 'ComplaintId', how = 'left').drop(['FacilityId'], axis = 1)

    print("FacilityEconomicSector finished!")
    print(df.info())
    print(df.shape)

    if 'ComplaintType' in exp_features:
        ComType = pd.concat([complaints[['ComplaintId']],pd.get_dummies(complaints[['ComplaintType']])], axis=1)
        df = df.merge(ComType, on = 'ComplaintId', how = 'left').fillna(0).drop_duplicates(subset = ['ComplaintId'])

    print("ComplaintType finished!")
    print(df.info())
    print(df.shape)

    if 'num_details' in exp_features:
        df['num_details'] = df['Number']

    print("num_details finished!")
    print(df.info())
    print(df.shape)


    # Select the feature transformations to include from the feature_transformations.py file
    if 'min_num_words' in exp_features:
        df = min_num_words(df)

    print("min_num_words finished!")
    print(df.info())
    print(df.shape)

    if 'max_num_words' in exp_features:
        df = max_num_words(df)

    print("max_num_words finished!")
    print(df.info())
    print(df.shape)

    if 'natural_region' in exp_features:
        facility_region = natural_region(complaints_facilities)
        region_df = pd.concat([facility_region[['FacilityId']],pd.get_dummies(facility_region['natural_region'])], axis=1)
        complaints_facilities_df = pd.merge(complaints_facilities[['ComplaintId','FacilityId']],region_df, on = 'FacilityId', how = 'left')
        complaints_facilities_df['ComplaintId'] = complaints_facilities_df['ComplaintId'].astype(str)
        complaints_facilities_df['FacilityId'] = complaints_facilities_df['ComplaintId'].astype(str)
        complaints_facilities_df = complaints_facilities_df.groupby('ComplaintId').sum().reset_index()
        complaints_facilities_df['ComplaintId'] = complaints_facilities_df['ComplaintId'].astype(int)
        complaints_facilities_df['FacilityId'] = complaints_facilities_df['ComplaintId'].astype(int)
        df = df.merge(complaints_facilities_df, on = 'ComplaintId', how = 'left').drop(['FacilityId'], axis = 1).fillna(0)

    print("natural_region finished!")
    print(df.info())
    print(df.shape)

    if 'num_words' in exp_features:
        df = num_words(df)

    print("num_words finished!")
    print(df.info())
    print(df.shape)

    if 'facility_num_infractions' in exp_features:
        df = df.dropna(subset = ['ComplaintId'])
        fac_df = get_complaint_sanctions(df=df, complaints_facilities=complaints_facilities, facilities_sanctions=facilities_sanctions, sanctions=sanctions)
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

    print("facility_num_infractions finished!")
    print(df.info())
    print(df.shape)

    if 'money_fined' in exp_features:
        fac_df = get_complaint_sanctions(df=df, complaints_facilities=complaints_facilities, facilities_sanctions=facilities_sanctions, sanctions=sanctions)
        money_fined = fac_df.groupby(['ComplaintId'], as_index = False).MonetaryPenalty.sum()
        df = df.merge(money_fined, how = 'left').rename(columns = {'MonetaryPenalty':'MoneyFined'}).drop_duplicates(subset = ['ComplaintId'])
        df['MoneyFined'] = df['MoneyFined'].fillna(0)


    if 'facility_num_sanctions' in exp_features:
        fac_df = get_complaint_sanctions(df=df, complaints_facilities=complaints_facilities, facilities_sanctions=facilities_sanctions, sanctions=sanctions)
        sanction_number = (
            fac_df.groupby(['ComplaintId','SanctionId'], as_index = False)
            .size()
            .groupby(['ComplaintId'], as_index = False)
            .size())
        df = df.merge(sanction_number, how = 'left').rename(columns = {'size':'SanctionNumber'}).drop_duplicates(subset = ['ComplaintId'])
        df['SanctionNumber'] = df['SanctionNumber'].fillna(0)

    print("facility_num_sanctions finished!")
    print(df.info())
    print(df.shape)

    # Define the inputs and target
    X = df
    y = df['Target']
    print(X.shape, y.shape)
    if save_trained_text_features == True:
        # Save the column names for use in the prediction script
        with open('sanction_column_names.csv', 'w') as f:
            write = csv.writer(f)
            X1 = df.drop(columns=['Target'])
            write.writerow(X1.columns.to_list())   
        print('Column names other than TF-IDF and LDA are exported to /sanction_column_names.csv')     


    # Extract the text features
    if 'TF-IDF' in exp_features:
        print('\nAdding top TF-IDF words for each class (learned from the training data only)...')
        tfidf_words = top_tfidf_words(X, NUMBER_OF_TFIDF_FEATURES)
        #Save the top TF-IDF words for use in the prediction script
        if save_trained_text_features == True:
            with open('sanction_tfidf_words.csv', 'w') as f:
                write = csv.writer(f)
                write.writerow(tfidf_words.to_list())
            print('Top TF-IDF words exported to /sanction_tf_idf_words.csv')
        X = tfidf_word_counts(X, tfidf_words)
        print('Added top TF-IDF words to model features.')
        print('Shape of X:', X.shape)

 
    if 'LDA' in exp_features:
        lda_model, corpus,id2word, bigram = TopicScores(X, num_topics = NUMBER_OF_LDA_TOPICS)
        if save_trained_text_features == True:
            pickle.dump(lda_model, open('sanction_lda_model.pkl', 'wb'))
            print('lda pickle file exported to /sanction_lda_model.pkl')
            
        def bigrams(words, bi_min=15, tri_min=10):
            bigram = gensim.models.Phrases(words, min_count = bi_min)
            bigram_mod = gensim.models.phrases.Phraser(bigram)
            return bigram_mod

        def get_bigram(df):
            tokenized_list = [simple_preprocess(doc) for doc in df['concat_text']]
            bigram = bigrams(tokenized_list)
            bigram = [bigram[review] for review in tokenized_list]
            id2word = corpora.Dictionary(bigram)
            id2word.filter_extremes(no_below=10, no_above=0.35)
            id2word.compactify()
            return bigram, id2word
        
        bigram_X,id2word = get_bigram(X)
        corpus = [id2word.doc2bow(text) for text in bigram_X]

        vecs = []
        for i in range(len(X)):
            top_topics = lda_model.get_document_topics(corpus[i], minimum_probability=0.0)
            topic_vec = [top_topics[i][1] for i in range(NUMBER_OF_LDA_TOPICS)]
            vecs.append(topic_vec)

        #print(test_vecs)
        print(len(vecs))

        topics_df = pd.DataFrame(vecs, dtype = float).reset_index(drop=True)
        topics_df.columns = pd.Index(np.arange(1,len(topics_df.columns)+1).astype(str))
        X = X.reset_index(drop=True)
        X = pd.concat([X,topics_df],axis = 1)
    
        print(X.info())
        print(X.shape)
        print("LDA finished!")


    def clean_dataset(df):
        assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
        df.dropna(inplace=True)
        indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
        return df[indices_to_keep].astype(np.float64)
    X = X.drop(['concat_text','Number'], axis=1)
    X = clean_dataset(X)
    return X, y


def main(data_path):
    #data = merge_etl_data(data_folder_path=data_path, include_territories=True)
    data = {'complaints_registry': data_path + 'complaints_registry.csv',
        'complaints_facilities_registry': data_path + 'complaints_facilities_registry.csv',
        'facilities_sanction_id': data_path + 'facilities_sanction_id.csv',
        'sanctions_registry': data_path + 'sanctions_registry.csv',
        'inspections_registry': data_path + 'inspections_registry.csv',
        'facilities_registry': data_path + 'facilities_registry.csv',
        'complaints_inspections_registry': data_path + 'complaints_inspections_registry.csv',
        'complaints_sanctions_registry': data_path + 'complaints_sanctions_registry.csv'
    }
    
    X, y = transform_raw_data_to_features(exp_features, data=data)

    # Drop the complaint ID
    if ('ComplaintId' in X.columns):
        X.drop(["ComplaintId"],axis = 1, inplace=True)
    if ('Target' in X.columns):
        X.drop(["Target"],axis = 1, inplace=True)
    # train model
    model_type = 'rf'
    m = TrainModel(X_train=X, y_train=y, model_type=model_type,hyper_params=h_params)
    model = m.train()
    # save model to disk
    if save_trained_text_features == True:
        pickle.dump(model, open('sanction_model_saved.pkl', 'wb'))
        print('sg classification Model has been exported as sanction_model_saved.pkl under same folder.')
    print("Models for TF-IDF, LDA and Classification have been trained and exported!")
    

# RUN:
# python sanction_model_training.py 
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", help="Data folder where the data files will be read from")
    
    args = parser.parse_args()
    main(args.data)
    # main(data)