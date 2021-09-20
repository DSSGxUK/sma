import pandas as pd
import os
import sys
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

# Path to the data files
data = {'complaints': '../../../../files/data_merge/complaint_registry.csv',
              'complaints_facilities': '../../../../files/data_merge/complaints_facilities_registry.csv',
              'facilities_sanction_id': '../../../../files/data_merge/facilities_sanction_id.csv',
              'complaints_sanctions' : '../../../../files/data_merge/complaints_sanctions_registry.csv',
              'sanctions': '../../../../files/data_merge/sanctions_registry.csv'}

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


def balanced_sampling(X, y, sampling_type, strategy = 'default', target_col: str = 'Target'):
    """
    Performs undersampling or oversampling according to the specified strategy.
    Allowed samplers are undersampling or oversampling.
    Returns the specified number of randomly-sampled data points for each class.
    """
    ascending_counts = sorted(Counter(y).items(), key = lambda tup: tup[1])

    if sampling_type == 'oversample':
        ros = RandomOverSampler(sampling_strategy='minority')
        X_resampled, y_resampled = ros.fit_resample(X, y)
        
    elif sampling_type == 'undersample':
        rus = RandomUnderSampler(sampling_strategy='majority')
        X_resampled, y_resampled = rus.fit_resample(X, y)

    else:
        print('Input error: sampling_type must be one of [oversample,undersample,smote]')
        
    return X_resampled, y_resampled
# Set the seed
np.random.seed(0)


# Define the experiment feature
exp_features = ['ComplaintType','TF-IDF','LDA','EnvironmentalTopic','num_words','num_details',
                'facility_num_infractions',
                'natural_region','populated_district','FacilityEconomicSector',
                'month']

# exp_features = ['facility_num_infractions']
# Set the list of features to unify
features_to_unify = []

def transform_raw_data_to_features(exp_features, data, split_data=True):

    # Load the data
    complaints = pd.read_csv(data['complaints_registry'], encoding='latin-1')
    complaints_facilities = pd.read_csv(data['complaints_facilities_registry'], encoding='latin-1')
    facilities_sanction_id = pd.read_csv(data['facilities_sanction_id'], encoding='latin-1')
    sanctions = pd.read_csv(data['sanctions_registry'], encoding='latin-1')
    complaints_sanctions = pd.read_csv(data["complaints_sanctions_registry"], encoding='latin-1')
    
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

    # # Import and create target datasets

    # complaints = normalize_ascii(data["complaints_registry"])
    # complaints_facilities= normalize_ascii(data["complaints_facilities_registry"])
    # complaints_sanctions = normalize_ascii(data["complaints_sanctions_registry"])
    # facilities_sanctions = normalize_ascii(data["facilities_sanction_id"])
    # sanctions = data["sanctions_registry"]


    # # Merge datasets
    # #sanction_df = pd.merge(sanction_df[['SanctionId','MonetaryPenalty']],infraction_df[['SanctionId','SanctionInfractionId','InfractionCategory']],on = 'SanctionId',how = 'left').dropna(subset = ['InfractionCategory','MonetaryPenalty'], how = 'all')
    # com_sac_df = (
    #     complaints_sanctions.dropna(subset = ['SanctionId','ComplaintId'])
    #     .drop_duplicates())
    # #print(com_sac_df.info())
    # com_sac_df_worst = add_target(com_sac_df, method = 'worst').dropna(subset=['SanctionLevel', 'MonetaryPenalty'], how='all')

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


    # Split the data to training and testing. In the first step we will split the data in training and remaining dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=17)

    # Carry out data balancing
    X_train, y_train = balanced_sampling(X_train,y_train,'oversample')


    if 'TF-IDF' in exp_features:
        dense_tfidf = DenseTFIDFVectorization(ngram_range=(1, 3), max_features=NUMBER_OF_TFIDF_FEATURES)
        # dtidf = dense_tfidf.fit_transform(df['concat_text'])
        # change it into target low/high
        X_0 = X_train[X_train['Target']==0]['concat_text']
        X_1 = X_train[X_train['Target']==1]['concat_text']

        X_0 = ' '.join(X_0)
        X_1 = ' '.join(X_1)

        # Take a random subsample of the words, as there is a 1,000,000 character limit to TfidfVectorizer
        X_0_split = X_0.split()
        X_1_split = X_1.split()

        if len(X_0) > 1000000:
            X_0_split = random.sample(X_0_split, 50000)
        if len(X_1) > 1000000:
            X_1_split = random.sample(X_1_split, 50000)

        # Make each group into a string again
        X_0 = ' '.join(X_0_split)
        X_1 = ' '.join(X_1_split)

        # Combine into a DataFrame to pass to the TfidfVectorizer
        X_0 = pd.DataFrame({'class_text': [X_0]})
        X_1 = pd.DataFrame({'class_text': [X_1]})

        X = pd.concat([X_0, X_1], axis=0)

        # Apply the TF-IDF transform
        dtfidf = dense_tfidf.fit_transform(X['class_text'])
        dtfidf['class'] = ['group0','group1']
        dtfidf = dtfidf.set_index(['class'])
        # Get the columns (words) with the highest TF-IDF score for each group
        dtfidf_0 = dtfidf.sort_values(by='group0', axis=1, ascending=False)
        dtfidf_1 = dtfidf.sort_values(by='group1', axis=1, ascending=False)

        # Concatenate the columns into one DataFrame
        dtfidf = pd.concat([dtfidf_0, dtfidf_1], axis=1)
        dtfidf = dtfidf.loc[:, ~dtfidf.columns.duplicated()]
        tfidf_words = dtfidf.columns

        def tfidf_word_counts(df, tfidf_words):
            """
            Counts the number of time each of the TF-IDF words occur in each complaint.
                df: The dataframe containing the complaint text (one row per complaint)
                dtfidf: the dataframe
            """
            # Count how many times each word occurs for each row (class)
            def count_words(row): 
                for word in tfidf_words:
                    row[word] = row['words'].count(word)
                return row

            df['words'] = df['concat_text'].apply(lambda x : x.split(' '))
            df = df.apply(count_words, axis=1)
            return df.drop(columns=['words'])

        X_train = tfidf_word_counts(X_train, tfidf_words).drop(['Target'],axis=1)
        if '00' in X_train.columns:
            X_train = X_train.drop(['00'],axis=1)
        print(X_train.head())
        print(X_train.info())
        X_test = tfidf_word_counts(X_test, tfidf_words).drop(['Target'],axis=1)
        if '00' in X_test.columns:
            X_test = X_test.drop(['00'],axis=1)
        print(X_test.head())
        print(X_test.info())

    print("tf-idf finished!")
 

    if 'LDA' in exp_features:
        model, corpus,id2word, bigram = TopicScores(X_train, num_topics = NUMBER_OF_LDA_TOPICS)

        print(model.print_topics(NUMBER_OF_LDA_TOPICS,num_words=15)[:10])

        train_vecs = []
        for i in range(len(X_train)):
            top_topics = model.get_document_topics(corpus[i], minimum_probability=0.0)
            topic_vec = [top_topics[i][1] for i in range(NUMBER_OF_LDA_TOPICS)]
            train_vecs.append(topic_vec)
        print(len(train_vecs))

        train_topics_df = pd.DataFrame(train_vecs, dtype = float) 
        train_topics_df.columns = pd.Index(np.arange(1,len(train_topics_df.columns)+1).astype(str))

        X_train = pd.concat([X_train,train_topics_df],axis = 1)
        
        def bigrams(words, bi_min=15, tri_min=10):
            bigram = gensim.models.Phrases(words, min_count = bi_min)
            bigram_mod = gensim.models.phrases.Phraser(bigram)
            return bigram_mod

        def get_bigram(df):
            tokenized_list = [simple_preprocess(doc) for doc in df['concat_text']]
            bigram = bigrams(tokenized_list)
            #print(bigram)
            bigram = [bigram[review] for review in tokenized_list]
            return bigram
        

        bigram_test = get_bigram(X_test)

        test_corpus = [id2word.doc2bow(text) for text in bigram_test]

        test_vecs = []
        for i in range(len(X_test)):
            top_topics = model.get_document_topics(test_corpus[i], minimum_probability=0.0)
            topic_vec = [top_topics[i][1] for i in range(NUMBER_OF_LDA_TOPICS)]
            test_vecs.append(topic_vec)

        #print(test_vecs)
        print(len(test_vecs))

        test_topics_df = pd.DataFrame(test_vecs, dtype = float).reset_index(drop=True)
        test_topics_df.columns = pd.Index(np.arange(1,len(test_topics_df.columns)+1).astype(str))
        X_test = X_test.reset_index(drop=True)
        X_test = pd.concat([X_test,test_topics_df],axis = 1)
    
    print("LDA finished!")

    def clean_dataset(df):
        assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
        df.dropna(inplace=True)
        indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
        return df[indices_to_keep].astype(np.float64)

    X_train = X_train.drop(['concat_text','Number'], axis=1)
    X_train = clean_dataset(X_train)
    X_test = X_test.drop(['concat_text','Number'], axis=1)
    X_test = clean_dataset(X_test)

    print(X_train.info())
    print(X_test.info())
    print(y_train.shape)
    print(y_test.shape)

    if(split_data):
        return X_train, X_test, y_train, y_test
    X = pd.concat([X_train, X_test], ignore_index=True)
    return X


def run_experiment(X_train, X_test, y_train, y_test, model_type="rf", experiment_name="SG hyerparameter tuning", model_output_path="sanction_gravity_model/"):
    # IMPORTANT: set the experiment name here
    mlflow.set_experiment(experiment_name=experiment_name)

    run_id = None
    with mlflow.start_run(nested=True) as experiment_run:
        # get the experiment run id
        run_id = experiment_run.info.run_id

        # train model
        m = TrainModel(X_train=X_train, y_train=y_train, model_type=model_type,hyper_params=h_params)
        model = m.train()

        # make predictions with model using test set
        predictions = model.predict(X_test)

        # encode labels to prevent misrepresentation in categorical labels
        label_encoder = LabelEncoder()
        predicted_labels = label_encoder.fit_transform(predictions)
        actual_labels = label_encoder.fit_transform(y_test)

        # get the model metrics
        model_metrics = ModelMetrics(y_true=actual_labels, y_pred=predicted_labels)

        # track the metrics in mlflow
        lr_metrics = model_metrics.regression_metrics(include_report=True, classification_metric=True)

        # track the metrics in mlflow
        model_metrics.log_metric_to_mlflow(lr_metrics)

        # track the model in mlflow
        mlflow.sklearn.log_model(model, "model")
        mlflow.set_tag("model_type", model_type)

        # save model to disk
        model_metrics.save_model_to_disk(model, file_path=model_output_path)
        model_metrics.log_model_features(exp_features)
        # track feature importance
        model_metrics.log_feature_importance(m)
    mlflow.end_run()
    return run_id


def main(data_path, model_output_path):
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
    X_train, X_test, y_train, y_test = transform_raw_data_to_features(exp_features, data=data)

    # Drop the complaint ID
    if ('ComplaintId' in X_train.columns):
        X_train.drop(["ComplaintId"],axis = 1, inplace=True)
    if ('ComplaintId' in X_test.columns):
        X_test.drop(["ComplaintId"], axis=1, inplace=True)

    experiment_id = run_experiment(X_train, X_test, y_train, y_test, model_type="random_forest", experiment_name="SG new Model", model_output_path=model_output_path)
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
    # main(data)
