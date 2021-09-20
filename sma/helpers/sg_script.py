#Import libraries
import pandas as pd
import numpy as np# Read in data as pandas dataframe and display first 5 rows

import sys, os
#first change the cwd to the script path
scriptPath = os.path.realpath(os.path.dirname(sys.argv[0]))
os.chdir(scriptPath)
#append the relative location you want to import from
sys.path.append("../src")
# Gensim
import gensim
import gensim.corpora as corpora
import spacy

helpers_dir = (os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) + '/helpers/')
sys.path.append(helpers_dir)
sma_dir = (os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')) + '/sma/')
sys.path.append(sma_dir)
sma_project_dir = (os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')) + '/sma-project/')
sys.path.append(sma_project_dir)
from text_cleaning import *
from tf_idf import *
from document_term_matrix import *
from feature_extraction import *
from random import randrange
import matplotlib.pyplot as plt  
import seaborn 
import sklearn 
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
from sklearn.ensemble import RandomForestRegressor

from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter
from scipy import stats
import smogn


# Functions


# Create Monetary penalty - target variable
def add_label (row):
    if pd.isnull(row['MonetaryPenalty']):
        if pd.isnull(row['InfractionCategory']):
              return 0.1
        elif row['InfractionCategory'] == 'Leves' :
              return randrange(1,1000)
        elif row['InfractionCategory'] == 'Graves' :
              return randrange(1001,5000)
        else:
            #row['InfractionCategory'] == 'Gravísimas':
            return randrange(5001, 10000)

    return row['MonetaryPenalty']


 # Import and create target datasets
a = pd.read_csv('/home/desktop0/files/data_merge/sanctions_registry.csv', encoding = "ISO-8859-1")
com_reg = pd.read_csv('/home/desktop0/files/data_merge/complaint_registry.csv', encoding = "ISO-8859-1").drop(['Unnamed: 0','Unnamed: 0.1'],1)

#com_topic = pd.read_csv('/home/desktop0/files/data/data/raw/complaints_topic.csv', encoding = 'ISO-8859-1')
#com_df = pd.read_csv('/home/desktop0/files/data/data/raw/complaints_detail.csv', encoding = 'ISO-8859-1')
com_fac = pd.read_csv('/home/desktop0/files/data_merge/complaints_facilities_registry.csv', encoding = 'ISO-8859-1')
com_sac_reg = pd.read_csv('/home/desktop0/files/data_merge/complaints_sanctions_registry.csv', encoding = "ISO-8859-1")


#com_df = pd.merge(com_df, com_reg[['ComplaintId','ComplaintStatus', 'ComplaintType', 'EndType', 'Clasification']], 
#              on = 'ComplaintId', how = 'outer')
com_df = com_reg[['ComplaintId','ComplaintDetail','Number','ComplaintType','EndType','EnvironmentalTopic']]
            
com_df = com_df.merge(com_fac[['ComplaintId','FacilityId','FacilityRegion','FacilityDistrict','FacilityEconomicSector','FacilityEconomicSubSector'
]], how = 'left')

com_df['ComplaintDetail'] = com_df['ComplaintDetail'].astype(str)
#print(com_df.shape)


com_sac_df = (
    com_sac_reg
    .dropna(subset = ['SanctionId'])
    .loc[:,['ComplaintId','SanctionId','MonetaryPenalty']]
    .merge(a[['SanctionId','SanctionInfractionId','InfractionCategory']], on = 'SanctionId', how = 'outer')
    .dropna(subset = ['SanctionId','ComplaintId'])
    .drop_duplicates())

com_sac_df['ComplaintId'] = com_sac_df['ComplaintId'].astype(int)
com_sac_df['SanctionId'] = com_sac_df['SanctionId'].astype(int)
print(com_sac_df.info())



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


# Delete observations out of SMA's remit(outliers).
to_dropped = com_agg.loc[(com_agg['EndType'] == 'DerivaciÃ³n Total a Organismo Competente') | (com_agg['EndType'] == 'Archivo I')]
com_sac_df['Target'] = com_sac_df.apply (lambda row: label_money(row), axis=1)

# # # Take the sum of each complaintID group penalty value as the target variable.
# com_sanc_sample_sum = (
#     com_sac_df.groupby('ComplaintId')
#     .Target
#     .sum()
#     .reset_index()
#     .merge(com_agg, how = 'left'))
#     #.drop(to_dropped.index, axis = 0))

# com_sanc_sample_sum['Target']=com_sanc_sample_sum['Target'].replace(0, 0.1)
# print(com_sanc_sample_sum['Target'].describe())

com_sanc_sample_avg = (
    com_sac_df
    .groupby('ComplaintId')
    .Target
    .mean()
    .reset_index()
    .merge(com_agg.drop(to_dropped.index, axis = 0), how = 'left'))

com_sanc_sample_avg['Target']=com_sanc_sample_avg['Target'].replace(0, 0.1)
print(com_sanc_sample_avg['Target'].describe())
print(com_sanc_sample_avg.head())

# Conduct text cleaning on ComplaintDetail
com_sanc_sample_avg= (
    com_sanc_sample_avg.pipe(clean_text, text_column = 'ComplaintDetail')
    .pipe(lemmatize, text_column = 'cleaned_text')
    .pipe(stemmer, text_column = 'cleaned_text')
    .drop(['cleaned_text','ComplaintDetail','EndType'], axis = 1)
    .rename(columns = {'stemmed':'concat_text'})
    .dropna(subset = ['Number'])
)


# com_sanc_sample_sum= (
#     com_sanc_sample_sum.pipe(clean_text, text_column = 'ComplaintDetail')
#     .pipe(lemmatize, text_column = 'cleaned_text')
#     .pipe(stemmer, text_column = 'cleaned_text')
#     .drop(['cleaned_text','ComplaintDetail','EndType'], axis = 1)
#     .rename(columns = {'stemmed':'concat_text'})
# )


#print(com_sanc_sample_sum.info())
print(com_sanc_sample_avg.info())

# Prepare samples from Archivo II  

# archivo2_sample = (
#     com_agg.loc[com_agg['EndType'] == 'Archivo II']
#     #.sample(n=400)
# )

# # Conduct text cleaning on ComplaintDetail
# archivo2_sample = (
#     archivo2_sample.pipe(clean_text, text_column = 'ComplaintDetail')
#     .pipe(lemmatize, text_column = 'cleaned_text')
#     .pipe(stemmer, text_column = 'cleaned_text')
#     .drop(['cleaned_text','ComplaintDetail','EndType'], axis = 1)
#     .rename(columns = {'stemmed':'concat_text'})
#     .assign(Target = 0)
# )
from pandas.api.types import CategoricalDtype

from pandas.api.types import CategoricalDtype
def add_target(df, method = 'worst'):
    #df = df.dropna(subset = ['InfractionCategory'])
    level_type = CategoricalDtype(categories= ['Leves','Graves','Gravísimas'], ordered=True)
    df['SanctionLevel'] = df['InfractionCategory'].astype(level_type)
    if method == 'worst':
        tfidf_df = df.groupby('ComplaintId', as_index=False).SanctionLevel.max()
        return tfidf_df.dropna()
    if method == 'most':
        tfidf_df = (
            df.groupby(['ComplaintId','SanctionLevel'], as_index=False)['SanctionInfractionId']
            .count()
            .sort_values(['ComplaintId','SanctionInfractionId','SanctionLevel'], ascending=[True,False,False])
            )
        tfidf_df = (
            tfidf_df[tfidf_df['SanctionInfractionId'] != 0].groupby(['ComplaintId'])
            .head(1)
            .reset_index(drop=True)
            .loc[:,['ComplaintId','SanctionLevel']])
        return tfidf_df

tfidf_sum = tfidf_target(com_sac_df, method ='worst')
tfidf_avg = tfidf_target(com_sac_df, method ='most')


# print(tfidf_sum.info())
# print(tfidf_avg.info())
# print(com_with_sanc_sample.info())
#archivo2_sample.to_csv('/home/desktop0/files/archivo2_sample.csv', index=False)
com_sanc_sample_avg.to_csv('/home/desktop0/files/com_with_sanc_sample_avg_log.csv',index = False)
#com_sanc_sample_sum.to_csv('/home/desktop0/files/com_with_sanc_sample_sum_log.csv',index = False)

tfidf_sum.to_csv('/home/desktop0/files/tfidf_sum.csv', index=False)
tfidf_avg.to_csv('/home/desktop0/files/tfidf_avg.csv',index=False)

