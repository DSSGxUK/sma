from feature_union import FeatureUnification
from model_metrics import ModelMetrics
from rake_extraction import DenseRakeFeatures
from tfidf_transform import DenseTFIDFVectorization
from train_model import TrainModel
from topic_models import TopicScores
from sklearn.preprocessing import OneHotEncoder
import mlflow
import pandas as pd

import pandas as pd
from topic_models import *
from sklearn.model_selection import train_test_split
import gensim
import gensim.corpora as corpora
from gensim.models.phrases import Phrases, Phraser
from gensim.utils import simple_preprocess
import numpy as np
df = pd.read_csv('/home/desktop0/files/cleaned_df.csv')
print(df.info())

# Define the inputs and target
#X = df.drop(columns=['Target'])
X = df
y = df['Target']
print(X.shape, y.shape)


# Split the data to training and testing. In the first step we will split the data in training and remaining dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=17)

dense_tfidf = DenseTFIDFVectorization(ngram_range=(1, 3), max_features=20)
# dtidf = dense_tfidf.fit_transform(df['concat_text'])
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

X_train = tfidf_word_counts(X_train, tfidf_words)
print(X_train.info())
print(X_train.head())
X_test = tfidf_word_counts(X_test, tfidf_words)
print(X_test.info())
print(X_test.head())