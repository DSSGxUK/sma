import pandas as pd
import numpy as np
from document_term_matrix import document_term_matrices


import time
import ssl
from pprint import pprint
from gensim import models
import numpy as np
from gensim import corpora
import gensim
from gensim.utils import simple_preprocess
import pandas as pd
import os
import re
import string
import time
import matplotlib.pyplot as plt


def rank_tfidf(data: pd.DataFrame, variable: str, text_column: str):
    """
    Ranks the words appearing in the 'text_column' column of the 'data' dataframe by descending TF-IDF score, grouped by the specified 'variable' variable.

    :param data: The dataframe containing the text column we want to perform TF-IDF analysis on
    :type data: pd.DataFrame
    :param variable: The variable of the dataframe to group by
    :type variable: string
    :param text_column: The name of the column to perform stemming on
    :type text_column: string
    :return: A dataframe containing the TF-IDF score of each word (token) in the text_column, ranked in descending order, broken down by category of the specified variable
    :rtype: pd.DataFrame
    """

    data = data.rename(columns={text_column:"text", variable:"category"})

    data['text'] = data['text'].astype(str)
    # Create corpus corresponding to each category by extracting the label and concatenate the strings according to category
    category_df = data.groupby(['category'], as_index = False).agg({'text': ' '.join})

    # Tokenize the docs
    tokenized_list = [simple_preprocess(doc) for doc in category_df['text']]

    # Create the Corpus and dictionary
    mydict = corpora.Dictionary()

    # The (0, 1) in line 1 means, the word with id=0 appears once in the 1st document.
    # Likewise, the (4, 4) in the second list item means the word with id 4 appears 4 times in the second document. And so on.
    mycorpus = [mydict.doc2bow(doc, allow_update=True) for doc in tokenized_list]

    # Not human readable. Convert the ids to words.
    # Notice, the order of the words gets lost. Just the word and it?s frequency information is retained.
    word_counts = [[(mydict[id], count) for id, count in line] for line in mycorpus]

    # Save the Dict and Corpus
    # mydict.save('mydict.dict')  # save dict to disk
    # corpora.MmCorpus.serialize('mycorpus.mm', mycorpus)  # save corpus to disk

    # Create the TF-IDF model
    tfidf = models.TfidfModel(mycorpus, smartirs='ntc')
    tf_idf_info = pd.DataFrame(columns = ['category', 'token', 'freq'])
    tf_idf_info['category'] = category_df['category']

    # Show the TF-IDF weights
    num = 0
    token_l = []
    freq_l = []
    for doc in mycorpus:
        token_l.append([mydict[id[0]] for id in doc])
        freq_l.append([np.around(freq[1], decimals=8) for freq in tfidf[doc]])
        num += 1

    tf_idf_info['token'] = token_l
    tf_idf_info = tf_idf_info.explode('token')
    tf_idf_info['tfidf'] = pd.Series(np.concatenate(freq_l).flat)
    tf_idf_info = tf_idf_info.drop('freq', axis = 1)
    # Rank by tfidf within each category group.
    tf_idf_info_ranked = tf_idf_info.groupby('category').apply(lambda x: x.sort_values('tfidf', ascending=False))

    return tf_idf_info_ranked


def rank_tfidf_trigram(data: pd.DataFrame, variable: str, text_column: str):
    """
    Ranks the words appearing in the 'text_column' column of the 'data' dataframe by descending TF-IDF score, grouped by the specified 'variable' variable.

    :param data: The dataframe containing the text column we want to perform TF-IDF analysis on
    :type data: pd.DataFrame
    :param variable: The variable of the dataframe to group by
    :type variable: string
    :param text_column: The name of the column to perform stemming on
    :type text_column: string
    :return: A dataframe containing the TF-IDF score of each word (token) in the text_column, ranked in descending order, broken down by category of the specified variable
    :rtype: pd.DataFrame
    """

    data = data.rename(columns={text_column:"text", variable:"category"})

    data['text'] = data['text'].astype(str)
    # Create corpus corresponding to each category by extracting the label and concatenate the strings according to category
    category_df = data.groupby(['category'], as_index = False).agg({'text': ' '.join})

    # Tokenize the docs
    tokenized_list = [simple_preprocess(doc) for doc in category_df['text']]

    # Build the bigram and trigram models
    bigram = gensim.models.Phrases(tokenized_list, min_count=5) # higher threshold fewer phrases.
    trigram = gensim.models.Phrases(bigram[tokenized_list])  

    # Faster way to get a sentence clubbed as a trigram/bigram
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    trigram_mod = gensim.models.phrases.Phraser(trigram)


    def make_bigrams(texts):
        return [bigram_mod[doc] for doc in texts]

    def make_trigrams(texts):
        words_ngram = [ [token for token in trigram_mod[bigram_mod[doc]] if '_' in token] for doc in texts ]
        return words_ngram

    
    data_words_trigrams = make_trigrams(tokenized_list)

    #print(data_words_bigrams[:1])

    # Create the Corpus and dictionary
    mydict = corpora.Dictionary()

    # # The (0, 1) in line 1 means, the word with id=0 appears once in the 1st document.
    # # Likewise, the (4, 4) in the second list item means the word with id 4 appears 4 times in the second document. And so on.
    mycorpus = [mydict.doc2bow(doc, allow_update=True) for doc in tokenized_list]
    #bicorpus = [mydict.doc2bow(doc, allow_update=True) for doc in data_words_bigrams]
    tricorpus = [mydict.doc2bow(doc, allow_update=True) for doc in data_words_trigrams]


    # # Not human readable. Convert the ids to words.
    # # Notice, the order of the words gets lost. Just the word and it?s frequency information is retained.
    word_counts = [[(mydict[id], count) for id, count in line] for line in tricorpus]


    # Create the TF-IDF model
    tfidf = models.TfidfModel(tricorpus, smartirs='ntc')
    tf_idf_info = pd.DataFrame(columns = ['category', 'token', 'freq'])
    tf_idf_info['category'] = category_df['category']

    # Show the TF-IDF weights
    num = 0
    token_l = []
    freq_l = []
    for doc in tricorpus:
        token_l.append([mydict[id[0]] for id in doc])
        freq_l.append([np.around(freq[1], decimals=4) for freq in tfidf[doc]])
        num += 1

    tf_idf_info['token'] = token_l
    tf_idf_info = tf_idf_info.explode('token')
    tf_idf_info['tfidf'] = pd.Series(np.concatenate(freq_l).flat)
    tf_idf_info = tf_idf_info.drop('freq', axis = 1)
    # Rank by tfidf within each category group.
    tf_idf_info_ranked = tf_idf_info.groupby('category').apply(lambda x: x.sort_values('tfidf', ascending=False))
    return tf_idf_info_ranked


def create_tfidf_vectorizer_df(data: pd.DataFrame, group_by: str, text_col: str, top_n: int, 
                               complaint_id_field: str = 'ComplaintId'):
    """
    Creates a dataframe with columns for the words with highest TF-IDF score in each category.
    
    :param data: The dataframe to apply this feature extraction to
    :type data: pd.DataFrame
    :param group_by: The column to group the categories by
    :type group_by: string
    :param text_col: The column containing the text to extract the features from
    :type text_col: string
    :param top_n: The number of most frequent words to consider in each category
    :type top_n: integer
    :param complaint_id_field: The name of the field containing the complaint ID
    :type complaint_id_field: string
    :return: A dataframe containing the complaint text and columns for the words with highest TF-IDF, 
             indexed by complaint ID
    :rtype: pd.DataFrame
    """
    # Get the top n words with the highest TF-IDF scores for each category
    tfidf_ranked = rank_tfidf(data, group_by, text_col)
    #print(tfidf_ranked)
    tfidf_ranked.rename(columns={'category':'cat'}, inplace=True)
    frequent_words = set(tfidf_ranked.groupby('cat').head(top_n)['token'])
    # tfidf_ranked_ngram = rank_tfidf_trigram(data, group_by, text_col)
    # tfidf_ranked_ngram.rename(columns={'category':'cat'}, inplace=True)
    # print(tfidf_ranked_ngram.groupby('cat').head(top_n)['token'])
    #frequent_words = pd.concat([tfidf_ranked.groupby('cat').head(top_n)['token'],tfidf_ranked_ngram.groupby('cat').head(top_n)['token']],axis=1)
    # print(frequent_words)
    # #frequent_words = frequent_words.loc[:, ~frequent_words.columns.duplicated()]
    # print(frequent_words)
    # word_df = []
    # for col in frequent_words.columns:
    #     this_word = frequent_words[col].values.flatten()
    #     [word_df.append(w) for w in this_word]
    # word_df = set(word_df)
    # print(word_df)

    #print(frequent_words.columns)

    ## Create a dataframe where the most frequent words are additional columns
    # Group the data by complaint ID and join all the details into one text block
    #grouped = data.groupby(complaint_id_field).agg({text_col: ' '.join})
    #grouped = data[text_col]
    
    # Create a new dataframe to hold columns for the most frequent words
    df = data[text_col].to_frame()
    for word in frequent_words:
        df[word] = np.nan
    
    return df


def word_count_vectorizer(data: pd.DataFrame, variable: str, text_col: str):
    """
    Populates the dataframe created by create_{}_vectorizer_df() with the word counts.
    
    :param data: The dataframe (created by create_{}_vectorizer_df) to apply this feature extraction to 
    :type data: pd.DataFrame
    :param variable: The column to group the categories by
    :type variable: string
    :param text_col: The column containing the text to extract the features from
    :type text_col: string
    :return: A dataframe containing the number of times each frequent word appears in the complaint, 
             indexed by complaint ID
    :rtype: pd.DataFrame
    """

    # Split the text into a list of individual words and count the words
    data['words'] = data[text_col].apply(lambda x : x.split(' '))      
    def count_words(row):
        frequent_words = data.columns.drop([text_col,'words'])
        for word in frequent_words:
            row[word] = row['words'].count(word)

        return row
    data  = data.apply(count_words, axis = 1)
    # # Add the word counts in for each of the most frequent words
    # frequent_words = data.columns.drop([text_col,'words'])
    # print(frequent_words)
    # for word in frequent_words:
    #     for i in range(len(data)):
    #         row = data.iloc[i]
    #         row[word] = row['words'].count(word)
    
    # # Tokenize the docs
    # tokenized_list = [simple_preprocess(doc) for doc in data[text_col]]

    # # Build the bigram and trigram models
    # bigram = gensim.models.Phrases(tokenized_list, min_count=5) # higher threshold fewer phrases.
    # trigram = gensim.models.Phrases(bigram[tokenized_list])  

    # # Faster way to get a sentence clubbed as a trigram/bigram
    # bigram_mod = gensim.models.phrases.Phraser(bigram)
    # trigram_mod = gensim.models.phrases.Phraser(trigram)

    # See trigram example
    #print(trigram_mod[bigram_mod[tokenized_list[0]]])

    # def make_trigrams(texts):
    #     words_ngram = [ [token for token in trigram_mod[bigram_mod[doc]] if '_' in token] for doc in texts ]
    #     return words_ngram

    # # Form trigrams
    # data_words_trigrams = make_trigrams(tokenized_list)

    # #print(data_words_bigrams[:1])

    # # Create the Corpus and dictionary
    # mydict = corpora.Dictionary()
    # tricorpus = [mydict.doc2bow(doc, allow_update=True) for doc in data_words_trigrams]


    # # # Notice, the order of the words gets lost. Just the word and it's frequency information is retained.
    # word_counts = [[(mydict[id], count) for id, count in line] for line in tricorpus]
    # #print(word_counts)
    # for i, wc in enumerate(word_counts):
    #     for ngram in wc:
    #         if ngram[0] in data.columns:
    #             data.loc[i,ngram[0]] = ngram[1]
    # data = data.fillna(0)

    return data.drop(columns=['words'])


def num_details_all(data: pd.DataFrame, complaint_id_field: str = 'ComplaintId'):
    """
    Counts the number of details associated with each complaint ID in the dataframe.

    :param data: The dataframe containing the complaint details
    :type data: pd.DataFrame
    :param complaint_id_field: The name of the field containing the complaint ID
    :type complaint_id_field: string
    :return: A Series containing the number of details in each complaint, indexed by complaint ID
    :rtype: pd.Series
    """
    return data[complaint_id_field].value_counts()


def num_details_one(data: pd.DataFrame, complaint_id: int,
                    complaint_id_field: str = 'ComplaintId'):
    """
    Counts the number of details associated with a specific complaint ID in the dataframe.

    :param data: The dataframe containing the complaint details
    :type data: pd.DataFrame
    :param complaint_id: The ID of the complaint to count the details of
    :type complaint_id: integer
    :param complaint_id_field: The name of the field containing the complaint ID
    :type complaint_id_field: string
    :return: A Series containing the number of details in each complaint, indexed by complaint ID
    :rtype: pd.Series
    """
    if complaint_id not in data[complaint_id_field].values:
        raise ValueError('The specified complaint ID does not appear in the dataframe.')

    return num_details_all(data)[complaint_id]


def num_words_all(data: pd.DataFrame, text_col: str = 'ComplaintDetail',
                  complaint_id_field: str = 'ComplaintId'):
    """
    Counts the number of words in each complaint in the dataframe.

    :param data: The dataframe containing the complaint text (details)
    :type data: pd.DataFrame
    :param text_col: The column containing the text to count the words of
    :type text_col: string
    :param complaint_id_field: The name of the field containing the complaint ID
    :type complaint_id_field: string
    :return: A Series containing the number of words in each complaint, indexed by complaint ID
    :rtype: pd.Series
    """
    data[text_col] = data[text_col].astype(str)
    # Group the data by complaint ID and join all the details into one text block
    grouped = data.groupby(complaint_id_field).agg({text_col: ' '.join})

    # Split the text into a list of individual words and count the words
    grouped['num_words'] = grouped[text_col].apply(lambda x : len(x.split(' ')))

    return grouped['num_words']


def num_words_one(data: pd.DataFrame, text_col: str, complaint_id: int,
                  complaint_id_field: str = 'ComplaintId'):
    """
    Counts the number of words in a specific complaint in the dataframe.

    :param data: The dataframe containing the complaint text (details)
    :type data: pd.DataFrame
    :param text_col: The column containing the text to count the words of
    :type text_col: string
    :param complaint_id: The ID of the complaint to count the words in
    :type complaint_id: integer
    :param complaint_id_field: The name of the field containing the complaint ID
    :type complaint_id_field: string
    :return: A Series containing the number of words in each complaint, indexed by complaint ID
    :rtype: pd.Series
    """
    if complaint_id not in data[complaint_id_field].values:
        raise ValueError('The specified complaint ID does not appear in the dataframe.')

    return num_words_all(data, text_col, complaint_id_field)[complaint_id]



def pivot_env_topics(complaints_df: pd.DataFrame, complaint_id_col: str = 'ComplaintId', 
                     env_topic_col: str = 'EnvironmentalTopic'):
    """
    Create a pivoted dataframe where each row corresponds to a complaint ID and each
    column corresponds to an environmental topic. The value is 1 if the topic is one 
    of the topics of that complaint, else 0.
    
    :param data: The dataframe containing the environmental topics of the complaints
    :type data: pd.DataFrame
    :param complaint_id_col: The column containing the complaint IDs
    :type complaint_id_col: string
    :param env_topic_col: The column containing the environmental topics
    :type env_topic_col: string
    :return: The pivoted dataframe
    :rtype: pd.DataFrame
    """
    # Restrict the dataframe to the columns of interest
    complaints_topics = complaints_df[['ComplaintId','EnvironmentalTopic']]

    # Get the list of environmental topics (without NaN)
    env_topics = complaints_topics['EnvironmentalTopic'].dropna().unique()

    # Group by ComplaintId
    grouped = complaints_topics.groupby('ComplaintId')['EnvironmentalTopic'].apply(list)

    # Create the pivot dataframe
    pivot_topics = pd.DataFrame(columns=env_topics)
    for c_id in grouped.index:
        topics_present = np.unique(grouped[c_id], return_counts=True)
        # For each topic...
        for topic in topics_present[0]:
            # ... if the topic is not NaN
            if not isinstance(topic, float):
                # ... add a row for that ComplaintId with a 1 if the topic is present.
                pivot_topics.loc[c_id, topics_present[0]] = topics_present[1]
                
    # Replace NaN entries with 0
    pivot_topics.fillna(0, inplace=True)
    
    # Change type of the 'NaN' column to integer
    #pivot_topics.iloc[:,-1] = pivot_topics.iloc[:,-1].astype(int)
    pivot_topics =  pivot_topics.loc[:, pivot_topics.columns.notnull()].reset_index().rename(columns = {'index':'ComplaintId'})
    
    return pivot_topics



def concat_complaint_details(complaints: pd.DataFrame, complaint_id_col: str = 'ComplaintId', 
                             complaint_det_col: str = 'ComplaintDetail'):
    """
    Concatenate all the complaint details into a single string per complaint ID.
    
    :param data: The dataframe containing the complaint details
    :type data: pd.DataFrame
    :param complaint_id_col: The column containing the complaint IDs
    :type complaint_id_col: string
    :param complaint_det_col: The column containing the complaint details
    :type complaint_det_col: string
    :return: A series containing the concatenated text, indexed by complaint ID
    :rtype: pd.Series
    """
    # Restrict the dataframe to the columns of interest and drop details which are NaN
    complaints_det = complaints[[complaint_id_col, complaint_det_col]].dropna()
    
    # Group by complaint ID and concatenate the complaint details into a single string
    concatenated = complaints_det.groupby('ComplaintId')['ComplaintDetail'].apply(lambda x: ', '.join(x))
    
    return concatenated
