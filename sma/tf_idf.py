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
        freq_l.append([np.around(freq[1], decimals=4) for freq in tfidf[doc]])
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

    # See trigram example
    #print(trigram_mod[bigram_mod[tokenized_list[0]]])

    def make_bigrams(texts):
        return [bigram_mod[doc] for doc in texts]

    def make_trigrams(texts):
        words_ngram = [ [token for token in trigram_mod[bigram_mod[doc]] if '_' in token] for doc in texts ]
        return words_ngram

    # Form Bigrams
    data_words_bigrams = make_bigrams(tokenized_list)
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
    #print(word_counts)

    # # Save the Dict and Corpus
    # # mydict.save('mydict.dict')  # save dict to disk
    # # corpora.MmCorpus.serialize('mycorpus.mm', mycorpus)  # save corpus to disk

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


def visualise_ranked_words(data: pd.DataFrame, ranked_df: pd.DataFrame, variable: str, text_column: str, top_n_words: int =10):
    """
    Plots the output of the rank_tfidf() function.

    :param data: The dataframe we applied rank_tfidf() to
    :type data: pd.DataFrame
    :param ranked_df: The dataframe returned by the rank_tfidf() function
    :type ranked_df: pd.DataFrame
    :param variable: The variable of the dataframe to group by
    :type variable: string
    :param text_column: The name of the column to perform stemming on
    :type text_column: string
    :param top_n_words: The number of words to display for each category on our plot
    :type top_n_words: int
    :return: A plot of the top n words by TF-IDF score, broken down by category of the variable of interest
    """

    facet_categories = data[variable].value_counts().index.tolist()
    num_facets = len(data[variable].value_counts())
    num_columns = 3
    num_rows = int(np.floor(num_facets / 3)) + 1

    fig, ax = plt.subplots(num_rows, num_columns, sharex = True,
                           figsize=(16, num_rows*4 * np.floor(top_n_words)/10))
    for i in range(num_facets):
        x = int(np.floor(i / 3))
        y = i % 3

        group = ranked_df[ranked_df['category'] == facet_categories[i]]
        
        if num_facets > 3:
            # Case where there are multiple rows of plots
            ax[x,y].barh(group['token'][:top_n_words][::-1], 
                     group['tfidf'][:top_n_words][::-1])
            ax[x,y].set_title(facet_categories[i])
            ax[x,y].tick_params(labelbottom=True)

        else:
            # Case where there is a single row of plots
            ax[y].barh(group['token'][:top_n_words][::-1], 
                     group['tfidf'][:top_n_words][::-1])
            ax[y].set_title(facet_categories[i])
            
            
    # Remove any unused subplots
    if num_facets > 3:
        # Case where there are multiple rows of plots
        if num_rows*num_columns > len(facet_categories):
            for j in range(num_rows*num_columns - len(facet_categories)):
                fig.delaxes(ax[num_rows-1, num_columns-j-1])
                
    else:
        # Case where there is a single row of plots
        if num_columns > len(facet_categories):
            for j in range(num_columns - len(facet_categories)):
                fig.delaxes(ax[num_columns-j-1])
            
    
            
        fig.suptitle('TF-IDF of the most common words for {}'.format(variable), 
                     fontsize=14, ha='left')
    plt.show()
