# import logging

# import gensim
# import gensim.corpora as corpora
# import pandas as pd
# from gensim.models.phrases import Phrases, Phraser
# from gensim.utils import simple_preprocess

# logger = logging.getLogger(__name__)
# logger.setLevel(logging.WARNING)
# logging.getLogger('gensim').setLevel(logging.ERROR)

# # Constants
# MALLET_PATH = '/files/mallet/mallet-2.0.8/bin/mallet'


# def TopicScores(df, num_topics, include_bigrams=False, bigram_min_count=5, bigram_threshold=2):
#     data = df[['ComplaintId','concat_text']]
#      # Tokenize the docs
#     tokenized_list = [simple_preprocess(doc) for doc in data['concat_text']]
#     # Create the Corpus and dictionary
#     id2word = corpora.Dictionary()

#     if(include_bigrams):
#         bigrams = Phrases(tokenized_list, min_count=bigram_min_count, threshold=bigram_threshold)
#         tokenized_list = [bigrams[doc] for doc in tokenized_list]
#         id2word = corpora.Dictionary(tokenized_list)

#     data_corpus = [id2word.doc2bow(doc, allow_update=True) for doc in tokenized_list]
#     model = gensim.models.wrappers.LdaMallet(MALLET_PATH, corpus=data_corpus, num_topics=num_topics, id2word=id2word)

#     # Init output
#     sent_topics_df = pd.DataFrame()

#     # Get topic in each document
#     for i, row in enumerate(model[data_corpus]):
#         cid = df[['ComplaintId']].iloc[i,0]
#         # Get the topics, Perc Contribution and for each document
#         for j, (topic_num, prop_topic) in enumerate(row):
#             sent_topics_df = sent_topics_df.append(pd.Series([i, cid, int(topic_num), prop_topic]), ignore_index=True)
#     sent_topics_df.columns = ['row_number','ComplaintId','Topic', 'Topic_Contribution']
#     sent_topics_df = sent_topics_df.pivot(index="ComplaintId", columns="Topic", values="Topic_Contribution").reset_index()
#     sent_topics_df['ComplaintId'] = sent_topics_df['ComplaintId'].astype(int)
#     return sent_topics_df

import logging

import gensim
import gensim.corpora as corpora
import pandas as pd
from gensim.models.phrases import Phrases, Phraser
from gensim.utils import simple_preprocess

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)
logging.getLogger('gensim').setLevel(logging.ERROR)

# Constants
MALLET_PATH = '/files/mallet/mallet-2.0.8/bin/mallet'

def bigrams(words, bi_min=15, tri_min=10):
    bigram = gensim.models.Phrases(words, min_count = bi_min)
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    return bigram_mod
    
def TopicScores(df, num_topics, include_bigrams=False, bigram_min_count=5, bigram_threshold=2):
    data = df[['ComplaintId','concat_text']]
    # Tokenize the docs
    tokenized_list = [simple_preprocess(doc) for doc in data['concat_text']]
    bigram = bigrams(tokenized_list)
    tokenized_list = [bigram[doc] for doc in tokenized_list]
    # Create the Corpus and dictionary
    id2word = corpora.Dictionary(tokenized_list)
    id2word.filter_extremes(no_below=10, no_above=0.35)
    id2word.compactify()
    corpus = [id2word.doc2bow(text,allow_update=True) for text in tokenized_list]
    #return corpus, id2word, bigram

    # if(include_bigrams):train_vecs = []
    #     bigrams = Phrases(tokenized_list, min_count=bigram_min_count, threshold=bigram_threshold)
    #     tokenized_list = [bigrams[doc] for doc in tokenized_list]
    #     id2word = corpora.Dictionary(tokenized_list)

    #model = gensim.models.wrappers.LdaMallet(MALLET_PATH, corpus=corpus, num_topics=num_topics, id2word=id2word)
    model = gensim.models.ldamulticore.LdaMulticore(
                           corpus=corpus,
                           num_topics=num_topics,
                           id2word=id2word,
                           chunksize=100,
                           workers=4, # Num. Processing Cores - 1
                           passes=50,
                           eval_every = 1,
                           per_word_topics=True)
    return model, corpus, id2word, bigram 

    # for i in range(len(rev_train)):
    #     top_topics = lda_train4.get_document_topics(train_corpus4[i], minimum_probability=0.0)
    #     topic_vec = [top_topics[i][1] for i in range(20)]
    #     topic_vec.extend([rev_train.iloc[i].real_counts]) # counts of reviews for restaurant
    #     topic_vec.extend([len(rev_train.iloc[i].text)]) # length review
    #     train_vecs.append(topic_vec)

    # # Init output
    # sent_topics_df = pd.DataFrame()

    # # Get topic in each document
    # for i, row in enumerate(model[corpus]):
    #     cid = df[['ComplaintId']].iloc[i,0]
    #     # Get the topics, Perc Contribution and for each document
    #     for j, (topic_num, prop_topic) in enumerate(row):
    #         sent_topics_df = sent_topics_df.append(pd.Series([i, cid, int(topic_num), prop_topic]), ignore_index=True)
    # sent_topics_df.columns = ['row_number','ComplaintId','Topic', 'Topic_Contribution']
    # sent_topics_df = sent_topics_df.pivot(index="ComplaintId", columns="Topic", values="Topic_Contribution").reset_index()
    # sent_topics_df['ComplaintId'] = sent_topics_df['ComplaintId'].astype(int)
    # return sent_topics_df


