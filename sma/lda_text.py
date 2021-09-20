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
X = df.drop(columns=['Target'])
y = df['Target']
print(X.shape, y.shape)


# Split the data to training and testing. In the first step we will split the data in training and remaining dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=17)

print(X_test.info())
model, corpus,id2word, bigram = TopicScores(X_train, num_topics = 20)

#print(model.print_topics(20,num_words=15)[:10])

train_vecs = []
for i in range(len(X_train)):
    top_topics = model.get_document_topics(corpus[i], minimum_probability=0.0)
    topic_vec = [top_topics[i][1] for i in range(20)]
    # topic_vec.extend([rev_train.iloc[i].real_counts]) # counts of reviews for restaurant
    # topic_vec.extend([len(rev_train.iloc[i].text)]) # length review
    train_vecs.append(topic_vec)
print(len(train_vecs))
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
    print(bigram)
    bigram = [bigram[review] for review in tokenized_list]
    return bigram
  

bigram_test = get_bigram(X_test)

test_corpus = [id2word.doc2bow(text) for text in bigram_test]

test_vecs = []
for i in range(len(X_test)):
    top_topics = model.get_document_topics(test_corpus[i], minimum_probability=0.0)
    topic_vec = [top_topics[i][1] for i in range(20)]
    test_vecs.append(topic_vec)

#print(test_vecs)
print(len(test_vecs))

topics_df = pd.DataFrame(test_vecs, dtype = float).reset_index(drop=True)

print(topics_df.info())
topics_df.columns = pd.Index(np.arange(1,len(topics_df.columns)+1).astype(str))
print(X_test.dropna().info())
X_test = X_test.reset_index(drop=True)
test_df = pd.concat([X_test,topics_df],axis = 1)
print(test_df.info())
print(topics_df.tail())

#[... for inner_list in outer_list for item in inner_list]
