import pandas as pd
import numpy as np
from document_term_matrix import document_term_matrices
from tf_idf import rank_tfidf


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




def create_count_vectorizer_df(data: pd.DataFrame, variable: str, text_col: str, top_n: int, 
                               complaint_id_field: str = 'ComplaintId'):
    """
    Creates a dataframe with columns for the words appearing most often in each category.
    
    :param data: The dataframe to apply this feature extraction to
    :type data: pd.DataFrame
    :param variable: The column to group the categories by
    :type variable: string
    :param text_col: The column containing the text to extract the features from
    :type text_col: string
    :param top_n: The number of most frequent words to consider in each category
    :type top_n: integer
    :param complaint_id_field: The name of the field containing the complaint ID
    :type complaint_id_field: string
    :return: A dataframe containing the complaint text and columns for the most frequent words, 
             indexed by complaint ID
    :rtype: pd.DataFrame
    """
    # Generate the document-term matrices
    doc_term_mat = document_term_matrices(data, variable, text_col)

    # Create a set to store the top n words across all categories
    frequent_words = set()
    for key in doc_term_mat.keys():
        top_words = doc_term_mat[key].sum(axis=0).sort_values(ascending=False)[:top_n]
        [frequent_words.add(word) for word in top_words.index]
        
        
    ## Create a dataframe where the most frequent words are additional columns
    
    data[text_col] = data[text_col].astype(str)
    # Group the data by complaint ID and join all the details into one text block
    grouped = data.groupby(complaint_id_field).agg({text_col: ' '.join})
    
    # Create a new dataframe to hold columns for the most frequent words
    df = grouped[[text_col]].copy()
    for word in frequent_words:
        df[word] = np.nan
    
    return df


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
    tfidf_ranked.rename(columns={'category':'cat'}, inplace=True)
    print(tfidf_ranked.groupby('cat').head(top_n))
    print(tfidf_ranked.groupby('cat')['tfidf'].max())
    frequent_words = set(tfidf_ranked.groupby('cat').head(top_n)['token'])
        
        
    ## Create a dataframe where the most frequent words are additional columns
    
    data[text_col] = data[text_col].astype(str)
    # Group the data by complaint ID and join all the details into one text block
    grouped = data.groupby(complaint_id_field).agg({text_col: ' '.join})
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
        """
        A function to count the occurrences of each of the most frequent words in a complaint.
        """
        frequent_words = data.columns.drop([text_col,'words'])
        for word in frequent_words:
            row[word] = row['words'].count(word)
        
        return row
            
    # Add the word counts in for each of the most frequent words
    data = data.apply(count_words, axis=1)
    
    return data.drop(columns=['words'])


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
