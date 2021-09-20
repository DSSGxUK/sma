from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import pandas as pd

def OneHotEncode(df: list):
    """
    Generates one hot encoded matrix for each category of 'EnvironmentalTopic'.

    :param df: The dataframe containing the encoded column for which we want to generate the one hot encoded matrix
    :type data: pd.DataFrame
    :param encoded_column: The name of the column to perform encoding on
    :type encoded_column: string
    :return: array containing encoded variable values, and array containing variable names
    :rtype: array
    """
    # impute missing data
    imputer = SimpleImputer(strategy='constant', fill_value='Missing')
    target_col = df.copy()
    imputed_target_col = imputer.fit_transform(target_col)
    # Create instance of one hot encoder

    one_hot_encoder = OneHotEncoder(handle_unknown = 'error', sparse = False, drop = ['Missing'])
    # Apply one-hot encoder 

    one_hot_encoder_col = one_hot_encoder.fit_transform(imputed_target_col)

    one_hot_encoder_col_name = one_hot_encoder.get_feature_names(['EnvironmentalTopic'])
    
    return one_hot_encoder_col, one_hot_encoder_col_name