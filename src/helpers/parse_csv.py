from unidecode import unidecode
import pandas as pd
import numpy as np

def character_replacement(text: str):
    if(pd.isna(text)): return np.nan
    text = str(text)
    u = unidecode(text, "utf-8")
    return unidecode(u)

def normalize_ascii(dataframe: pd.DataFrame, columns: list = [], all_cols = False):
    """ Replace accented characters in a column

    :param dataframe: the dataframe from which to replace the text
    :type dataframe: pd.DataFrame
    :param columns: specifies the columns to perform replacement
    :type column: list
    :return: An ascii representation of the text
    :rtype: pd.DataFrame
    """
    if(all_cols):
        return dataframe.apply(character_replacement)
    
    for column in columns:
        dataframe[column] = dataframe[column].astype(str).apply(character_replacement)
    return dataframe.replace('nan', np.nan)
