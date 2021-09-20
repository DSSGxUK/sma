import pandas as pd
import numpy as np
import os
import sys

sma_dir = (os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')) + '/sma/')
sys.path.append(sma_dir)
from helpers.helper_function import (normalize_text, lemmatize_and_stem_text)


def num_words(df):
    """
    Return the number of words in each complaint.

    Args:
            df (pd.DataFrame): the dataframe containg the complaint details (there can be many complaint details per complaint)

    Example:
        >>> df = pd.DataFrame({"ComplaintId": [1, 1, 2], 
                                "text": ["Text of the first detail of the first complaint", 
                                        "Text of the second detail of the first complaint",
                                        "Text from the first detail of the second complaint"]})
        >>> df = num_words(df)
    """
    def word_number(text):
        if isinstance(text, float):
            return 0 # Case NaN, so length is 0
        else:
            return len(text.split(' '))
    df['num_words'] = df['concat_text'].apply(word_number)
    
    return df
def min_num_words(df, text_col: str = 'concat_text', n: int = 20):
    """
    Return whether or not the complaint contains fewer than a specified minimum number of words.

    Args:
            df (pd.DataFrame): The dataframe containg the complaint details (there can be many complaint details per complaint)
            text_col (str, optional): The name of the dataframe column containing the complaint text
            n (int, optional): The number of words to set as the minimum threshold

    Example:
        >>> df = pd.DataFrame({"ComplaintId": [1, 1, 2], 
                                "text": ["Text of the first detail of the first complaint", 
                                        "Text of the second detail of the first complaint",
                                        "Text from the first detail of the second complaint"]})
        >>> df = min_num_words(df)
    """
    def text_min_length(text):
        # Check if text is NaN
        if isinstance(text, float):
            return 0
        elif len(text.split(' ')) < n:
            return 0
        else:
            return 1

    df['min_num_words'] = df[text_col].apply(text_min_length)
    return df

def max_num_words(df, text_col: str = 'concat_text', n: int = 1500):
    """
    Return whether or not the complaint contains more than a specified maximum number of words.

    Args:
            df (pd.DataFrame): The dataframe containg the complaint details (there can be many complaint details per complaint)
            text_col (str, optional): The name of the dataframe column containing the complaint text
            n (int, optional): The number of words to set as the maximum threshold

    Example:
        >>> df = pd.DataFrame({"ComplaintId": [1, 1, 2], 
                                "text": ["Text of the first detail of the first complaint", 
                                        "Text of the second detail of the first complaint",
                                        "Text from the first detail of the second complaint"]})
        >>> df = max_num_words(df, text_col="text")
    """
    def text_max_length(text):
        # Check if text is NaN
        if isinstance(text, float):
            return 0
        elif len(text.split(' ')) >= n:
            return 1
        else:
            return 0

    df['max_num_words'] = df[text_col].apply(text_max_length)
    return df

def natural_region(df):
    """
    Return the "Natural Region" of the complaint.

    Args:
            df (pd.DataFrame): The dataframe containg the region of the complaint

    Example:
        >>> df = pd.DataFrame({"ComplaintId": [1, 2, 3], 
                                "FacilityRegion": ["Region del Maule", 
                                                    "Region Metropolitana",
                                                    "Region de los Rios"]})
        >>> df = natural_region(df)
    """
    region_groups = {'Region de Arica y Parinacota': 'Far North',
                     'Region de Tarapaca': 'Far North',
                     'Region de Antofagasta': 'Far North',
                     'Region de Atacama': 'Near North',
                     'Region de Coquimbo': 'Near North',
                     'Region de Valparaiso': 'Near North',
                     'Region Metropolitana': 'Central',
                     "Region del Libertador General Bernardo O'Higgins": 'Central',
                     'Region del Maule': 'Central',
                     'Region de Nuble': 'Central',
                     'Region del Bio-Bio': 'South',
                     'Region del Biobio': 'South',
                     'Region de la Araucania': 'South',
                     'Region de los Rios': 'South',
                     'Region de Los Rios': 'South',
                     'Region de Los Rios': 'South',
                     'Region de los Lagos': 'Austral',
                     'Region de Aysen del General Carlos Ibanez del Campo': 'Austral',
                     'Region de Magallanes y la Antartica Chilena': 'Austral'}
    def nat_reg(region):
        # Case where the region is missing (NaN)
        if isinstance(region, float) == False:
            return region_groups[region]

    df['natural_region'] = df['FacilityRegion'].apply(nat_reg)
    return df

def populated_districts(df):
    """
    Return whether the population of the district exceeds 200,000 (according to 2014 data)
    
    Args:
            df (pd.DataFrame): The dataframe containg the district of the complaint

    Example:
        >>> df = pd.DataFrame({"ComplaintId": [1, 2, 3], 
                                "FacilityDistrict": ["Arica", 
                                                    "Talca",
                                                    "Las Condes"]})
        >>> df = populated_districts(df)
    """
    populated_districts = {'Arica', 'Antofagasta', 'La Serena', 'Coquimbo', 'Viña del Mar', 
                            'Valparaíso', 'Puente Alto', 'San Bernardo', 'Santiago', 'Quilicura', 
                            'Pudahuel', 'Peñalolén', 'Nunoa', 'Maipú', 'Las Condes', 'La Pintana', 
                            'La Florida', 'Rancagua', 'Talca', 'Concepción', 'Temuco', 'Puerto Montt'}
    def high_pop(district):
        if district == district:
            if district in populated_districts:
                return 1
            else:
                return 0
        else:
            return 0 # NaN case
    df['populated_districts'] = df['FacilityDistrict'].apply(high_pop)
    return df

def facility_mentioned(df):
    """
    Return whether or not the complaint identifies a specific facility
    
    Args:
            df (pd.DataFrame): The dataframe containg the facility ID

    Example:
        >>> df = pd.DataFrame({"ComplaintId": [1, 2, 3], 
                                "FacilityId": [1432, 
                                                1567,
                                                3245]})
        >>> df = facility_mentioned(df)
    """
    def facility(facility_id):
        if facility_id != facility_id:
            # Case NaN
            return 0
        else:
            return 1        
    df['facility_mentioned'] = df['FacilityId'].apply(facility)
    return df

def month(df):
    """
    Return the month of the year when the complaint was made
    
    Args:
            df (pd.DataFrame): The dataframe containg the complaint date

    Example:
        >>> df = pd.DataFrame({"ComplaintId": [1, 2, 3], 
                                "DateComplaint": ["2020/05/01", 
                                                    "2020/09/07",
                                                    "2021/02/28"]})
        >>> df = month(df)
    """
    # Get the month of the complaint
    df['month'] = pd.to_datetime(df['DateComplaint'], format='%Y/%m/%d').apply(lambda x: x.month)
    return df

def quarter(df):
    """
    Return the quarter of the year when the complaint was made
    
    Args:
            df (pd.DataFrame): The dataframe containg the complaint date

    Example:
        >>> df = pd.DataFrame({"ComplaintId": [1, 2, 3], 
                                "DateComplaint": ["2020/05/01", 
                                                    "2020/09/07",
                                                    "2021/02/28"]})
        >>> df = quarter(df)
    """
    # Get the month of the complaint
    df['quarter'] = pd.to_datetime(df['DateComplaint'], format='%Y/%m/%d').apply(lambda x: x.quarter)
    return df

def weekday(df):
    """
    Return the day of the week when the complaint was made
    
    Args:
            df (pd.DataFrame): The dataframe containg the complaint date

    Example:
        >>> df = pd.DataFrame({"ComplaintId": [1, 2, 3], 
                                "DateComplaint": ["2020/05/01", 
                                                    "2020/09/07",
                                                    "2021/02/28"]})
        >>> df = weekday(df)
    """
    # Get the month of the complaint
    df['weekday'] = pd.to_datetime(df['DateComplaint'], format='%Y/%m/%d').apply(lambda x: x.dayofweek)
    return df

def proportion_urban(df):
    """
    Return the proportion of the surface of the district which is covered by urban zones.
    
    Args:
            df (pd.DataFrame): The dataframe containg the geographical information about the district of the complaint

    Example:
        >>> df = pd.DataFrame({"ComplaintId": [1, 2, 3], 
                                "urban_zones_km2": [24, 
                                                    123,
                                                    4], 
                                "surface_km2": [5903, 
                                                6078,
                                                4325]})
        >>> df = proportion_urban(df)
    """
    df['proportion_urban'] = df['urban_zones_km2'] / df['surface_km2']
    return df

def proportion_protected(df):
    """
    Return the proportion of the surface of the district which is covered by protected areas.
    
    Args:
            df (pd.DataFrame): The dataframe containg the geographical information about the district of the complaint

    Example:
        >>> df = pd.DataFrame({"ComplaintId": [1, 2, 3], 
                                "protected_areas_km2": [24, 
                                                        123,
                                                        4], 
                                "surface_km2": [5903, 
                                                6078,
                                                4325]})
        >>> df = proportion_protected(df)
    """
    df['proportion_protected'] = df['protected_areas_km2'] / df['surface_km2']
    return df

def proportion_poor_air(df):
    """
    Return the proportion of the surface of the district which is covered by declared areas of poor air quality.
    
    Args:
            df (pd.DataFrame): The dataframe containg the geographical information about the district of the complaint

    Example:
        >>> df = pd.DataFrame({"ComplaintId": [1, 2, 3], 
                                "declared_area_poor_air_quality_km2": [2, 
                                                                       5,
                                                                       1], 
                                "surface_km2": [5903, 
                                                6078,
                                                4325]})
        >>> df = proportion_poor_air(df)
    """
    df['declared_area_poor_air_quality_km2'] = df['declared_area_poor_air_quality_km2'] / df['surface_km2']
    return df

def filter_sanctions(complaintId, complaints_facilities, facilities_sanctions, sanctions):
    """
    Given a complaint ID, filter sanctions data to only return sanctions having occurred before 
    the date of the complaint.
    """
    empty_sanctions = pd.DataFrame(columns= sanctions.columns)
    
    if complaintId not in complaints_facilities['ComplaintId'].values:
        return empty_sanctions
    
    row = complaints_facilities.loc[complaints_facilities['ComplaintId'] == complaintId]
    if row.empty:
        return empty_sanctions
    # print(row)
    # print(row['DateComplaint'].values[0])
    date_complaint = pd.to_datetime(row['DateComplaint'].values[0])
    facility_id = pd.to_numeric(row['FacilityId'].values[0])
    #print(facility_id)
    #print(date_complaint)
    
    if not pd.notna(facility_id):
        return empty_sanctions
    
    matching_sanctions = facilities_sanctions.loc[facilities_sanctions['FacilityId'] == facility_id]['SanctionId'].values
    matching_sanctions = matching_sanctions[pd.notna(matching_sanctions)]
    #print(matching_sanctions)
    
    #get any related sanctions
    relevant_sanctions = sanctions.loc[sanctions['SanctionId'].isin(matching_sanctions)]
    #print(relevant_sanctions)
        
    #get rid of any sanctions newer than the complaint
    relevant_sanctions = relevant_sanctions.loc[pd.to_datetime(relevant_sanctions['DateBegining']) <= date_complaint]
    
    return relevant_sanctions

def get_complaint_sanctions(df, complaints_facilities, facilities_sanctions, sanctions):
    """
    Wrapper for filter_sanctions(...), which applies it to a whole dataframe of complaints.
    """
    ids = df['ComplaintId'].values
    cols = pd.Series(['ComplaintId'])
    cols = cols.append(pd.Series(sanctions.columns.values))
    total = pd.DataFrame(columns= cols)
    
    for i in ids:
        #print(i)
        infractions = filter_sanctions(i, complaints_facilities, facilities_sanctions, sanctions)
        infractions.insert(0, 'ComplaintId', i)
        total = total.append(infractions)
    return total

def num_past_sanctions(df, complaints_facilities, facilities_sanctions, sanctions):  
    """
    Return the number of previous sanctions for the facility mentioned in the complaint (if any).

    Args:
            df (pd.DataFrame): The dataframe containg the complaints
            complaints_facilities (pd.DataFrame): The dataframe containing the complaints and facilities
            facilities_sanctions (pd.DataFrame): The dataframe containing the facilities and sanctions
            sanctions (pd.DataFrame): The dataframe containing the sanctions

    Example:
        >>> df = pd.DataFrame({"ComplaintId": [1, 1, 2], 
                                "FacilityId": [1432, 
                                                4598,
                                                6554]})
        >>> complaints_facilities = pd.read_csv('complaints_facilities.csv')
        >>> facilities_sanctions = pd.read_csv('facilities_sanctions.csv')
        >>> sanctions = pd.read_csv('sanctions.csv')
        >>> df = num_past_sanctions(df, complaints_facilities, facilities_sanctions, sanctions)
    """  
    complaint_sanction_df = get_complaint_sanctions(df, complaints_facilities, facilities_sanctions, sanctions)
    num_past_sanctions = complaint_sanction_df.groupby('ComplaintId')['SanctionId'].unique().apply(len)
    num_past_sanctions = num_past_sanctions.rename('num_past_sanctions')
    df = complaints_facilities.merge(num_past_sanctions, how='left', on='ComplaintId')
    df['num_past_sanctions'] = df['num_past_sanctions'].fillna(0)
    return df['num_past_sanctions']

def total_past_fines(df, complaints_facilities, facilities_sanctions, sanctions):
    """
    Return the total sum of fines handed out to the facility mentioned in the complaint (if any).

    Args:
            df (pd.DataFrame): The dataframe containg the complaints
            complaints_facilities (pd.DataFrame): The dataframe containing the complaints and facilities
            facilities_sanctions (pd.DataFrame): The dataframe containing the facilities and sanctions
            sanctions (pd.DataFrame): The dataframe containing the sanctions

    Example:
        >>> df = pd.DataFrame({"ComplaintId": [1, 1, 2], 
                                "FacilityId": ["1432", 
                                                "4598",
                                                "6554"]})
        >>> complaints_facilities = pd.read_csv('complaints_facilities.csv')
        >>> facilities_sanctions = pd.read_csv('facilities_sanctions.csv')
        >>> sanctions = pd.read_csv('sanctions.csv')
        >>> df = total_past_fines(df, complaints_facilities, facilities_sanctions, sanctions)
    """  
    complaint_sanction_df = get_complaint_sanctions(df, complaints_facilities, facilities_sanctions, sanctions)
    sum_mon_pen = complaint_sanction_df.groupby('ComplaintId')['MonetaryPenalty'].apply(sum)
    total_past_fines = sum_mon_pen.rename('total_past_fines')
    df = complaints_facilities.merge(total_past_fines, how='left', on='ComplaintId')
    df['total_past_fines'] = df['total_past_fines'].fillna(0)
    return df['total_past_fines']

def ComplaintType_archivo1(df):
    """
    Return whether or not the complaint was made by a citizen (as Archivo I complaints almost always come from citizens).
    
    Args:
            df (pd.DataFrame): The dataframe containg the complaint type

    Example:
        >>> df = pd.DataFrame({"ComplaintId": [1, 2, 3], 
                               "ComplaintType": ["Ciudadana", 
                                                 "Sectorial",
                                                 "Ciudadana"]})
        >>> df = ComplaintType_archivo1(df)
    """
    def complaint_type(complaint_type):
        if complaint_type != complaint_type:
            # Case NaN
            return 0
        elif complaint_type == 'Ciudadana':
            return 1
        else:
            return 0     
    df['ComplaintType_archivo1'] = df['ComplaintType'].apply(complaint_type)
    return df

# def remove_missing_env_topics(df):
#     """
#     Return whether or not the complaint mentions any environmental topics.
    
#     Args:
#             df (pd.DataFrame): The dataframe containg the environmental topics

#     Example:
#         >>> df = pd.DataFrame({"ComplaintId": [1, 1, 2], 
#                                 "EnvironmentalTopic": ["Ruidos", 
#                                                     "Olor",
#                                                     "Vectores"]})
#         >>> df = populated_districts(df)
#     """
#     df['has_env_topic'] = [0 if df['EnvironmentalTopic'].isnull() else 1]
#     return df

# def archivo1_words(df):
#     archivo1_words = ['camion','etern','feo','noch','alarm','carabiner','peatonal','vehicul','voz']
#     def contains_archivo1_words(text):
#         i = 0
#         for word in archivo1_words:
#             if word in text:
#                 i += 1
#         return i

#     text = df['concat_text'].apply(normalize_text).apply(lemmatize_and_stem_text)
#     df['archivo1_words'] = text.apply(contains_archivo1_words)
#     return df
