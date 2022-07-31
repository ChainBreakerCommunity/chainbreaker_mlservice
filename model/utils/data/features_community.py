import en_core_web_sm
import pandas as pd
from typing import List
import numpy as np
import model.settings as st
nlp = en_core_web_sm.load() #spacy.load("en_core_web_sm")
from logger.logger import get_logger
logger = get_logger(__name__, "DEBUG", stream = True)

"""Communities feature engineering
"""

def get_maximum_distance(df: pd.DataFrame) -> float:
    maximum_distance = 1
    for i in range(int(len(df) / 2)):
      index1 = np.random.choice(range(0, len(df)))
      index2 = np.random.choice(range(0, len(df)))
      text1 = nlp(df.iloc[index1]["TEXT"])
      text2 = nlp(df.iloc[index2]["TEXT"])
      distance = text1.similarity(text2)

      if distance < maximum_distance:
         maximum_distance = distance
    return maximum_distance

def several_regions(df: pd.DataFrame) -> bool:
    if len(df["REGION"].unique()) > 1:
       return 1
    return 0

def several_cities(df: pd.DataFrame) -> bool:
    if len(df["CITY"].unique()) > 1:
       return 1
    return 0

def get_community_normalized_size(df: pd.DataFrame, num_communities: int) -> float:
    return len(df) / (max(num_communities - 1))

def community_contains_third_person(df: pd.DataFrame) -> bool:
    values = df["THIRD_PERSON"].value_counts()
    if 1 in values:
      return 1
    return 0

def community_contains_first_plural_person(df: pd.DataFrame) -> bool:
    values = df["FIRST_PERSON_PLURAL"].value_counts()
    if 1 in values:
      return 1
    return 0

def community_service_place(df: pd.DataFrame) -> bool:
    values = df["SERVICE_PLACE"].value_counts()
    # Only Incall !
    if 1 in values:
      return 1
    return 0
    # Outcall / Incall
    if 0 in values: 
      return 1
    # Missing values.
    return 0

def number_of_ethnicities(df: pd.DataFrame) -> bool:
    values = df["ETHNICITY"].unique()
    if 8 in values:
      values = values[np.where(values != 8)]
    return len(values)

def human_trafficking_keywords(df: pd.DataFrame) -> bool:
    values = df["HT_FIND_KEYWORDS"].value_counts()
    if 1 in values:
      return 1
    return 0

def service_is_restricted(df: pd.DataFrame) -> bool:
    # If it is restricted just by one, return 1
    values = df["SERVICE_IS_RESTRICTED"].value_counts()
    if 1 in values:
       return 1
    return 0

def community_age(df: pd.DataFrame) -> int:
    ages = df["AGE"].values
    #ages = ages[np.where(ages != np.nan)]
    try:
      return int(ages.min())
    except: 
      return 0

def get_communities_dataframes(dataset: pd.DataFrame, communities: List) -> List[pd.DataFrame]:
    """Create a dataframe for each of the communities and save it in an array.
    """
    dataset.columns = dataset.columns.str.upper()
    dataset[st.ID_COLUMN] = dataset[st.ID_COLUMN].astype(str)
    dataframes = list()
    for i in range(len(communities)):
      sub_df = dataset[dataset[st.ID_COLUMN].isin(communities[i])]
      sub_df = sub_df.reset_index()
      sub_df.drop("index", axis = 1, inplace = True)
      dataframes.append(sub_df)
    return dataframes

def communities_feature_engineering(communities: List, ads_dataframe: pd.DataFrame) -> tuple[pd.DataFrame, List[pd.DataFrame]]:

    """Get communities dataframe.
    """
    logger.warning("Execute communities feature engineering routine...")

    # Get communities dataframe.
    dataframes = get_communities_dataframes(ads_dataframe, communities)
    return process_communities(dataframes), dataframes


def process_communities(dataframes: List[pd.DataFrame]):

    columns = st.COMMUNITY_FEATURES
    dataset_communities = pd.DataFrame(columns = columns)

    for i in range(len(dataframes)):
        community_df = dataframes[i]
        # text_similarity = get_maximum_distance(community_df)
        several_re = several_regions(community_df)
        several_ci = several_cities(community_df)
        # community_size = get_community_normalized_size(community_df, len(dataframes))
        third_person = community_contains_third_person(community_df)
        plural_person = community_contains_first_plural_person(community_df)
        service_place = community_service_place(community_df)
        ht_words = human_trafficking_keywords(community_df)
        ethnicities_number = number_of_ethnicities(community_df)
        service_restricted = service_is_restricted(community_df)
        min_age = community_age(community_df)
        data = [several_re, several_ci, third_person, plural_person, 
                service_place, ht_words, ethnicities_number, service_restricted, min_age]
        df = pd.DataFrame(columns = columns)
        df.loc[0] = data
        dataset_communities = pd.concat([dataset_communities, df])

    # Save dataset in csv.
    dataset_communities.to_csv(st.COMMUNITY_DATASET_PATH, sep = ";")
    return dataset_communities