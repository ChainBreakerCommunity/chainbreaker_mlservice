from typing import List
import joblib
import pandas as pd
import model.settings as st
from model.utils.data.pull import get_data
from model.utils.data.features_ads import ads_feature_engineering
from model.utils.data.features_community import communities_feature_engineering
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss
from collections import Counter
from sklearn.model_selection import train_test_split
from logger.logger import get_logger

logger = get_logger(name = __name__, level = "DEBUG", stream = True)

def oversampling(X, Y):
    """Oversampling using SMOTE.
    """
    logger.info("Oversample training dataset using SMOTE")
    sm = SMOTE(random_state = st.RANDOM_STATE)
    X_res, Y_res = sm.fit_resample(X, Y)
    return X_res, Y_res

def undersampling(X, Y):
    counter = Counter(Y)
    logger.info(counter)
    undersample = NearMiss(version = 1, n_neighbors=3)
    X_res, Y_res = undersample.fit_resample(X, Y)
    counter = Counter(Y_res)
    logger.info(counter)
    return X_res, Y_res


def split_dataset(data: pd.DataFrame):
    logger.info("Split dataset...")

    # Get variables.
    X = data[st.COMMUNITY_FEATURES]
    Y = data[st.TARGET]

    # Split dataset.
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size = st.TEST_SIZE, stratify = Y, random_state = st.RANDOM_STATE
    )

    return X_train, Y_train, X_test, Y_test

def weak_supervision(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Get labels with weak supervision...")
    labels = list()
    def mark_flag(row, features: List[str]):
        for feature in features:
            flag = (row[feature] == 1)
            if flag:
               return 1
        return 0
    for i in range(len(df)):
        row = df.iloc[i]
        labels.append(mark_flag(row, st.WEAK_FEATURES))
    df[st.TARGET] = labels
    return df

def get_model_data(language: str, website: str, country: str, download: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:

    # Get model data.
    #communities, df = get_data(language = language, website = website, country = country, download = download)
    communities = joblib.load(st.COMMUNITIES_PKL_PATH)
    #df = pd.read_csv(st.ADS_DATASET_PATH, sep = ";")
    #print(df["text"])

    # Ads feature engineering.
    #ads_dataframe = ads_feature_engineering(df)

    ads_dataframe = pd.read_csv(st.ADS_DATASET_PROCESSED_PATH, sep = ";")

    # Get communities and get features.
    df, _ = communities_feature_engineering(communities, ads_dataframe)

    # Get labels.
    df = weak_supervision(df)

    # Drop rows with missing values.
    df = df.dropna()
    logger.info(f"Dataframe final shape: {df.shape}")

    # Split data into train and test.
    X_train, Y_train, X_test, Y_test = split_dataset(df)

    # Oversample training data using SMOTE.
    X_train, Y_train = undersampling(X_train, Y_train)

    # Join target.
    X_train[st.TARGET] = Y_train
    X_test[st.TARGET] = Y_test

    # Save training and testing data.
    logger.info("Saving training dataset and testing dataset...")
    X_train.to_csv(st.TRAIN_DATASET_PATH, sep = ";")
    X_test.to_csv(st.TEST_DATASET_PATH, sep = ";")

    return X_train.drop(st.TARGET, axis = 1), Y_train, X_test.drop(st.TARGET, axis = 1), Y_test