from typing import List
import requests
import joblib
from chainbreaker_api import ChainBreakerClient
import pandas as pd
import model.settings as st
from utils.env import get_config
from logger.logger import get_logger
config = get_config()
logger = get_logger(__name__, level = "DEBUG", stream = True)

def get_list_communities(country: str = "") -> bool:
    logger.info("Getting communities...")
    route = config["ENDPOINT"] + "/graph/get_communities"
    data = {"country": country}
    communities = requests.post(route, data = data).json()["communities"]
    joblib.dump(communities, st.COMMUNITIES_PKL_PATH)
    return communities

def get_ads_dataset(client: ChainBreakerClient, language: str, website: str) -> bool:
    logger.info("Getting ads...")
    data = client.get_sexual_ads(
        language = language, 
        website = website, 
        start_date = "0001-01-01", 
        end_date = "9999-01-01"
    )
    data.to_csv(st.ADS_DATASET_PATH, sep = ";")
    return data

def download_data(language: str, website: str, country: str) -> tuple[List, pd.DataFrame]:
    """Get list of communities and ads dataset.
    """
    client = ChainBreakerClient(config["ENDPOINT"])
    client.login(config["MAIL_USERNAME"], config["MAIL_PASSWORD"])
    communities = get_list_communities(country)
    df = get_ads_dataset(client, language = language, website = website)
    return (communities, df)

def get_data(language: str, website: str, country: str, download = False
    ) -> tuple[List, pd.DataFrame]:
    logger.info("Get data...")
    if download: 
        logger.info("Download data...")
        return download_data(language, website, country)
    else: 
        logger.info("Load data from folder...")
        communities = joblib.load(st.COMMUNITIES_PKL_PATH)
        df = pd.read_csv(st.ADS_DATASET_PATH, sep = ";")
    return (communities, df)