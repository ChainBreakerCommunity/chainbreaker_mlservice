import joblib
import requests
import model.settings as st
import utils.api
from utils.env import get_config
from model.utils.data.features_ads import ads_feature_engineering
from model.utils.data.features_community import process_communities
from chainbreaker_api import ChainBreakerAdmin
from typing import List
from concurrent.futures import ThreadPoolExecutor
from logger.logger import get_logger
logger = get_logger(__name__, level = "DEBUG", stream = True)
config = get_config()

def get_model():
    model = joblib.load(st.MODEL_PATH)
    return model

def calculate_score(X, model):
    proba = model.predict_proba(X)[0][1]
    proba = round(float(proba), 6)
    return proba

def batch_classification(language: str):
    """Weekly routine that classifies unlabel data in Chainbreaker DB.

    Args:
        - country (str)

    Return
        - None

    """
    client = utils.api.get_client()
    unlabel_ads_ids = utils.api.get_unlabel_ads_ids(client, language = language)

    print("Number of ads to classify: ", len(unlabel_ads_ids))
    #joblib.dump(unlabel_ads, "unlabel_ads.pkl")
    #print(unlabel_ads)
    #return 
    communities = utils.api.get_communities(client)
    #joblib.dump(communities, "communitie.pkl")
    
    #unlabel_ads_ids = joblib.load("unlabel_ads.pkl")
    #communities = joblib.load("communities.pkl")
    index = 1
    with ThreadPoolExecutor(max_workers=5) as executor:
        for id_ad in unlabel_ads_ids:
            #args = (client, id_ad, communities, unlabel_ads_ids, )
            future = executor.submit(classify_ad, client, id_ad, communities, unlabel_ads_ids)
            print(future.result())
            index += 1
            print(f"Progress: {index/len(unlabel_ads_ids)}")

def classify_ad(client: ChainBreakerAdmin, id_ad: int, communities: List[List[int]], unlabel_ads_ids: List[int]):
    print("classify ad...")
    label = utils.api.get_ad_label(client, id_ad)

    # This ad has been classified in some point of this process.
    if label != None:
       return

    # Find community.
    community = utils.api.find_ad_community(id_ad, communities)

    # Get already label ads.
    label_ads_ids = list(set(community).difference(unlabel_ads_ids))

    # If community is suspicious set label of this ad as suspicious.
    if len(label_ads_ids) > 0:
        label = utils.api.get_ad_label(label_ads_ids[0])
        if label == 1:
            utils.api.set_labels(client, [id_ad], suspicious = label)

    # If we dont have any information of this ad, proceed to classify.
    ads_dataframe = client.get_sexual_ads_by_id(community, reduced_version=False)
    ads_dataframe = ads_feature_engineering(ads_dataframe)
    community_df = process_communities([ads_dataframe])

    # Get predictors.
    filter = st.vars_final["FINAL_DATA"]
    community_df = community_df[filter].copy()

    # Load the model and make predictions.
    model = joblib.load(st.MODEL_PATH)
    predictions = model.predict(community_df)
    label = bool(predictions[0])
    utils.api.set_labels(client, community, suspicious = label)