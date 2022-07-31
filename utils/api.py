
from chainbreaker_api import ChainBreakerAdmin
import requests
from utils.env import get_config
from typing import List
config = get_config()

def get_client() -> ChainBreakerAdmin:
    """Connect to chainbreaker.

    Args:
        - None

    Return:
        - ChainBreakerAdmin

    """
    client = ChainBreakerAdmin(config["ENDPOINT"])
    client.login(config["MAIL_USERNAME"], config["MAIL_PASSWORD"])
    return client

def get_unlabel_ads_ids(client: ChainBreakerAdmin, language: str) -> list:
    """Get unlabel ads

    Args:
        - client (ChainBreakerAdmin)
        - language (str)

    Return:
        - 

    """
    headers = {"x-access-token": client.get_token()}
    route = config["ENDPOINT"] + "/data/get_unlabel_ads"
    print("route: ", route)
    data = {"language": "english"}
    ads = requests.post(route, data = data, headers = headers).json()["ads"]
    return ads

def get_communities(client: ChainBreakerAdmin, country: str = "") -> List[List[int]]:
    """Get communities

    Args:
        - client (ChainBreakerAdmin)
        - country (str)

    Return
        - List[List[int]]
    
    """
    communities = client.get_communities(country = country)
    return communities

def find_ad_community(id_ad: int, communities: List[List[int]]) -> List[int]:
    for community in communities: 
        if id_ad in community:
            return community
    return [id_ad]
        
def get_ad_label(client: ChainBreakerAdmin, id_ad: int) -> int:
    label = client.get_sexual_ads_by_id([id_ad], reduced_version = False).iloc[0]["score_risk"]
    return label

def set_labels(client: ChainBreakerAdmin, id_ads: List[int], suspicious: bool):
    """Set label of ad.
    """
    headers = {"x-access-token": client.get_token()}
    route = config["ENDPOINT"] + "/data/set_labels"
    data = {int(suspicious): id_ads, int(not suspicious): []}
    res = requests.post(route, data = data, headers = headers)
    return res.status_code == 200