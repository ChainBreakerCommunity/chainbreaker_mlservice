import spacy
import en_core_web_sm
import pandas as pd
import numpy as np
import model.settings as st
from logger.logger import get_logger
from model.utils.data.search import generate_bigrams, search_sequence_of_words, search_words, get_special_bigrams
from utils.exceptions import ModelParamException

nlp = en_core_web_sm.load() 
logger = get_logger(name = __name__, level = "DEBUG", stream = True)

"""Ads feature engineering
"""

# Get third person function.
def get_third_person(text: str, bigram_window: int = 5, verbose = False) -> int:
    """
    This functions determines if the text is written in third person or not.
    Input: text (str)
    Output: Boolean (true => it is written in third person, false => is not written in third person)
    """

    # Third person pronouns.
    THIRD_PERSON = ["she", "her", "hers", "herself"]

    # Generate third-person bigrams.
    tp_bigram_one = ["new", "young", "beautiful", "different", "open-minded", 
                    "hottest", "sizzling", "sexy", "gorgeous", "few", "busty", "exotic"]
                    
    tp_bigram_two = ["girl", "girls", "ladie", "ladies", "dancer", "dancers", 
                    "chick", "chicks", "playmates", "babes"]

    TP_SPECIAL_BIGRAMS = generate_bigrams(tp_bigram_one, tp_bigram_two)
    TP_OTHER_BIGRAMS = ["all nationalities"]
    TP_SPECIAL_BIGRAMS = TP_SPECIAL_BIGRAMS + TP_OTHER_BIGRAMS

    # The text is written in third person singular or in third person plural.
    # or contains special word.
    doc = nlp(text)
    b1 = search_words(doc, THIRD_PERSON, verbose = verbose)
    b2 = get_special_bigrams(doc, TP_SPECIAL_BIGRAMS, bigram_window = bigram_window, verbose = verbose)
    #print(b1, b2, b3)
    return int(b1 or b2)

# Get first person plural function.
def get_first_person_plural(text: str, bigram_window: int = 5, verbose = False) -> int:
    """
    This functions determines if the text is written in third person or not.
    Input: text (str)
    Output: Boolean (true => it is written in third person, false => is not written in third person)
    """
    # First person plural pronouns
    FIRST_PERSON_PLURAL = ["us", "our", "ours", "ourselves"]

    # Generate third-person bigrams.
    fp_bigram_one = ["with", "message", "visit"]
    fp_bigram_two = ["us"] 
    FP_SPECIAL_BIGRAMS = generate_bigrams(fp_bigram_one, fp_bigram_two)
    FP_OTHER_BIGRAMS = ["our website"]
    FP_SPECIAL_BIGRAMS = FP_SPECIAL_BIGRAMS + FP_OTHER_BIGRAMS

    # The text is written in third person singular or in third person plural.
    # or contains special word.
    doc = nlp(text)
    b1 = search_words(doc, FIRST_PERSON_PLURAL, verbose=verbose)
    b2 = get_special_bigrams(doc, FP_SPECIAL_BIGRAMS, bigram_window = bigram_window, verbose=verbose)
    #print(b1, b2)
    return int(b1 or b2)

# Get human trafficking keywords.
def ht_find_keywords(text: str, verbose: bool = False) -> int:
    HT_KEYWORDS = ["new in town", "new girl", "came to the town", "out off town", 
               "few days",  "back to town", "out of town", "in your town", 
               "for a few days", "back in town",  "only stay for a few", "new arrived", 
               "just arrived", "new here", "new to this",  "new agency", "new in town", 
               "new to your city", "i'm new", "i am new", "brand new",  "little new",
               "very new", "look new", "short-term", "new to the area", "new to the industry", 
               "new in the business", "new face", "short time", "in town for", "new to town", 
               "short stay", "short term", "short time", "short period", "arrived in town", 
               "arrived short term", "for a limited time", "newly arrived", "passing through", 
               "on vacation", "in town for a few days", "i am a newb", "i am a newbie", "first time to", 
               "first time in", "first time here", "first time visiting", "new in the field",
               "just turned 18", "turned 18", "hot teen"]
    b1 = search_sequence_of_words(text, HT_KEYWORDS, verbose = verbose) 
    # print(b1, b2)
    return int(b1)

# Service is restricted somehow.
def service_is_restricted(text: str, verbose: bool = False) -> int:
    WITH_CONDOM_SEQUENCE = ["with condom", "use of condoms", "with a condom", "no bb"]
    RESTRICTED_SEX_SEQUENCE = ["no oral", "no anal", "no black", "no greek"]
    b1 = search_sequence_of_words(text, WITH_CONDOM_SEQUENCE, verbose = verbose) 
    b2 = search_sequence_of_words(text, RESTRICTED_SEX_SEQUENCE, verbose = verbose) 
    return int(b1 or b2)

# Determine if service is offer incall
def offer_incall(text: str, verbose: bool = False) -> bool:
    INCALL_WORDS = ["incall", "in-call", "incalls"]
    SPECIAL_INCALL_SEQUENCE = ["in call", "in call only", "in-call only", "incall only"]
    SPECIAL_NOT_INCALL_SEQUENCE = ["no incalls", "no incall", "no in-calls", "no in calls", "no incall"]

    doc = nlp(text)
    b1 = search_words(doc, INCALL_WORDS, verbose = verbose)
    b2 = search_sequence_of_words(text, SPECIAL_INCALL_SEQUENCE, verbose = verbose) 
    b3 =  search_sequence_of_words(text, SPECIAL_NOT_INCALL_SEQUENCE, verbose = verbose)
    # print(b1, b2)
    return (b1 or b2) and not b3
      
# Determine if service is offer outcall
def offer_outcall(text: str, verbose: bool = False) -> bool:
    OUTCALL_WORDS = ["outcall", "out-call", "outcalls"]
    SPECIAL_OUTCALL_SEQUENCE = ["out call", "out call only", "out-call only", "outcall only"]
    SPECIAL_NOT_OUTCALL_SEQUENCE = ["no outcall", "no out call", "no out-call"]

    doc = nlp(text)
    b1 = search_words(doc, OUTCALL_WORDS, verbose = verbose)
    b2 = search_sequence_of_words(text, SPECIAL_OUTCALL_SEQUENCE, verbose = verbose)
    b3 =  search_sequence_of_words(text, SPECIAL_NOT_OUTCALL_SEQUENCE, verbose = verbose)
    #print(b1, b2)
    return (b1 or b2) and not b3

# Get service place.
def service_place(text: str, verbose: bool = False) -> int:
    incall = offer_incall(text, verbose = verbose)
    outcall = offer_outcall(text, verbose = verbose)
    if incall and not outcall:
      return 1
    return 0
    if (incall and outcall) or outcall:
      return 0
    # Missing values.
    return -1

# Ads feature engineering.
def ads_feature_engineering(data: pd.DataFrame) -> pd.DataFrame:
    """Get ads df with new features

    """
    logger.info("Execute ads feature engineering routine...")

    # Delete rows with text missing
    data = data.dropna(axis = 0, subset = "text")
    data.columns = data.columns.str.upper()

    # Get features based on text.
    data["THIRD_PERSON"] = np.vectorize(get_third_person)(data["TEXT"])
    data["FIRST_PERSON_PLURAL"] = np.vectorize(get_first_person_plural)(data["TEXT"])
    data["HT_FIND_KEYWORDS"] = np.vectorize(ht_find_keywords)(data["TEXT"])
    data["SERVICE_IS_RESTRICTED"] = np.vectorize(service_is_restricted)(data["TEXT"])
    data["SERVICE_PLACE"] = np.vectorize(service_place)(data["TEXT"])

    # Save new dataset.
    data.to_csv(st.ADS_DATASET_PROCESSED_PATH, sep = ";")

    # Get only important columns.
    #columns = [st.ID_COLUMN] + st.ADS_FEATURES
    #new_dataset = data[columns].copy()

    # Return dataset.
    return data

def get_feature_vector(feature_dict: dict) -> list:

    # Validate input
    input_vars = set(feature_dict.keys())
    if "csrfmiddlewaretoken" in input_vars:
        input_vars.remove("csrfmiddlewaretoken")

    if input_vars != st.MODEL_INPUT:
        # Check if there is missing param
        missing = set(st.MODEL_INPUT) - set(input_vars)
        if len(missing) > 0:
            raise ModelParamException(f'Missing params: {list(missing)}')

    text = str(feature_dict["text"][0])
    data = [get_third_person(text),
            get_first_person_plural(text),
            ht_find_keywords(text),
            service_is_restricted(text),
            service_place(text)]

    # Get feature vector.
    X = pd.DataFrame(columns = st.FINAL_DATA)
    X.loc[0] = data
    return X