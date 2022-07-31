import json 
from scipy.stats import uniform, randint

# FOLDERS
COMMUNITIES_PKL_PATH = "./model/data/datasets/communities.pkl"
ADS_DATASET_PATH = "./model/data/datasets/ads_raw.csv"
ADS_DATASET_PROCESSED_PATH = "./model/data/datasets/ads_raw_processed.csv"
COMMUNITY_DATASET_PATH = "./model/data/datasets/communities_processed.csv"
TRAIN_DATASET_PATH = "./model/data/datasets/train.csv"
TEST_DATASET_PATH = "./model/data/datasets/test.csv"
MODEL_PATH = "./model/model/model.joblib.dat"
TRAIN_SCORE_PATH = "./model/data/score/train_score.csv"
TEST_SCORE_PATH = "./model/data/score/test_score.csv"
METRICS_PATH = "./model/metrics/"
IMAGES_PATH = "./model/metrics/img/"

# Training parameters.
TEST_SIZE = 0.15
RANDOM_STATE = 23
ID_COLUMN = "ID_AD"
TARGET = 'SUSPICIOUS'

VARS_RAW_PATH = "./model/data/variables/vars_raw.json"
VARS_FILTERED_PATH = "./model/data/variables/vars_filtered.json"
METRICS_JSON_PATH = "./model/metrics/metrics.json"

with open(VARS_RAW_PATH, 'r') as file:
    vars_input = json.load(file)

with open(VARS_FILTERED_PATH, "r") as file:
    vars_final = json.load(file)

ADS_FEATURES = vars_input.get("ADS_FEATURES")
COMMUNITY_FEATURES = vars_input.get("COMMUNITY_FEATURES")
WEAK_FEATURES = vars_input.get("WEAK_FEATURES")

CRITERION = "entropy"
MAX_FEATURES = 4

# Distributions for Random search
DISTRIBUTIONS = {
    'max_depth': randint(low=4, high=20),
    'n_estimators': randint(low=20, high=300)
}

# Metrics parameters.
CUT_POINTS = [0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.97]
COLORS = ['#1A3252', '#EB5434', '#377A7B']

"""
TRAIN_DATASET_PATH = "./data/train.csv"
TEST_DATASET_PATH = "./data/test.csv"
TEST_SAMPLE_PATH = "./tests/test.json"

VARS_RAW_PATH = "./data/variables/vars_raw.json"
VARS_FILTERED_PATH = "./data/variables/vars_filtered.json"

CRITERION = "entropy"
MAX_FEATURES = 4

PRED_VARIABLE = "LABEL"F
IMAGES_PATH = "./metrics/img/"
METRICS_PATH = "./metrics/"
MODEL_PATH = "./data/model/model.joblib.dat"
METRIC_NAME = 'SUSPICIOUS'
TEST_SIZE = 0.15
RANDOM_STATE = 23
CUT_POINTS = [0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.97]
COLORS = ['#1A3252', '#EB5434', '#377A7B']

TRAIN_SCORE_PATH = "./data/train_score.csv"
TEST_SCORE_PATH = "./data/test_score.csv"

with open('./data/variables/vars_raw.json', 'r') as file:
    vars_input = json.load(file)

with open('./data/variables/vars_filtered.json', 'r') as file:
    vars_output = json.load(file)
FINAL_DATA = vars_output.get("FINAL_DATA")
MODEL_INPUT = vars_input.get("MODEL_INPUT")
CATEGORICAL_FEATURES = vars_input.get("CATEGORICAL_FEATURES")
NUMERICAL_FEATURES = vars_input.get("NUMERICAL_FEATURES")
BOOLEAN_FEATURES = vars_input.get("BOOLEAN_FEATURES")
CATEGORIES = vars_input.get("CATEGORIES")
OTHER = vars_input.get("OTHER")

# Distributions for Random search
DISTRIBUTIONS = {
    'max_depth': randint(low=1, high=10),
    'n_estimators': randint(low=20, high=300)
}

VARIABLES_INPUT = CATEGORICAL_FEATURES + NUMERICAL_FEATURES + \
    BOOLEAN_FEATURES
"""