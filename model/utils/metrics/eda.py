import pandas as pd
import sweetviz as sv
#import settings as st

TRAIN_DATASET_PATH = "./model/data/datasets/train.csv"
TARGET = 'SUSPICIOUS'

def generate_report():
    df = pd.read_csv(TRAIN_DATASET_PATH, sep = ";")
    df.drop("Unnamed: 0", axis = 1, inplace = True)
    my_report = sv.analyze(source=df,
                           target_feat= TARGET,
                           pairwise_analysis="on")
    my_report.show_html(filepath="./templates/eda.html")

if __name__ == "__main__":
    generate_report()