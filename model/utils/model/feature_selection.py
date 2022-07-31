
from tkinter import Y
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel
import matplotlib.pyplot as plt
import pandas as pd
import model.settings as st
import json
from logger.logger import get_logger
logger = get_logger(name = __name__, level = "DEBUG", stream = True)

def filter_non_correlated_predictors(data: pd.DataFrame) -> set:
    corr = data.corr()
    cor_target = abs(corr[st.TARGET])
    relevant_features = cor_target[cor_target > 0.2]
    names = [index for index, value in relevant_features.iteritems()]
    names.remove(st.TARGET)
    names = set(names)
    return names


def feature_selection(X_train: pd.DataFrame, Y_train: pd.Series, X_test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Feature selection using Support Vector Machine with Linear Kernel
    K(x, y) = <x, y>_{R^d}
    with Lasso Regression (L1) for feature selection.
    """
    # Load dataset.
    #data = pd.read_csv(train_dataset_path)
    #data.drop("Unnamed: 0", axis = 1, inplace = True)
    #X_train = data.drop(st.METRIC_NAME, axis = 1)
    #Y_train = data[st.METRIC_NAME]
    data = X_train.copy()
    data[st.TARGET] = Y_train

    # Select L1 regulated features from LinearSVC output 
    selection = SelectFromModel(LinearSVC(C=1, penalty='l1', dual=False))
    selection.fit(X_train, Y_train)

    # Get best features.
    feature_names = data.drop(st.TARGET, axis = 1).columns[(selection.get_support())]
    feature_names = list(feature_names)

    # Intersection.
    #feature_names = list(feature_names.intersection(names))
    logger.warning(f"Selected variables:\n {feature_names}")

    # Save final features.
    with open(st.VARS_FILTERED_PATH, "w") as f:
        json.dump(
            {
                "FINAL_DATA": feature_names
            }, 
            f, indent=4
        )
    return X_train[feature_names], X_test[feature_names]

def get_feature_importance(model, X_test):
    # Plot feature importance
    plt.figure(figsize=(15, 12))
    feat_importances = pd.Series(model.feature_importances_, index= X_test.columns)
    feat_importances.sort_values(ascending=False).plot(kind='barh', color = "red")
    plt.title("Permutation Feature Importance")
    plt.savefig(st.IMAGES_PATH + 'permutation_feature_importance.png')
    plt.close()
