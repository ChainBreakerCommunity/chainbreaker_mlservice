from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
import model.settings as st
import joblib
import logging 

def get_randomforest_classifier():

    # Initialize RandomForestClassifier
    clf = RandomForestClassifier(
        criterion = st.CRITERION,
        random_state = st.RANDOM_STATE, 
        max_features = st.MAX_FEATURES
    )

    # Return random forest.
    return clf

def train_randomforest(clf, X_train, y_train, cv = 10):
    # Initialize Random Search with CV
    rsearch = RandomizedSearchCV(
        estimator = clf,
        param_distributions = st.DISTRIBUTIONS,
        random_state = st.RANDOM_STATE,
        scoring = 'precision',
        cv = cv,
        verbose = 2,
        n_iter = 20
    )

    # Train model
    logging.warning('Training...\n')
    models = rsearch.fit(X_train, y_train)
    model = models.best_estimator_

    # Export model
    logging.warning('Exporting model...')
    joblib.dump(model, st.MODEL_PATH)
    return model
