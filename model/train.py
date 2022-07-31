import model.settings as st
import model.utils.data.preprocess as pp
import model.utils.model.feature_selection as fs 
import model.utils.model.random_forest as rf 
import model.utils.model.metrics as mt 
import model.utils.metrics.figures as fg
from datetime import datetime
import pandas as pd
import mlflow 
from logger.logger import get_logger

logger = get_logger(name = __name__, level = "DEBUG", stream = True)

def train_model():  

    mlflow.set_experiment("Human Trafficking Communities Classifier")
    with mlflow.start_run(run_name=datetime.now().strftime("%Y/%m/%d %H:%M:%S")):

        # 1. Get train and test datasets.
        X_train, Y_train, X_test, Y_test = pp.get_model_data(
            language = "english", 
            website = "leolist", 
            country = "canada", 
            download = True
        )

        X_train = pd.read_csv(st.TRAIN_DATASET_PATH, sep = ";")
        X_test = pd.read_csv(st.TEST_DATASET_PATH, sep = ";")
        X_train = X_train.drop("Unnamed: 0", axis = 1)
        X_test = X_test.drop("Unnamed: 0", axis = 1)

        Y_train = X_train[st.TARGET]
        Y_test = X_test[st.TARGET]

        X_train = X_train.drop(st.TARGET, axis = 1)
        X_test = X_test.drop(st.TARGET, axis = 1)

        # 2. Feature selection.
        X_train, X_test = fs.feature_selection(X_train, Y_train, X_test)

        # 3. Train model.
        clf = rf.get_randomforest_classifier()
        clf = rf.train_randomforest(clf, X_train, Y_train, cv = 5)

        # 4. Predictions and metrics.
        train_y_pred_scores, train_y_pred_labels, _ = mt.predict_and_calculate_metrics(clf, X_train, Y_train)
        test_y_pred_scores, test_y_pred_labels, test_metrics = mt.predict_and_calculate_metrics(clf, X_test, Y_test)

        # 5. Plots
        
        ## 5.1. Train set
        fg.prediction_distribution(train_y_pred_scores, "Train set")

        ## 5.2. Test set
        fg.plot_metrics(Y_test,
                        test_y_pred_scores,
                        test_y_pred_labels)

        ## 5.3. Plot validation and learning curves
        fg.plot_learning_curve(X_train, Y_train)
        fg.plot_validation_curve(X_train, Y_train)

        # 6. Explainability
        fs.get_feature_importance(clf, X_test)

        # 7. Save predictions
        mt.save_predictions(X_train, Y_train,
                            train_y_pred_scores, train_y_pred_labels,
                            X_test, Y_test,
                            test_y_pred_scores, test_y_pred_labels)

        # 8. Track experiment & log metrics achieved
        mlflow.log_params(clf.get_params())
        mlflow.log_metrics(test_metrics)
        mlflow.log_artifacts(local_dir = "./model/data")
        mlflow.sklearn.log_model(clf, "model")