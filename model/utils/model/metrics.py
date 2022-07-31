
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, roc_curve, precision_recall_curve
import model.settings as st 
import json 
import numpy as np 

def predict_and_calculate_metrics(model, X_test, Y_test):
    '''Get model evaluation metrics on the test set.'''
    
    # Predict scores.
    y_pred_logits = model.predict_proba(X_test)[:,1]
    precisions, recalls, thresholds = precision_recall_curve(Y_test, y_pred_logits)

    # Get metrics.
    f1 = (2 * (precisions * recalls)) / (precisions + recalls)
    threshold = float(thresholds[np.argmax(f1)])
    precision = precisions[np.argmax(f1)]
    recall = recalls[np.argmax(f1)]

    # Get labels according to best threshold.
    y_predict_r = (y_pred_logits >= threshold).astype(int)

    # Calculate evaluation metrics for assesing performance of the model.
    roc = roc_auc_score(Y_test, y_pred_logits)
    acc = accuracy_score(Y_test, y_predict_r)
    f1_metric = f1_score(Y_test, y_predict_r)

    metrics_dict = {
        f"optimal_threshold": round(threshold, 3),
        f"roc_auc_score": round(roc, 3), 
        f"accuracy_score": round(acc, 3),
        f"precision_score": round(precision, 3), 
        f"recall_score": round(recall, 3), 
        f"f1": round(f1_metric, 3)
    }

    # Save metrics
    with open(st.METRICS_PATH + "metrics.json", 'w') as fd:
        json.dump(
            metrics_dict,
            fd, indent=4
        )
    return y_pred_logits, y_predict_r, metrics_dict

def save_predictions(X_train, y_train,
                     train_y_pred_scores, train_y_pred_labels,
                     X_test, y_test,
                     test_y_pred_scores, test_y_pred_labels):
                     
    # Train dataset
    X_train['GROUND_TRUTH'] = y_train
    X_train['PREDICTED_SCORE'] = train_y_pred_scores
    X_train['PREDICTED_LABEL'] = train_y_pred_labels

    # Test dataset
    X_test['GROUND_TRUTH'] = y_test
    X_test['PREDICTED_SCORE'] = test_y_pred_scores
    X_test['PREDICTED_LABEL'] = test_y_pred_labels

    # Save datasets
    X_train.to_csv(st.TRAIN_SCORE_PATH, index=False, sep = ";")
    X_test.to_csv(st.TEST_SCORE_PATH, index=False, sep = ";")