import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import model.settings as st
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import learning_curve, validation_curve
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

def prediction_distribution(y_pred, set):
    # Create distribution plot
    pd.cut(y_pred, bins=np.arange(0, 1.1, 0.1)
           ).value_counts().plot(kind='bar')
    plt.title(f'{set} score distribution')
    plt.subplots_adjust(left=0.25)
    plt.savefig(st.IMAGES_PATH + f'{set} scores.png')
    plt.close()

def confusion_matrices(set, y_set,
                       y_pred_scores, y_pred_labels):
    # Confusion matrices plots for different thresholds
    for cut_point in st.CUT_POINTS:
        cm = metrics.confusion_matrix(y_set,
                                      (y_pred_scores >= cut_point).astype(int))
        disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()
        plt.savefig(st.IMAGES_PATH + f'{set} cm {str(cut_point)[2:]}.png')

    # Confusion matrices plots for recommended threshold
    cm = metrics.confusion_matrix(y_set, y_pred_labels)
    disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.savefig(st.IMAGES_PATH + f'{set} confusion matrix.png')


def format_plot(title, xlabel, ylabel):
    '''
    Function to add format to plot
    '''
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid('on')
    plt.axis('square')
    plt.ylim((-0.05, 1.05))
    plt.legend()
    plt.tight_layout()
    pass


def roc_curves(labels, prediction_scores, legend,
               title, x_label, y_label,
               color=st.COLORS[1]):
    # ROC AUC
    fpr, tpr, _ = metrics.roc_curve(labels, prediction_scores, pos_label=1)
    auc = metrics.roc_auc_score(labels, prediction_scores)
    legend_string = legend + ' ($AUC = {:0.4f}$)'.format(auc)

    # Create plot
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=legend_string, color=color)

    # Format plot
    format_plot(title, x_label, y_label)

    # Save plot
    plt.savefig(st.IMAGES_PATH + 'roc.png', dpi=150)
    plt.close()


def plot_prc(labels, prediction_scores, legend,
             title, x_label, y_label,
             color=st.COLORS[1]):
    '''
    Function to plot PRC curve
    '''
    precision, \
        recall, \
        _ = metrics.precision_recall_curve(labels,
                                           prediction_scores)
    average_precision = metrics.average_precision_score(
        labels, prediction_scores)
    legend_string = legend + ' ($AP = {:0.4f}$)'.format(average_precision)

    # Create plot1
    plt.plot(recall, precision, label=legend_string, color=color)

    # Format plot
    format_plot(title, x_label, y_label)

    # Save plot
    plt.savefig(st.IMAGES_PATH + 'pcr.png', dpi=150)
    plt.close()


def plot_ks(labels, prediction_scores,
            color=st.COLORS):
    '''
    Function to plot KS plot
    '''
    # KS

    fpr, tpr, \
        thresholds = metrics.roc_curve(labels, prediction_scores,
                                       pos_label=1)
    fnr = 1 - tpr
    tnr = 1 - fpr
    thresholds[0] = 1
    plt.plot(thresholds, fnr, label='FNR (Class 1 Cum. Dist.)',
             color=color[0], lw=1.5)
    plt.plot(thresholds, tnr, label='TNR (Class 0 Cum. Dist.)',
             color=color[1], lw=1.5)

    kss = tnr - fnr
    ks = kss[np.argmax(np.abs(kss))]
    t_ = thresholds[np.argmax(np.abs(kss))]

    # Create plot1
    plt.vlines(t_, tnr[np.argmax(np.abs(kss))], fnr[np.argmax(
        np.abs(kss))], colors='red', linestyles='dashed')

    # Format plot
    format_plot(f'Test KS = {ks}; {t_} Threshold',
                'Threshold', 'Rate (Cumulative Distribution)')

    # Save plot
    plt.savefig(st.IMAGES_PATH + 'ks_curve.png', dpi=150)
    plt.close()


def plot_metrics(y_test,
                 test_y_pred_scores,
                 test_y_pred_labels):
    # Plot distribution
    prediction_distribution(test_y_pred_scores, "Test set")
    # Plot confusion matrix
    confusion_matrices("Test set", y_test,
                       test_y_pred_scores, test_y_pred_labels)
    # Plot roc auc
    roc_curves(y_test, test_y_pred_scores,
               'Test', 'ROC',
               'False Positive Rate',
               'True Positive Rate (Recall)')

def plot_validation_curve(X, y, degree = 1):
    """
    - Validation curve is a graph showing the results on training and validation sets depending on the complexity of the model:
        - if the two curves are close to each other and both errors are large, it is a sign of underfitting
        - if the two curves are far from each other, it is a sign of overfitting
    """
    alphas = np.logspace(-2, 3, 20)
    sgd_logit = SGDClassifier(loss="log_loss", n_jobs=-1, random_state=17, max_iter=1000, alpha = 100)
    logit_pipe = Pipeline(
        [
            ("standarScaler", StandardScaler()),
            ("poly", PolynomialFeatures(degree=degree)),
            ("sgd_logit", sgd_logit),
        ]
    )
    val_train, val_test = validation_curve(
        estimator=logit_pipe, X=X, y=y, param_name="sgd_logit__alpha", param_range=alphas, cv=5, scoring="roc_auc"
    )
    plot_with_err(alphas, val_train, label="training scores")
    plot_with_err(alphas, val_test, label="validation scores")
    plt.xlabel(r"$\alpha$")
    plt.ylabel("ROC AUC")
    plt.legend()
    #plt.grid(True)

    # Save plot
    plt.savefig(st.IMAGES_PATH + 'validation_curve.png', dpi=150)
    plt.close()


def plot_with_err(x, data, **kwargs):
    mu, std = data.mean(1), data.std(1)
    lines = plt.plot(x, mu, "-", **kwargs)
    plt.fill_between(
        x,
        mu - std,
        mu + std,
        edgecolor="none",
        facecolor=lines[0].get_color(),
        alpha=0.2,
    )

def plot_learning_curve(X, y, degree=2, alpha=200):
    """
    - Learning Curve is a graph showing the results on training and validation sets depending on 
      the number of observations:
        - if the curves converge, adding new data won’t help, and it is necessary to change the complexity of the model
        - if the curves have not converged, adding new data can improve the result 
    """
    train_sizes = np.linspace(0.05, 1, 20)
    logit_pipe = Pipeline(
        [
            ("standarScaler", StandardScaler()),
            ("poly", PolynomialFeatures(degree=degree)),
            (
                "sgd_logit",
                SGDClassifier(n_jobs=-1, random_state=17, alpha=alpha, max_iter=1000),
            ),
        ]
    )
    N_train, val_train, val_test = learning_curve(
        logit_pipe, X, y, train_sizes=train_sizes, cv=5, scoring="roc_auc"
    )
    plot_with_err(N_train, val_train, label="training scores")
    plot_with_err(N_train, val_test, label="validation scores")
    plt.xlabel("Training Set Size")
    plt.ylabel("AUC")
    plt.legend()
    #plt.grid(True)
    # Save plot
    plt.savefig(st.IMAGES_PATH + 'learning_curve.png', dpi=150)
    plt.close()