import numpy as np
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
from statsmodels.tools.tools import add_constant


# Fisher Score
def fisher_score(X, y):
    classes = np.unique(y)
    c = len(classes)

    overall_mean = np.mean(X, axis=0)
    n = overall_mean.shape

    S_b = np.zeros(n)
    S_w = np.zeros(n)

    for cls in classes:
        X_c = X[y == cls]

        n_i = X_c.shape[0]
        mean_c = np.mean(X_c, axis=0)

        S_b += n_i * (mean_c - overall_mean) ** 2
        S_w += n_i * np.var(X_c, axis=0)

    fisher_scores = S_b / (S_w + 1e-10)

    return fisher_scores.mean()


def mutual_info(X, y, regression=True):
    if regression:
        mutual_info = mutual_info_regression(X, y)
    else:
        mutual_info = mutual_info_classif(X, y)

    mutual_info_df = pd.DataFrame(mutual_info, index=X.columns, columns=['Mutual Information'])

    return mutual_info_df['Mutual Information'].mean()


def vif(X):
    data_with_const = add_constant(X)
    data_with_const = data_with_const.select_dtypes(exclude='object')

    try:
        vif_data = [variance_inflation_factor(data_with_const.values, i)
                    for i in range(data_with_const.shape[1])]
    except:
        vif_data = [np.inf] * data_with_const.shape[1]

    # vif_data = [variance_inflation_factor(data_with_const.values, i)
    #             for i in range(data_with_const.shape[1])]

    vif_data = vif_data[1:]
    mean_vif = np.mean(vif_data)

    return mean_vif


# Bayesian information criterion (BIC)
def bic(model, X, y):
    # For linear/logist regression
    n_samples = X.shape[0]
    n_params = len(model.coef_) + 1
    log_likelihood = model.score(X, y)

    # BIC = ln(n)*k - 2ln(L)
    bic = np.log(n_samples) * n_params - 2 * log_likelihood

    return bic

