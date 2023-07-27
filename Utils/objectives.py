import numpy as np
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif


# Fisher Score
def multi_class_fisher_score(X, y):
    # Number of classes
    num_classes = len(np.unique(y))

    # Mean of the entire feature
    m = np.mean(X)

    # Class probabilities, means, and variances
    p = np.zeros(num_classes)
    m_k = np.zeros(num_classes)
    var_k = np.zeros(num_classes)

    for k in range(num_classes):
        X_k = X[y == k]
        p[k] = len(X_k) / len(X)
        m_k[k] = np.mean(X_k)
        var_k[k] = np.var(X_k)

    # Fisher score
    between_class_variance = np.sum(p * (m_k - m) ** 2)
    within_class_variance = np.sum(p * var_k)
    fisher_score = between_class_variance / within_class_variance

    return fisher_score


def mean_fisher(X, y):
    fisher_scores = {column: multi_class_fisher_score(X[column], y) for column in X.columns}
    fisher_scores_df = pd.DataFrame.from_dict(fisher_scores, orient='index', columns=['Fisher Score'])

    return fisher_scores_df['Fisher Score'].mean()


# Mutual Information
def mean_mi(X, y, regression=True):
    if regression:
        mutual_info = mutual_info_regression(X, y)
    else:
        mutual_info = mutual_info_classif(X, y)

    mutual_info_df = pd.DataFrame(mutual_info, index=X.columns, columns=['Mutual Information'])

    return mutual_info_df['Mutual Information'].mean()


# Variance Inflation Factor (VIF)
def mean_vif(X):
    vif = pd.DataFrame()
    vif['index'] = X.columns
    vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

    return vif["VIF"].mean()


# Bayesian information criterion (BIC)
def bic(model, X, y):
    # For linear/logist regression
    n_samples = X.shape[0]
    n_params = len(model.coef_) + 1
    log_likelihood = model.score(X, y)

    # BIC = ln(n)*k - 2ln(L)
    bic = np.log(n_samples) * n_params - 2 * log_likelihood

    return bic

