import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.feature_selection import mutual_info_regression


# Fisher Score
def fisher(X, y):
    # Split samples according to class
    X1 = X[y == 0]
    X2 = X[y == 1]

    # Means and variances
    m1, m2 = np.mean(X1), np.mean(X2)
    var1, var2 = np.var(X1), np.var(X2)
    m = np.mean(X)

    # Class probabilities
    p1 = len(X1) / len(X)
    p2 = len(X2) / len(X)

    # Fisher score
    between_class_variance = p1 * (m1 - m) ** 2 + p2 * (m2 - m) ** 2
    within_class_variance = p1 * var1 + p2 * var2
    fisher_score = between_class_variance / within_class_variance

    return fisher_score


# Mutual Information
def mi(X, y):
    mi = mutual_info_regression(X, y)

    return mi


# Bayesian information criterion (BIC)
def bic(model, X, y):
    # TODO: For linear regression temporarily, change n_samples later
    n_samples = X.shape[0]
    n_params = len(model.coef_) + 1
    log_likelihood = model.score(X, y)

    # BIC = ln(n)*k - 2ln(L)
    bic = np.log(n_samples) * n_params - 2 * log_likelihood

    return bic


# Variance Inflation Factor (VIF)
def vif(X, y):
    vif = variance_inflation_factor(X, y)

    return vif

