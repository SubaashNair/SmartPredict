# smartpredict/feature_engineering.py
"""
Feature engineering module for SmartPredict.
Provides functions for automated feature engineering.
"""

from sklearn.preprocessing import PolynomialFeatures
from sklearn.decomposition import PCA
# from sklearn.pipeline import Pipeline

def engineer_features(X_train, X_test):
    """
    Apply feature engineering to the data.

    Parameters:
    X_train (array-like): Training features.
    X_test (array-like): Testing features.

    Returns:
    tuple: Transformed training and testing features.
    """
    try:
        poly = PolynomialFeatures(degree=2)
        X_train_poly = poly.fit_transform(X_train)
        X_test_poly = poly.transform(X_test)

        pca = PCA(n_components=0.95)
        X_train_pca = pca.fit_transform(X_train_poly)
        X_test_poly = pca.transform(X_test_poly)

        return X_train_pca, X_test_poly
    except Exception as e:
        logging.error(f"Feature engineering failed: {e}")
        return X_train,X_test