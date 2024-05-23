# smartpredict/feature_engineering.py
"""
Feature engineering module for SmartPredict.
Provides functions for automated feature engineering.
"""

from sklearn.preprocessing import StandardScaler

def engineer_features(X_train, X_test):
    """
    Apply feature engineering to the data.

    Parameters:
    X_train (array-like): Training features.
    X_test (array-like): Testing features.

    Returns:
    tuple: Transformed training and testing features.
    """
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test