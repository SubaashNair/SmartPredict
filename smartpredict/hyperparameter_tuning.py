# smartpredict/hyperparameter_tuning.py
"""
Hyperparameter tuning module for SmartPredict.
Provides functions to perform hyperparameter optimization.
"""

from sklearn.model_selection import GridSearchCV

def tune_hyperparameters(model, param_grid, X_train, y_train):
    """
    Perform hyperparameter tuning using GridSearchCV.

    Parameters:
    model (estimator): The machine learning model to tune.
    param_grid (dict): The parameter grid to search over.
    X_train (array-like): Training features.
    y_train (array-like): Training labels.

    Returns:
    estimator: The best model found by GridSearchCV.
    """
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_