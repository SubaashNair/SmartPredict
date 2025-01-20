import pytest
from smartpredict.hyperparameter_tuning import tune_hyperparameters
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

def test_tune_hyperparameters():
    X, y = load_iris(return_X_y=True)
    model = RandomForestClassifier()
    param_distributions = {'n_estimators': [10, 50, 100]}
    tuned_model = tune_hyperparameters(model, param_distributions, X, y)
    assert tuned_model is not None