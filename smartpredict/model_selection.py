# smartpredict/model_selection.py
"""
Model selection module for SmartPredict.
Provides functions to retrieve a dictionary of machine learning models.
"""

from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor, AdaBoostClassifier, AdaBoostRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier, MLPRegressor
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from catboost import CatBoostClassifier, CatBoostRegressor

def get_models(task):
    """
    Retrieve a dictionary of models based on the task.

    Parameters:
    task (str): The task type ('classification' or 'regression').

    Returns:
    dict: A dictionary of models.
    """
    models = {}
    if task == 'classification':
        models['Logistic Regression'] = LogisticRegression()
        models['Random Forest'] = RandomForestClassifier()
        models['Gradient Boosting'] = GradientBoostingClassifier()
        models['AdaBoost'] = AdaBoostClassifier()
        models['Decision Tree'] = DecisionTreeClassifier()
        models['Support Vector Machine'] = SVC()
        models['K-Nearest Neighbors'] = KNeighborsClassifier()
        models['Gaussian Naive Bayes'] = GaussianNB()
        models['Neural Network'] = MLPClassifier()
        models['XGBoost'] = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
        models['LightGBM'] = LGBMClassifier()
        models['CatBoost'] = CatBoostClassifier(verbose=0)
    elif task == 'regression':
        models['Linear Regression'] = LinearRegression()
        models['Ridge Regression'] = Ridge()
        models['Lasso Regression'] = Lasso()
        models['Random Forest'] = RandomForestRegressor()
        models['Gradient Boosting'] = GradientBoostingRegressor()
        models['AdaBoost'] = AdaBoostRegressor()
        models['Decision Tree'] = DecisionTreeRegressor()
        models['Support Vector Machine'] = SVR()
        models['K-Nearest Neighbors'] = KNeighborsRegressor()
        models['Neural Network'] = MLPRegressor()
        models['XGBoost'] = XGBRegressor()
        models['LightGBM'] = LGBMRegressor()
        models['CatBoost'] = CatBoostRegressor(verbose=0)
    return models