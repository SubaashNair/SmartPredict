"""Model Registry Module."""

from typing import Dict, Any, Type, Optional
from sklearn.svm import LinearSVC, SVC, NuSVC, LinearSVR, NuSVR, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor, NearestCentroid
from sklearn.tree import (
    DecisionTreeClassifier, DecisionTreeRegressor,
    ExtraTreeClassifier, ExtraTreeRegressor
)
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    RidgeCV,
    Lasso,
    LassoCV,
    ElasticNet,
    ElasticNetCV,
    Lars,
    LarsCV,
    LassoLars,
    LassoLarsCV,
    LassoLarsIC,
    OrthogonalMatchingPursuit,
    OrthogonalMatchingPursuitCV,
    BayesianRidge,
    ARDRegression,
    SGDRegressor,
    SGDClassifier,
    PassiveAggressiveRegressor,
    PassiveAggressiveClassifier,
    RANSACRegressor,
    TheilSenRegressor,
    HuberRegressor,
    PoissonRegressor,
    TweedieRegressor,
    GammaRegressor,
    LogisticRegression,
    LogisticRegressionCV,
    Perceptron,
    RidgeClassifier,
    RidgeClassifierCV
)
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.calibration import CalibratedClassifierCV
from sklearn.semi_supervised import LabelPropagation, LabelSpreading
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
    AdaBoostClassifier, AdaBoostRegressor,
    BaggingClassifier, BaggingRegressor,
    ExtraTreesClassifier, ExtraTreesRegressor,
    HistGradientBoostingClassifier, HistGradientBoostingRegressor
)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.kernel_ridge import KernelRidge
from sklearn.gaussian_process import GaussianProcessRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.compose import TransformedTargetRegressor
from sklearn.base import BaseEstimator
import logging

class ModelRegistry:
    """
    Registry for all available classification and regression models.
    """
    
    @staticmethod
    def get_classifier_models() -> Dict[str, Type[Any]]:
        """
        Returns a dictionary of all available classification models.
        """
        return {
            'LinearSVC': LinearSVC,
            'SGDClassifier': SGDClassifier,
            'MLPClassifier': MLPClassifier,
            'Perceptron': Perceptron,
            'LogisticRegression': LogisticRegression,
            'LogisticRegressionCV': LogisticRegressionCV,
            'SVC': SVC,
            'CalibratedClassifierCV': CalibratedClassifierCV,
            'PassiveAggressiveClassifier': PassiveAggressiveClassifier,
            'LabelPropagation': LabelPropagation,
            'LabelSpreading': LabelSpreading,
            'RandomForestClassifier': RandomForestClassifier,
            'GradientBoostingClassifier': GradientBoostingClassifier,
            'QuadraticDiscriminantAnalysis': QuadraticDiscriminantAnalysis,
            'HistGradientBoostingClassifier': HistGradientBoostingClassifier,
            'RidgeClassifierCV': RidgeClassifierCV,
            'RidgeClassifier': RidgeClassifier,
            'AdaBoostClassifier': AdaBoostClassifier,
            'ExtraTreesClassifier': ExtraTreesClassifier,
            'KNeighborsClassifier': KNeighborsClassifier,
            'BaggingClassifier': BaggingClassifier,
            'BernoulliNB': BernoulliNB,
            'LinearDiscriminantAnalysis': LinearDiscriminantAnalysis,
            'GaussianNB': GaussianNB,
            'NuSVC': NuSVC,
            'DecisionTreeClassifier': DecisionTreeClassifier,
            'NearestCentroid': NearestCentroid,
            'ExtraTreeClassifier': ExtraTreeClassifier,
            'DummyClassifier': DummyClassifier
        }

    @staticmethod
    def get_regressor_models() -> Dict[str, Type[Any]]:
        """
        Returns a dictionary of all available regression models.
        """
        return {
            'ExtraTreesRegressor': ExtraTreesRegressor,
            'OrthogonalMatchingPursuitCV': OrthogonalMatchingPursuitCV,
            'Lasso': Lasso,
            'LassoLars': LassoLars,
            'LarsCV': LarsCV,
            'LassoCV': LassoCV,
            'PassiveAggressiveRegressor': PassiveAggressiveRegressor,
            'LassoLarsIC': LassoLarsIC,
            'SGDRegressor': SGDRegressor,
            'RidgeCV': RidgeCV,
            'Ridge': Ridge,
            'BayesianRidge': BayesianRidge,
            'LassoLarsCV': LassoLarsCV,
            'TransformedTargetRegressor': TransformedTargetRegressor,
            'LinearRegression': LinearRegression,
            'Lars': Lars,
            'ElasticNetCV': ElasticNetCV,
            'HuberRegressor': HuberRegressor,
            'RandomForestRegressor': RandomForestRegressor,
            'AdaBoostRegressor': AdaBoostRegressor,
            'LGBMRegressor': LGBMRegressor,
            'HistGradientBoostingRegressor': HistGradientBoostingRegressor,
            'PoissonRegressor': PoissonRegressor,
            'ElasticNet': ElasticNet,
            'KNeighborsRegressor': KNeighborsRegressor,
            'OrthogonalMatchingPursuit': OrthogonalMatchingPursuit,
            'BaggingRegressor': BaggingRegressor,
            'GradientBoostingRegressor': GradientBoostingRegressor,
            'TweedieRegressor': TweedieRegressor,
            'XGBRegressor': XGBRegressor,
            'GammaRegressor': GammaRegressor,
            'RANSACRegressor': RANSACRegressor,
            'LinearSVR': LinearSVR,
            'ExtraTreeRegressor': ExtraTreeRegressor,
            'NuSVR': NuSVR,
            'SVR': SVR,
            'DummyRegressor': DummyRegressor,
            'DecisionTreeRegressor': DecisionTreeRegressor,
            'GaussianProcessRegressor': GaussianProcessRegressor,
            'MLPRegressor': MLPRegressor,
            'KernelRidge': KernelRidge
        }

# Registry of available models
CLASSIFIER_REGISTRY: Dict[str, Type[BaseEstimator]] = {
    'RandomForestClassifier': RandomForestClassifier,
    'LogisticRegression': LogisticRegression,
    # Add more classifiers here
}

REGRESSOR_REGISTRY: Dict[str, Type[BaseEstimator]] = {
    'RandomForestRegressor': RandomForestRegressor,
    'LinearRegression': LinearRegression,
    # Add more regressors here
}

def get_classifier(name: str) -> Optional[BaseEstimator]:
    """Get a new instance of a classifier by name.

    Parameters
    ----------
    name : str
        Name of the classifier

    Returns
    -------
    BaseEstimator or None
        New instance of the classifier, or None if not found
    """
    try:
        model_class = CLASSIFIER_REGISTRY.get(name)
        if model_class is None:
            logging.warning(f"Classifier {name} not found in registry")
            return None
        return model_class()
    except Exception as e:
        logging.error(f"Error creating classifier {name}: {str(e)}")
        return None

def get_regressor(name: str) -> Optional[BaseEstimator]:
    """Get a new instance of a regressor by name.

    Parameters
    ----------
    name : str
        Name of the regressor

    Returns
    -------
    BaseEstimator or None
        New instance of the regressor, or None if not found
    """
    try:
        model_class = REGRESSOR_REGISTRY.get(name)
        if model_class is None:
            logging.warning(f"Regressor {name} not found in registry")
            return None
        return model_class()
    except Exception as e:
        logging.error(f"Error creating regressor {name}: {str(e)}")
        return None 