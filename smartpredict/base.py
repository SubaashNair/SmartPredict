# smartpredict/base.py
"""
Base module for SmartPredict.
Defines the BasePredictor class to be inherited by specific predictors.
"""

class BasePredictor:
    """
    A base class for SmartPredict predictors.

    Parameters:
    verbose (int): Level of verbosity.
    ignore_warnings (bool): Whether to ignore warnings.
    custom_metric (callable): A custom evaluation metric function.
    """
    def __init__(self, verbose=0, ignore_warnings=True, custom_metric=None):
        self.verbose = verbose
        self.ignore_warnings = ignore_warnings
        self.custom_metric = custom_metric

    def fit(self, X_train, X_test, y_train, y_test):
        """
        Fit the model using training data.

        Parameters:
        X_train (array-like): Training features.
        X_test (array-like): Testing features.
        y_train (array-like): Training labels.
        y_test (array-like): Testing labels.

        Returns:
        dict: Model performance results.
        """
        raise NotImplementedError("This method needs to be overridden in subclasses")