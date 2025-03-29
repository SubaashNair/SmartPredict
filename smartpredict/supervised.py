# smartpredict/supervised.py
"""
Supervised learning module for SmartPredict.
Provides high-level interfaces for classification and regression tasks.
"""

from typing import List, Dict, Any, Optional, Callable, Union
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score
from sklearn.base import BaseEstimator
import logging

from .base import BaseModel
from .model_registry import get_classifier, get_regressor
from .feature_engineering import engineer_features
from .explainability import explain_model
from .ensemble_methods import apply_ensemble_methods

class SmartClassifier(BaseModel):
    """A class for automated classification tasks.

    Parameters
    ----------
    models : list, optional (default=None)
        List of classifier names to use
    custom_metric : callable, optional (default=None)
        Custom scoring function with signature (y_true, y_pred)
    verbose : int, optional (default=0)
        Verbosity level
    ignore_warnings : bool, optional (default=True)
        Whether to ignore warnings during model training
    **kwargs : dict
        Additional parameters to pass to the models
    """
    def __init__(self, 
                 models: Optional[List[str]] = None, 
                 custom_metric: Optional[Callable] = None,
                 verbose: int = 0,
                 ignore_warnings: bool = True,
                 **kwargs):
        super().__init__()
        self.models = models
        self.verbose = verbose
        self.ignore_warnings = ignore_warnings
        if custom_metric is not None and not callable(custom_metric):
            raise ValueError("custom_metric must be callable")
        self.custom_metric = custom_metric
        self.model_params = kwargs
        self.trained_models = {}
        self.results = {}
        self._is_fitted = False
        
        # Validate models if provided
        if models is not None:
            for model_name in models:
                if get_classifier(model_name) is None:
                    raise ValueError(f"Invalid model name: {model_name}")

    def fit(self, X_train: Union[np.ndarray, pd.DataFrame], 
            X_test: Union[np.ndarray, pd.DataFrame],
            y_train: Union[np.ndarray, pd.Series],
            y_test: Union[np.ndarray, pd.Series]) -> Dict[str, Dict[str, float]]:
        """Fit the models on training data and evaluate on test data.

        Parameters
        ----------
        X_train : array-like of shape (n_samples, n_features)
            Training data
        X_test : array-like of shape (n_samples, n_features)
            Test data
        y_train : array-like of shape (n_samples,)
            Training target values
        y_test : array-like of shape (n_samples,)
            Test target values

        Returns
        -------
        dict
            Dictionary containing evaluation metrics for each model
        """
        # Input validation
        X_train = np.asarray(X_train)
        y_train = np.asarray(y_train)
        X_test = np.asarray(X_test)
        y_test = np.asarray(y_test)
        
        if X_train.shape[0] != y_train.shape[0]:
            raise ValueError("X_train and y_train must have the same number of samples")
        
        if X_test.shape[0] != y_test.shape[0]:
            raise ValueError("X_test and y_test must have the same number of samples")

        # Use default models if none provided
        if self.models is None:
            self.models = ['RandomForestClassifier', 'LogisticRegression']

        for model_name in self.models:
            try:
                model = get_classifier(model_name)
                if model is None:
                    raise ValueError(f"Could not initialize model {model_name}")
                
                # Set parameters if provided
                if self.model_params.get(model_name):
                    model.set_params(**self.model_params[model_name])
                
                model.fit(X_train, y_train)
                self.trained_models[model_name] = model
            except Exception as e:
                if self.verbose > 0:
                    logging.error(f"Error training {model_name}: {str(e)}")
                continue

        if not self.trained_models:
            raise RuntimeError("No models were successfully trained")
        
        self._is_fitted = True
        
        # Evaluate all trained models
        results = {}
        for name, model in self.trained_models.items():
            try:
                y_pred = model.predict(X_test)
                metrics = {
                    'accuracy': accuracy_score(y_test, y_pred),
                    'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
                    'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
                    'f1': f1_score(y_test, y_pred, average='weighted', zero_division=0)
                }
                
                if self.custom_metric:
                    try:
                        metrics['custom'] = self.custom_metric(y_test, y_pred)
                    except Exception as e:
                        if self.verbose > 0:
                            logging.warning(f"Custom metric failed for {name}: {str(e)}")
                
                results[name] = metrics
            except Exception as e:
                if self.verbose > 0:
                    logging.error(f"Error evaluating {name}: {str(e)}")
        
        self.results = results
        return results

    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> Dict[str, np.ndarray]:
        """Make predictions using trained models.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict

        Returns
        -------
        dict
            Dictionary containing predictions from each model
        """
        if not self._is_fitted:
            raise RuntimeError("Models must be fitted before making predictions")
        
        X = np.asarray(X)
        predictions = {}
        
        for name, model in self.trained_models.items():
            try:
                predictions[name] = model.predict(X)
            except Exception as e:
                logging.error(f"Error predicting with {name}: {str(e)}")
                
        return predictions

    def evaluate(self, X: Union[np.ndarray, pd.DataFrame], 
                y: Union[np.ndarray, pd.Series]) -> Dict[str, Dict[str, float]]:
        """Evaluate models on test data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples
        y : array-like of shape (n_samples,)
            True labels

        Returns
        -------
        dict
            Dictionary containing evaluation metrics for each model
        """
        if not self._is_fitted:
            raise RuntimeError("Models must be fitted before evaluation")

        X = np.asarray(X)
        y = np.asarray(y)
        
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of samples")
        
        results = {}
        for name, model in self.trained_models.items():
            try:
                y_pred = model.predict(X)
                metrics = {
                    'accuracy': accuracy_score(y, y_pred),
                    'precision': precision_score(y, y_pred, average='weighted', zero_division=0),
                    'recall': recall_score(y, y_pred, average='weighted', zero_division=0),
                    'f1': f1_score(y, y_pred, average='weighted', zero_division=0)
                }
                
                if self.custom_metric:
                    try:
                        metrics['custom'] = self.custom_metric(y, y_pred)
                    except Exception as e:
                        if self.verbose > 0:
                            logging.warning(f"Custom metric failed for {name}: {str(e)}")
                
                results[name] = metrics
            except Exception as e:
                if self.verbose > 0:
                    logging.error(f"Error evaluating {name}: {str(e)}")
        
        return results

    def score(self, X: Union[np.ndarray, pd.DataFrame], 
             y: Union[np.ndarray, pd.Series]) -> Dict[str, float]:
        """Calculate the average score for each model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples
        y : array-like of shape (n_samples,)
            True labels

        Returns
        -------
        dict
            Dictionary containing accuracy score for each model
        """
        evaluation = self.evaluate(X, y)
        return {name: metrics['accuracy'] for name, metrics in evaluation.items()}

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """Get parameters for this estimator.

        Parameters
        ----------
        deep : bool, default=True
            If True, include parameters of all subobjects

        Returns
        -------
        dict
            Parameter names mapped to their values
        """
        params = {
            'models': self.models,
            'custom_metric': self.custom_metric,
            'verbose': self.verbose,
            'ignore_warnings': self.ignore_warnings
        }
        params.update(self.model_params)
        return params

    def set_params(self, **params) -> 'SmartClassifier':
        """Set the parameters of this estimator.

        Parameters
        ----------
        **params : dict
            Estimator parameters

        Returns
        -------
        self
            Estimator instance
        """
        for key, value in params.items():
            setattr(self, key, value)
        return self


class SmartRegressor(BaseModel):
    """A class for automated regression tasks.

    Parameters
    ----------
    models : list, optional (default=None)
        List of regressor names to use
    custom_metric : callable, optional (default=None)
        Custom scoring function with signature (y_true, y_pred)
    verbose : int, optional (default=0)
        Verbosity level
    ignore_warnings : bool, optional (default=True)
        Whether to ignore warnings during model training
    **kwargs : dict
        Additional parameters to pass to the models
    """
    def __init__(self, 
                 models: Optional[List[str]] = None, 
                 custom_metric: Optional[Callable] = None,
                 verbose: int = 0,
                 ignore_warnings: bool = True,
                 **kwargs):
        super().__init__()
        self.models = models
        self.verbose = verbose
        self.ignore_warnings = ignore_warnings
        if custom_metric is not None and not callable(custom_metric):
            raise ValueError("custom_metric must be callable")
        self.custom_metric = custom_metric
        self.model_params = kwargs
        self.trained_models = {}
        self.results = {}
        self._is_fitted = False
        
        # Validate models if provided
        if models is not None:
            for model_name in models:
                if get_regressor(model_name) is None:
                    raise ValueError(f"Invalid model name: {model_name}")

    def fit(self, X_train: Union[np.ndarray, pd.DataFrame], 
            X_test: Union[np.ndarray, pd.DataFrame],
            y_train: Union[np.ndarray, pd.Series],
            y_test: Union[np.ndarray, pd.Series]) -> Dict[str, Dict[str, float]]:
        """Fit the models on training data and evaluate on test data.

        Parameters
        ----------
        X_train : array-like of shape (n_samples, n_features)
            Training data
        X_test : array-like of shape (n_samples, n_features)
            Test data
        y_train : array-like of shape (n_samples,)
            Training target values
        y_test : array-like of shape (n_samples,)
            Test target values

        Returns
        -------
        dict
            Dictionary containing evaluation metrics for each model
        """
        # Input validation
        X_train = np.asarray(X_train)
        y_train = np.asarray(y_train)
        X_test = np.asarray(X_test)
        y_test = np.asarray(y_test)
        
        if X_train.shape[0] != y_train.shape[0]:
            raise ValueError("X_train and y_train must have the same number of samples")
        
        if X_test.shape[0] != y_test.shape[0]:
            raise ValueError("X_test and y_test must have the same number of samples")

        # Use default models if none provided
        if self.models is None:
            self.models = ['RandomForestRegressor', 'LinearRegression']

        for model_name in self.models:
            try:
                model = get_regressor(model_name)
                if model is None:
                    raise ValueError(f"Could not initialize model {model_name}")
                
                # Set parameters if provided
                if self.model_params.get(model_name):
                    model.set_params(**self.model_params[model_name])
                
                model.fit(X_train, y_train)
                self.trained_models[model_name] = model
            except Exception as e:
                if self.verbose > 0:
                    logging.error(f"Error training {model_name}: {str(e)}")
                continue

        if not self.trained_models:
            raise RuntimeError("No models were successfully trained")
        
        self._is_fitted = True
        
        # Evaluate all trained models
        results = {}
        for name, model in self.trained_models.items():
            try:
                y_pred = model.predict(X_test)
                metrics = {
                    'mse': mean_squared_error(y_test, y_pred),
                    'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                    'mae': mean_absolute_error(y_test, y_pred),
                    'r2': r2_score(y_test, y_pred)
                }
                
                if self.custom_metric:
                    try:
                        metrics['custom'] = self.custom_metric(y_test, y_pred)
                    except Exception as e:
                        if self.verbose > 0:
                            logging.warning(f"Custom metric failed for {name}: {str(e)}")
                
                results[name] = metrics
            except Exception as e:
                if self.verbose > 0:
                    logging.error(f"Error evaluating {name}: {str(e)}")
        
        self.results = results
        return results

    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> Dict[str, np.ndarray]:
        """Make predictions using trained models.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict

        Returns
        -------
        dict
            Dictionary containing predictions from each model
        """
        if not self._is_fitted:
            raise RuntimeError("Models must be fitted before making predictions")
        
        X = np.asarray(X)
        predictions = {}
        
        for name, model in self.trained_models.items():
            try:
                predictions[name] = model.predict(X)
            except Exception as e:
                logging.error(f"Error predicting with {name}: {str(e)}")
                
        return predictions

    def evaluate(self, X: Union[np.ndarray, pd.DataFrame], 
                y: Union[np.ndarray, pd.Series]) -> Dict[str, Dict[str, float]]:
        """Evaluate models on test data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples
        y : array-like of shape (n_samples,)
            True values

        Returns
        -------
        dict
            Dictionary containing evaluation metrics for each model
        """
        if not self._is_fitted:
            raise RuntimeError("Models must be fitted before evaluation")

        X = np.asarray(X)
        y = np.asarray(y)
        
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of samples")
        
        results = {}
        for name, model in self.trained_models.items():
            try:
                y_pred = model.predict(X)
                metrics = {
                    'mse': mean_squared_error(y, y_pred),
                    'rmse': np.sqrt(mean_squared_error(y, y_pred)),
                    'mae': mean_absolute_error(y, y_pred),
                    'r2': r2_score(y, y_pred)
                }
                
                if self.custom_metric:
                    try:
                        metrics['custom'] = self.custom_metric(y, y_pred)
                    except Exception as e:
                        if self.verbose > 0:
                            logging.warning(f"Custom metric failed for {name}: {str(e)}")
                
                results[name] = metrics
            except Exception as e:
                if self.verbose > 0:
                    logging.error(f"Error evaluating {name}: {str(e)}")
        
        return results

    def score(self, X: Union[np.ndarray, pd.DataFrame], 
             y: Union[np.ndarray, pd.Series]) -> Dict[str, float]:
        """Calculate the average score for each model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples
        y : array-like of shape (n_samples,)
            True values

        Returns
        -------
        dict
            Dictionary containing R^2 score for each model
        """
        evaluation = self.evaluate(X, y)
        return {name: metrics['r2'] for name, metrics in evaluation.items()}

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """Get parameters for this estimator.

        Parameters
        ----------
        deep : bool, default=True
            If True, include parameters of all subobjects

        Returns
        -------
        dict
            Parameter names mapped to their values
        """
        params = {
            'models': self.models,
            'custom_metric': self.custom_metric,
            'verbose': self.verbose,
            'ignore_warnings': self.ignore_warnings
        }
        params.update(self.model_params)
        return params

    def set_params(self, **params) -> 'SmartRegressor':
        """Set the parameters of this estimator.

        Parameters
        ----------
        **params : dict
            Estimator parameters

        Returns
        -------
        self
            Estimator instance
        """
        for key, value in params.items():
            setattr(self, key, value)
        return self