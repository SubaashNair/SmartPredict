# smartpredict/ensemble_methods.py
"""
Ensemble methods module for SmartPredict.
Provides functions to create ensemble models.
"""


from sklearn.ensemble import StackingClassifier, VotingClassifier, BaggingClassifier, VotingRegressor
import logging
from copy import deepcopy

def apply_ensemble_methods(base_learners, meta_learner, n_jobs=-1):
    try:
        stacking_model = StackingClassifier(estimators=base_learners, final_estimator=meta_learner, n_jobs=n_jobs)
        voting_model = VotingClassifier(estimators=base_learners, n_jobs=n_jobs)
        bagging_model = BaggingClassifier(estimator=meta_learner, n_jobs=n_jobs)
        return stacking_model, voting_model, bagging_model
    except Exception as e:
        logging.error(f"Ensemble method application failed: {e}")
        return None, None, None

import numpy as np
import joblib
from typing import List, Tuple, Optional, Any, Union, Dict
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.ensemble import StackingClassifier, VotingClassifier, StackingRegressor, VotingRegressor
from sklearn.model_selection import cross_val_predict
import pandas as pd
from sklearn.base import clone

class EnsembleModel:
    """A class for creating ensemble models.

    Parameters
    ----------
    models : list, optional (default=[])
        List of base models to use in the ensemble
    method : str, optional (default='voting')
        Method for combining predictions ('voting', 'averaging', 'weighted', or 'stacking')
    weights : list or None, optional (default=None)
        List of weights for each model (only used for weighted method)
    meta_model : object or None, optional (default=None)
        Meta-learner model for stacking (required if method='stacking')
    cv : int, optional (default=5)
        Number of cross-validation folds for stacking
    """
    def __init__(self, models=None, method='voting', weights=None, meta_model=None, cv=5):
        self.models = models if models is not None else []
        
        valid_methods = ['voting', 'averaging', 'weighted', 'stacking']
        if method not in valid_methods:
            raise ValueError(f"method must be one of {valid_methods}")
        self.method = method
        
        if weights is not None:
            if len(weights) != len(self.models):
                raise ValueError("number of weights must match number of models")
            if not all(w >= 0 for w in weights):
                raise ValueError("weights must be non-negative")
        self.weights = weights
        
        if method == 'stacking' and meta_model is None:
            raise ValueError("meta_model is required for stacking method")
        self.meta_model = meta_model
        
        self.cv = cv
        self.is_fitted = False
        self.is_classifier = None
        self.feature_names_ = None
        
        # Validate models
        if self.models:
            model_types = set()
            for name, model in self.models:
                if hasattr(model, 'predict_proba'):
                    model_types.add('classifier')
                else:
                    model_types.add('regressor')
            
            if len(model_types) > 1:
                raise ValueError("All models must be of the same type (classifier or regressor)")

    def get_params(self, deep=True):
        """Get parameters for this estimator.

        Parameters
        ----------
        deep : bool, default=True
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.

        Returns
        -------
        params : dict
            Parameter names mapped to their values
        """
        params = {
            'models': self.models,
            'method': self.method,
            'weights': self.weights,
            'meta_model': self.meta_model,
            'cv': self.cv
        }
        
        if deep:
            deep_items = []
            for k, v in params.items():
                if hasattr(v, 'get_params') and not isinstance(v, type):
                    deep_items.extend([
                        (f'{k}__{p}', val) for p, val in v.get_params().items()
                    ])
            params.update(dict(deep_items))
            
        return params
        
    def set_params(self, **params):
        """Set the parameters of this estimator.

        Parameters
        ----------
        **params : dict
            Estimator parameters

        Returns
        -------
        self : estimator instance
            Estimator instance
        """
        for key, value in params.items():
            if key in ['models', 'method', 'weights', 'meta_model', 'cv']:
                setattr(self, key, value)
            else:
                raise ValueError(f'Invalid parameter {key} for estimator {self}')
        return self

    def fit(self, X, y):
        """Fit the ensemble model.

        Parameters
        ----------
        X : array-like or pd.DataFrame
            Training data
        y : array-like
            Target values

        Returns
        -------
        self : object
            Returns self
        """
        # Convert data to numpy arrays
        X = np.asarray(X)
        y = np.asarray(y)

        # Determine if this is a classification task
        self.is_classifier = len(np.unique(y)) <= 10  # Arbitrary threshold

        # Store feature names if available
        if isinstance(X, pd.DataFrame):
            self.feature_names_ = X.columns.tolist()

        if self.method in ['voting', 'averaging', 'weighted']:
            # Fit each base model
            for name, model in self.models:
                model.fit(X, y)
        else:  # stacking
            # Generate cross-validation predictions
            cv_preds = np.zeros((X.shape[0], len(self.models)))
            for i, (name, model) in enumerate(self.models):
                # Get cross-validated predictions
                cv_pred = cross_val_predict(model, X, y, cv=self.cv)
                cv_preds[:, i] = cv_pred

            # Fit base models on full data
            for name, model in self.models:
                model.fit(X, y)

            # Fit meta-model
            self.meta_model.fit(cv_preds, y)

        self.is_fitted = True
        return self

    def predict(self, X):
        """Make predictions with the ensemble model.

        Parameters
        ----------
        X : array-like or pd.DataFrame
            Data to predict

        Returns
        -------
        array-like
            Predicted values
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")

        X = np.asarray(X)

        if self.method in ['voting', 'averaging', 'weighted']:
            # Get predictions from all models
            all_preds = np.array([model.predict(X) for name, model in self.models])
            
            # Combine predictions based on method
            if self.method == 'weighted':
                if self.weights is None:
                    return np.mean(all_preds, axis=0)
                return np.average(all_preds, axis=0, weights=self.weights)
            else:  # voting or averaging
                return np.mean(all_preds, axis=0)
        else:  # stacking
            # Get predictions from base models
            meta_features = np.column_stack([
                model.predict(X) for name, model in self.models
            ])
            return self.meta_model.predict(meta_features)

    def predict_proba(self, X):
        """Predict class probabilities for X.

        Parameters
        ----------
        X : array-like or pd.DataFrame
            Data to predict

        Returns
        -------
        array-like
            Predicted class probabilities
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")

        if not self.is_classifier:
            raise ValueError("predict_proba is only available for classification tasks")

        X = np.asarray(X)

        if self.method in ['voting', 'averaging', 'weighted']:
            probas = []
            for name, model in self.models:
                if hasattr(model, 'predict_proba'):
                    probas.append(model.predict_proba(X))
                else:
                    raise ValueError(f"Model {name} does not support predict_proba")
            
            probas = np.array(probas)
            
            # Combine probabilities
            if self.method == 'weighted':
                if self.weights is None:
                    return np.mean(probas, axis=0)
                return np.average(probas, axis=0, weights=self.weights)
            else:
                return np.mean(probas, axis=0)
        else:  # stacking
            # Get predictions from base models
            meta_features = np.column_stack([
                model.predict(X) for name, model in self.models
            ])
            
            if hasattr(self.meta_model, 'predict_proba'):
                return self.meta_model.predict_proba(meta_features)
            else:
                raise ValueError("Meta-model does not support predict_proba")

    def feature_importance(self):
        """Get feature importance from the ensemble model.

        Returns
        -------
        dict
            Feature importance for each base model and the ensemble overall
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting feature importance")

        importance = {}
        
        # Get feature importance from each base model
        for name, model in self.models:
            if hasattr(model, 'feature_importances_'):
                importance[name] = model.feature_importances_
            elif hasattr(model, 'coef_'):
                importance[name] = model.coef_
            else:
                importance[name] = None
        
        # Calculate ensemble importance as average of base models
        ensemble_importance = []
        for name, imp in importance.items():
            if imp is not None:
                ensemble_importance.append(imp)
        
        if ensemble_importance:
            importance['ensemble'] = np.mean(ensemble_importance, axis=0)
        
        return importance

    def cross_val_predict(self, X, y, cv=5):
        """Make cross-validated predictions with the ensemble model.

        Parameters
        ----------
        X : array-like or pd.DataFrame
            Data to predict
        y : array-like
            Target values
        cv : int, optional (default=5)
            Number of cross-validation folds

        Returns
        -------
        array-like
            Cross-validated predictions
        """
        # Direct use of sklearn's cross_val_predict
        return cross_val_predict(self, X, y, cv=cv)

    def save(self, path: str):
        """Save the ensemble model to a file.

        Parameters
        ----------
        path : str
            Path to save the model
        """
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str) -> 'EnsembleModel':
        """Load the ensemble model from a file.

        Parameters
        ----------
        path : str
            Path to load the model from

        Returns
        -------
        EnsembleModel
            Loaded ensemble model
        """
        return joblib.load(path)