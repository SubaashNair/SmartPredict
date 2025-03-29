# smartpredict/base.py
"""
Base module for SmartPredict.
Defines the BasePredictor class to be inherited by specific predictors.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Callable, Union
import logging
import numpy as np  # noqa: F401 - used in type hints
import numpy.typing as npt
import pandas as pd

class BaseModel(ABC):
    """Base class for all models in SmartPredict."""
    
    def __init__(self):
        self._is_fitted = False

    @abstractmethod
    def fit(self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series]) -> 'BaseModel':
        """Fit the model."""
        pass

    @abstractmethod
    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> Any:
        """Make predictions."""
        pass

    @abstractmethod
    def evaluate(self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series]) -> Dict[str, Any]:
        """Evaluate the model."""
        pass

    @abstractmethod
    def score(self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series]) -> Union[float, Dict[str, float]]:
        """Calculate the model's score."""
        pass

    @abstractmethod
    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """Get model parameters."""
        pass

    @abstractmethod
    def set_params(self, **params) -> 'BaseModel':
        """Set model parameters."""
        pass

class BasePredictor(ABC):
    """
    An abstract base class for SmartPredict predictors.

    Parameters
    ----------
    verbose : int, default=0
        Level of verbosity.
    ignore_warnings : bool, default=True
        Whether to ignore warnings.
    custom_metric : callable, optional
        A custom evaluation metric function.

    Attributes
    ----------
    verbose : int
        Level of verbosity.
    ignore_warnings : bool
        Whether to ignore warnings.
    custom_metric : callable
        Custom evaluation metric function.
    logger : logging.Logger
        Logger instance for the class.
    """
    def __init__(
        self,
        verbose: int = 0,
        ignore_warnings: bool = True,
        custom_metric: Optional[Callable] = None
    ) -> None:
        self.verbose = verbose
        self.ignore_warnings = ignore_warnings
        self.custom_metric = custom_metric
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(
            level=logging.INFO if verbose > 0 else logging.WARNING,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

    @abstractmethod
    def fit(
        self,
        X_train: npt.NDArray[Any],
        X_test: npt.NDArray[Any],
        y_train: npt.NDArray[Any],
        y_test: npt.NDArray[Any]
    ) -> Dict[str, Any]:
        """
        Abstract method to fit the model using training data.

        Parameters
        ----------
        X_train : array-like
            Training features.
        X_test : array-like
            Testing features.
        y_train : array-like
            Training labels.
        y_test : array-like
            Testing labels.

        Returns
        -------
        dict
            Model performance results.
        """
        pass

    @abstractmethod
    def evaluate_model(
        self,
        y_test: npt.NDArray[Any],
        predictions: npt.NDArray[Any]
    ) -> Dict[str, Any]:
        """
        Abstract method to evaluate model performance.

        Parameters
        ----------
        y_test : array-like
            True labels/values.
        predictions : array-like
            Predicted labels/values.

        Returns
        -------
        dict
            Evaluation metrics.
        """
        pass