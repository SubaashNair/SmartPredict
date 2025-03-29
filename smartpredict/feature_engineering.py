# smartpredict/feature_engineering.py
"""
Feature engineering module for SmartPredict.
Provides automated feature engineering capabilities.
"""

from typing import Tuple, Union, List, Optional, Dict, Any
import numpy as np
import pandas as pd
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
    PowerTransformer,
    QuantileTransformer,
    OneHotEncoder,
    LabelEncoder,
    TargetEncoder
)
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif, f_regression, VarianceThreshold
from itertools import combinations
from sklearn.feature_selection import mutual_info_regression

class FeatureEngineer:
    """Feature engineering operations.
    
    Parameters
    ----------
    numeric_features : list, optional
        List of numeric feature names to process
    categorical_features : list, optional
        List of categorical feature names to process
    binary_features : list, optional
        List of binary feature names to process
    scaler : str, optional
        Type of scaler to use ('standard', 'minmax', 'robust', or None)
    encoder : str, optional
        Type of encoder to use ('onehot', 'label', 'target', or None)
    handle_missing : str, optional
        Strategy for handling missing values ('mean', 'median', 'most_frequent', 'constant')
    n_features_to_select : int, optional
        Number of features to select using feature selection
    auto_detect : bool, optional
        Whether to automatically detect feature types
    create_interactions : bool, optional
        Whether to create interaction features between numeric features
    
    Attributes
    ----------
    fitted : bool
        Whether the transformer has been fitted
    """
    def __init__(self, numeric_features=None, categorical_features=None, binary_features=None,
                 scaler=None, encoder=None, handle_missing='mean',
                 n_features_to_select=None, auto_detect=False, create_interactions=False):
        self.numeric_features = numeric_features or []
        self.categorical_features = categorical_features or []
        self.binary_features = binary_features or []
        
        # Validate scaler
        valid_scalers = [None, 'standard', 'minmax', 'robust']
        if scaler not in valid_scalers:
            raise ValueError(f"scaler must be one of {valid_scalers}")
        self.scaler = scaler
        
        # Validate encoder
        valid_encoders = [None, 'onehot', 'label', 'target']
        if encoder not in valid_encoders:
            raise ValueError(f"encoder must be one of {valid_encoders}")
        self.encoder = encoder
        
        # Validate missing value handling
        valid_missing = ['mean', 'median', 'most_frequent', 'constant']
        if handle_missing not in valid_missing:
            raise ValueError(f"handle_missing must be one of {valid_missing}")
        self.handle_missing = handle_missing
        
        self.n_features_to_select = n_features_to_select
        self.auto_detect = auto_detect
        self.create_interactions = create_interactions
        self.fitted = False

        # Initialize transformers
        self.numeric_imputer = None
        self.categorical_imputer = None
        self.numeric_scaler = None
        self.categorical_encoder = None
        self.feature_selector = None
        self._create_transformers()

    def _detect_feature_types(self, X):
        """Automatically detect feature types from data.
        
        Parameters
        ----------
        X : pd.DataFrame
            Input data
        """
        for column in X.columns:
            if X[column].dtype in ['int64', 'float64']:
                if set(X[column].dropna().unique()) == {0, 1}:
                    self.binary_features.append(column)
                else:
                    self.numeric_features.append(column)
            else:
                self.categorical_features.append(column)

    def _create_transformers(self):
        """Create and configure the transformers."""
        # Create imputers
        if self.numeric_features:
            self.numeric_imputer = SimpleImputer(strategy=self.handle_missing)
        
        if self.categorical_features:
            self.categorical_imputer = SimpleImputer(strategy='constant', fill_value='missing')
        
        # Create scaler
        if self.scaler == 'standard':
            self.numeric_scaler = StandardScaler()
        elif self.scaler == 'minmax':
            self.numeric_scaler = MinMaxScaler()
        elif self.scaler == 'robust':
            self.numeric_scaler = RobustScaler()
        
        # Create encoder
        if self.encoder == 'onehot':
            self.categorical_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        elif self.encoder == 'label':
            self.categorical_encoder = LabelEncoder()
        elif self.encoder == 'target':
            self.categorical_encoder = TargetEncoder()
        
        # Create feature selector
        if self.n_features_to_select is not None:
            self.feature_selector = None  # Will be set in fit based on y availability

    def _create_interactions(self, X):
        """Create interaction features between numeric features.
        
        Parameters
        ----------
        X : pd.DataFrame
            Input data
            
        Returns
        -------
        pd.DataFrame
            Interaction features
        """
        if not self.create_interactions or len(self.numeric_features) < 2:
            return pd.DataFrame()
            
        interactions = pd.DataFrame(index=X.index)
        for f1, f2 in combinations(self.numeric_features, 2):
            feature_name = f"{f1}_x_{f2}"
            interactions[feature_name] = X[f1] * X[f2]
        return interactions

    def fit(self, X, y=None):
        """Fit the feature engineering transformers.
    
        Parameters
        ----------
        X : array-like or pd.DataFrame
            Training data
        y : array-like, optional
            Target values for target encoding or feature selection
    
        Returns
        -------
        self : object
            Returns self
        """
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])

        # Auto-detect feature types if requested
        if self.auto_detect:
            self._detect_feature_types(X)
            self._create_transformers()  # Recreate transformers with detected features

        # Fit numeric features
        if self.numeric_features:
            if self.numeric_imputer:
                self.numeric_imputer.fit(X[self.numeric_features])
            if self.numeric_scaler:
                self.numeric_scaler.fit(X[self.numeric_features])

        # Fit categorical features
        if self.categorical_features:
            if self.categorical_imputer:
                self.categorical_imputer.fit(X[self.categorical_features])
            if self.categorical_encoder:
                if self.encoder == 'target' and y is None:
                    raise ValueError("Target values required for target encoding")
                encoded_data = self.categorical_imputer.transform(X[self.categorical_features])
                if self.encoder == 'target':
                    self.categorical_encoder.fit(encoded_data, y)
                else:
                    self.categorical_encoder.fit(encoded_data)

        # Fit feature selector if needed
        if self.n_features_to_select is not None:
            # Get all features after preprocessing
            processed_data = self.transform(X, check_fitted=False)
            
            # Choose selector based on y availability
            if y is not None:
                self.feature_selector = SelectKBest(score_func=f_regression, k=self.n_features_to_select)
                self.feature_selector.fit(processed_data, y)
            else:
                # Calculate variances for each feature
                variances = processed_data.var()
                # Get top k features by variance
                selected_features = variances.nlargest(self.n_features_to_select).index.tolist()
                # Create a selector that keeps only these features
                self.feature_selector = type('VarianceSelector', (), {
                    'selected_features': selected_features,
                    'transform': lambda self, X: X[self.selected_features],
                    'get_support': lambda self: [col in self.selected_features for col in X.columns]
                })()

        self.fitted = True
        return self
    
    def transform(self, X, check_fitted=True):
        """Transform the input data.
        
        Parameters
        ----------
        X : array-like or pd.DataFrame
            Data to transform
        check_fitted : bool, optional (default=True)
            Whether to check if the transformer is fitted
        
        Returns
        -------
        pd.DataFrame : Transformed data
        """
        if check_fitted and not self.fitted:
            raise ValueError("FeatureEngineer must be fitted before transform")
            
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
        
        result = pd.DataFrame(index=X.index)
        
        # Transform numeric features
        if self.numeric_features:
            numeric_data = X[self.numeric_features].copy()
            
            # Apply imputer to numeric data
            if self.numeric_imputer:
                numeric_data = pd.DataFrame(
                    self.numeric_imputer.transform(numeric_data),
                    columns=self.numeric_features,
                    index=X.index
                )
            
            # Apply scaler to numeric data
            if self.numeric_scaler:
                numeric_data = pd.DataFrame(
                    self.numeric_scaler.transform(numeric_data),
                    columns=self.numeric_features,
                    index=X.index
                )
            
            # Add transformed numeric data to result
            result = pd.concat([result, numeric_data], axis=1)
        
        # Transform categorical features
        if self.categorical_features:
            categorical_data = X[self.categorical_features].copy()
            
            # Apply imputer to categorical data
            if self.categorical_imputer:
                categorical_data = pd.DataFrame(
                    self.categorical_imputer.transform(categorical_data),
                    columns=self.categorical_features,
                    index=X.index
                )
            
            # Apply encoder to categorical data
            if self.categorical_encoder:
                if self.encoder == 'onehot':
                    encoded_data = self.categorical_encoder.transform(categorical_data)
                    encoded_col_names = self.categorical_encoder.get_feature_names_out(self.categorical_features)
                    encoded_data = pd.DataFrame(
                        encoded_data,
                        columns=encoded_col_names,
                        index=X.index
                    )
                    result = pd.concat([result, encoded_data], axis=1)
                else:
                    # For other encoders
                    for col in self.categorical_features:
                        encoded = self.categorical_encoder.transform(categorical_data[[col]])
                        result[col] = encoded
            else:
                result = pd.concat([result, categorical_data], axis=1)
        
        # Add binary features as is
        if self.binary_features:
            binary_data = X[self.binary_features].copy()
            result = pd.concat([result, binary_data], axis=1)
        
        # Add interaction features
        if self.create_interactions:
            interactions = self._create_interactions(X)
            if not interactions.empty:
                result = pd.concat([result, interactions], axis=1)
        
        # Apply feature selection if needed
        if self.feature_selector is not None:
            if hasattr(self.feature_selector, 'transform'):
                result = self.feature_selector.transform(result)
                if isinstance(result, np.ndarray):
                    # Convert back to DataFrame with feature names
                    selected_indices = self.feature_selector.get_support()
                    selected_cols = [col for i, col in enumerate(result.columns) if selected_indices[i]]
                    result = pd.DataFrame(result, columns=selected_cols, index=X.index)
            else:
                # Handle our custom selector
                result = pd.DataFrame(
                    result[self.feature_selector.selected_features],
                    index=X.index
                )
        
        return result
    
    def fit_transform(self, X, y=None):
        """Fit and transform the data.
        
        Parameters
        ----------
        X : array-like or pd.DataFrame
            Training data
        y : array-like, optional
            Target values
            
        Returns
        -------
        pd.DataFrame : Transformed data
        """
        return self.fit(X, y).transform(X)
    
    def get_feature_names(self):
        """Get the names of the transformed features.
        
        Returns
        -------
        list : Feature names
        """
        feature_names = []
        
        # Numeric feature names
        if self.numeric_features:
            feature_names.extend(self.numeric_features)
        
        # Categorical feature names
        if self.categorical_features and self.categorical_encoder:
            if self.encoder == 'onehot':
                # Get one-hot encoded feature names
                feature_names.extend(self.categorical_encoder.get_feature_names_out(self.categorical_features))
            else:
                # For other encoders, keep original names
                feature_names.extend(self.categorical_features)
        elif self.categorical_features:
            feature_names.extend(self.categorical_features)
        
        # Binary feature names
        if self.binary_features:
            feature_names.extend(self.binary_features)
        
        # Interaction feature names
        if self.create_interactions and len(self.numeric_features) >= 2:
            for f1, f2 in combinations(self.numeric_features, 2):
                feature_names.append(f"{f1}_x_{f2}")
        
        # Filter by feature selection if needed
        if self.feature_selector is not None:
            if hasattr(self.feature_selector, 'get_support'):
                mask = self.feature_selector.get_support()
                feature_names = [name for i, name in enumerate(feature_names) if i < len(mask) and mask[i]]
            else:
                # Handle our custom selector
                feature_names = [name for name in feature_names if name in self.feature_selector.selected_features]
        
        return feature_names


def engineer_features(
    X_train: Union[pd.DataFrame, np.ndarray],
    X_test: Union[pd.DataFrame, np.ndarray],
    **kwargs
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply feature engineering to training and test data.
    
    Parameters
    ----------
    X_train : array-like or pd.DataFrame
        Training data
    X_test : array-like or pd.DataFrame
        Test data
    **kwargs : dict
        Parameters to pass to FeatureEngineer
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Transformed training and test data
    """
    engineer = FeatureEngineer(**kwargs)
    X_train_transformed = engineer.fit_transform(X_train)
    X_test_transformed = engineer.transform(X_test)
    
    return X_train_transformed, X_test_transformed