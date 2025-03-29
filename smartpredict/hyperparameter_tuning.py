# smartpredict/hyperparameter_tuning.py
"""
Hyperparameter tuning module for SmartPredict.
Provides functions to perform hyperparameter optimization.
"""

import optuna
from sklearn.model_selection import cross_val_score, KFold
import numpy as np
import logging

def tune_hyperparameters(model, param_distributions, X, y, n_trials=100, scoring=None, random_state=42):
    """Tune hyperparameters using Optuna.

    Args:
        model: The model to tune
        param_distributions: Dictionary of parameter names and their distributions
        X: Training data features
        y: Training data target
        n_trials: Number of trials for optimization
        scoring: Scoring metric to use
        random_state: Random state for reproducibility

    Returns:
        A fitted model with the best found parameters
    """
    # Validate parameters
    if not isinstance(param_distributions, dict):
        raise ValueError("param_distributions must be a dictionary")
    
    # Set a fixed random seed for reproducibility
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    # Create a cross-validation splitter with fixed random state for reproducibility
    cv = KFold(n_splits=5, shuffle=True, random_state=random_state)
    
    def objective(trial):
        params = {}
        for param_name, param_range in param_distributions.items():
            if isinstance(param_range, tuple) and len(param_range) == 2:
                if isinstance(param_range[0], int):
                    params[param_name] = trial.suggest_int(param_name, param_range[0], param_range[1])
                else:
                    params[param_name] = trial.suggest_float(param_name, param_range[0], param_range[1])
            elif isinstance(param_range, list):
                params[param_name] = trial.suggest_categorical(param_name, param_range)

        # If the model accepts a random_state parameter, set it for reproducibility
        model_params = model.get_params()
        if 'random_state' in model_params:
            params['random_state'] = random_state
            
        model.set_params(**params)
        try:
            scores = cross_val_score(model, X, y, scoring=scoring, cv=cv)
            return scores.mean()
        except Exception as e:
            # Handle common errors like mismatched dimensions
            if "shape" in str(e).lower() or "dimension" in str(e).lower():
                raise ValueError(f"Error with data dimensions: {str(e)}")
            raise

    # Create a study with a fixed random seed
    sampler = optuna.samplers.TPESampler(seed=random_state)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials)

    # Get the best parameters and create a new model instance
    best_params = study.best_params
    
    # Ensure random_state is preserved if the model supports it
    model_params = model.get_params()
    if 'random_state' in model_params and 'random_state' not in best_params:
        best_params['random_state'] = random_state
    
    # Create new instance with original model parameters plus best parameters
    best_model = model.__class__(**{**model.get_params(), **best_params})
    best_model.fit(X, y)

    return best_model