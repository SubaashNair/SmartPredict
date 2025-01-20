# smartpredict/hyperparameter_tuning.py
"""
Hyperparameter tuning module for SmartPredict.
Provides functions to perform hyperparameter optimization.
"""

import optuna
from sklearn.model_selection import cross_val_score

def tune_hyperparameters(model, param_distributions, X_train, y_train):
    def objective(trial):
        params = {
            'n_estimators':trial.suggest_int('n_estimators',10,100),
            'max_depth':trial.suggest_int('max_depth',1,10),
        }
        model.set_params(**params)
        score = cross_val_score(model,X_train,y_train,cv=5).mean()
        return score
    
    study = optuna.create_study(direction='maximize')
    study.optimize(objective,n_trials=100)
    best_params = study.best_params
    model.set_params(**best_params)
    model.fit(X_train,y_train)
    return model