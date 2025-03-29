# SmartPredict

[![PyPI version](https://badge.fury.io/py/smartpredict.svg)](https://pypi.org/project/smartpredict/)
[![Build Status](https://github.com/SubaashNair/SmartPredict/actions/workflows/pypi-publish.yml/badge.svg)](https://github.com/SubaashNair/SmartPredict/actions/workflows/pypi-publish.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

SmartPredict is an advanced machine learning library designed to simplify model training, evaluation, and selection. It provides a comprehensive set of tools for classification and regression tasks, including automated hyperparameter tuning, feature engineering, ensemble methods, and model explainability.

## Table of Contents

- [Installation](#installation)
- [Features](#features)
- [Quick Start](#quick-start)
- [Usage](#usage)
  - [Classification](#classification)
  - [Regression](#regression)
- [Advanced Features](#advanced-features)
  - [Feature Engineering](#feature-engineering)
  - [Ensemble Methods](#ensemble-methods)
  - [Hyperparameter Tuning](#hyperparameter-tuning)
  - [Explainability](#explainability)
- [Contributing](#contributing)
- [License](#license)

## Installation

You can install SmartPredict using pip:

```bash
pip install smartpredict
```

## Features

- **Unified API for ML Models**: Provides a consistent interface for both classification and regression tasks
- **Automated Feature Engineering**: Handles missing values, scaling, encoding, feature interactions, and selection
- **Robust Ensemble Methods**: Supports voting, averaging, weighted combining, and stacking approaches
- **Hyperparameter Tuning**: Uses Optuna for efficient reproducible hyperparameter optimization
- **Model Explainability**: Provides SHAP-based explanations and feature importance analysis
- **Comprehensive Error Handling**: Gracefully handles common errors during model training and evaluation

## Quick Start

Here's a quick example to get you started:

```python
from smartpredict import SmartClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

# Load and split data
data = load_breast_cancer()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create classifier and fit models
clf = SmartClassifier(
    models=['RandomForestClassifier', 'LogisticRegression'], 
    verbose=1
)
results = clf.fit(X_train, X_test, y_train, y_test)

# Display model performance results
print(results)

# Make predictions with all trained models
predictions = clf.predict(X_test)
```

## Usage

### Classification

```python
from smartpredict import SmartClassifier

# Create classifier with custom models and parameters
clf = SmartClassifier(
    models=['RandomForestClassifier', 'LogisticRegression', 'SVC'],
    # Pass custom parameters for each model
    RandomForestClassifier={'n_estimators': 200, 'max_depth': 10},
    LogisticRegression={'C': 0.1, 'max_iter': 200},
    verbose=1
)

# Fit and evaluate all models
results = clf.fit(X_train, X_test, y_train, y_test)

# The best model is automatically selected for predictions
predictions = clf.predict(new_data)
```

### Regression

```python
from smartpredict import SmartRegressor

# Create regressor with custom models
reg = SmartRegressor(
    models=['RandomForestRegressor', 'LinearRegression', 'SVR'],
    # Pass custom parameters for a specific model
    RandomForestRegressor={'n_estimators': 200, 'max_depth': 15},
    verbose=1
)

# Fit and evaluate all models
results = reg.fit(X_train, X_test, y_train, y_test)

# The best model is automatically selected for predictions
predictions = reg.predict(new_data)
```

## Advanced Features

### Feature Engineering

```python
from smartpredict.feature_engineering import FeatureEngineer

# Create feature engineer
fe = FeatureEngineer(
    scaler='standard',
    encoder='onehot',
    handle_missing='mean',
    create_interactions=True,
    feature_selection=5  # Keep top 5 features
)

# Fit and transform data
X_transformed = fe.fit_transform(X_train)
X_test_transformed = fe.transform(X_test)
```

### Ensemble Methods

```python
from smartpredict.ensemble_methods import EnsembleModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# Create base models
models = [
    ('rf', RandomForestClassifier(n_estimators=100)),
    ('lr', LogisticRegression())
]

# Create ensemble with voting method
ensemble = EnsembleModel(
    models=models,
    method='voting'  # 'voting', 'averaging', 'weighted', or 'stacking'
)

# Fit ensemble
ensemble.fit(X_train, y_train)

# Make predictions
predictions = ensemble.predict(X_test)
```

### Hyperparameter Tuning

```python
from smartpredict.hyperparameter_tuning import tune_hyperparameters
from sklearn.ensemble import RandomForestClassifier

# Create base model
model = RandomForestClassifier()

# Define parameter distributions to search
param_dist = {
    'n_estimators': (50, 300),
    'max_depth': (3, 15),
    'min_samples_split': (2, 10)
}

# Tune hyperparameters
best_model = tune_hyperparameters(
    model=model,
    param_distributions=param_dist,
    X=X_train,
    y=y_train,
    n_trials=100,
    scoring='f1',
    random_state=42
)

# Use the optimized model
predictions = best_model.predict(X_test)
```

### Explainability

```python
from smartpredict.explainability import ModelExplainer

# Create explainer
explainer = ModelExplainer(
    model=trained_model,
    feature_names=feature_names
)

# Set training data (needed for some explanation methods)
explainer.set_training_data(X_train, y_train)

# Get feature importance
importance_df = explainer.get_feature_importance()
print(importance_df)

# Explain a prediction
explanation = explainer.explain_prediction(X_test[0])
print(explanation)
```

## Contributing

We welcome contributions! Please feel free to submit a Pull Request.

## License

SmartPredict is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.