# Ensemble Methods

The Ensemble Methods module in SmartPredict provides tools for creating and using ensemble models to improve prediction performance.

## Overview

Ensemble methods combine multiple machine learning models to produce better predictive performance than any individual model. SmartPredict's Ensemble Methods module offers:

- Various ensemble techniques (voting, averaging, weighted, stacking)
- Support for both classification and regression tasks
- Easy-to-use interface for creating ensembles
- Methods for analyzing ensemble performance and feature importance

## EnsembleModel Class

The main class in this module is `EnsembleModel`, which provides a flexible interface for creating different types of ensembles.

### Basic Usage

Here's a simple example of creating a voting ensemble:

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from smartpredict.ensemble_methods import EnsembleModel
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Load data
iris = load_iris()
X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create base models
rf = RandomForestClassifier(n_estimators=100, random_state=42)
lr = LogisticRegression(max_iter=1000, random_state=42)
svc = SVC(probability=True, random_state=42)

# Create ensemble model
ensemble = EnsembleModel(
    models=[('rf', rf), ('lr', lr), ('svc', svc)],
    method='voting'
)

# Train the ensemble
ensemble.fit(X_train, y_train)

# Make predictions
predictions = ensemble.predict(X_test)

# Evaluate the ensemble
from sklearn.metrics import accuracy_score
print(f"Ensemble Accuracy: {accuracy_score(y_test, predictions):.4f}")
```

### Parameters

The `EnsembleModel` class accepts the following parameters:

- `models`: List of tuples containing (name, model) pairs
- `method`: Method for combining predictions ('voting', 'averaging', 'weighted', or 'stacking')
- `weights`: List of weights for each model (only used for 'weighted' method)
- `meta_model`: Meta-learner model for stacking (required if method='stacking')
- `cv`: Number of cross-validation folds for stacking (default: 5)

### Ensemble Methods

#### Voting

In voting, each model casts a vote for the predicted class, and the class with the most votes is selected. For regression, it calculates the mean of predictions.

```python
ensemble = EnsembleModel(
    models=[('rf', rf), ('lr', lr), ('svc', svc)],
    method='voting'
)
```

#### Weighted

Weighted ensemble assigns different weights to each model's prediction:

```python
ensemble = EnsembleModel(
    models=[('rf', rf), ('lr', lr), ('svc', svc)],
    method='weighted',
    weights=[2, 1, 1]  # RandomForest is given twice the weight
)
```

#### Averaging

For regression tasks, averaging simply takes the mean of all model predictions:

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR

# Create base regression models
rf = RandomForestRegressor(n_estimators=100, random_state=42)
lr = LinearRegression()
svr = SVR()

# Create averaging ensemble
ensemble = EnsembleModel(
    models=[('rf', rf), ('lr', lr), ('svr', svr)],
    method='averaging'
)
```

#### Stacking

Stacking uses a meta-model that learns how to best combine the predictions from the base models:

```python
from sklearn.ensemble import GradientBoostingClassifier

# Create meta-learner
meta_learner = GradientBoostingClassifier(random_state=42)

# Create stacking ensemble
ensemble = EnsembleModel(
    models=[('rf', rf), ('lr', lr), ('svc', svc)],
    method='stacking',
    meta_model=meta_learner,
    cv=5  # 5-fold cross-validation for generating meta-features
)
```

### Methods

The `EnsembleModel` class provides the following methods:

#### fit(X, y)

Trains all base models (and meta-model for stacking) on the provided data.

```python
ensemble.fit(X_train, y_train)
```

#### predict(X)

Makes predictions using the ensemble method:

```python
predictions = ensemble.predict(X_test)
```

#### predict_proba(X)

Returns probability estimates for classification tasks:

```python
probabilities = ensemble.predict_proba(X_test)
```

#### feature_importance()

Returns the feature importance scores (if available):

```python
importance = ensemble.feature_importance()
print(importance)
```

#### cross_val_predict(X, y, cv=5)

Generates cross-validated predictions:

```python
cv_predictions = ensemble.cross_val_predict(X, y, cv=5)
```

#### save(path) and load(path)

Save and load trained ensemble models:

```python
# Save the model
ensemble.save('ensemble_model.pkl')

# Load the model
from smartpredict.ensemble_methods import EnsembleModel
loaded_ensemble = EnsembleModel.load('ensemble_model.pkl')
```

## Practical Example

Here's a complete example using stacking with the Breast Cancer dataset:

```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from smartpredict.ensemble_methods import EnsembleModel

# Load data
cancer = load_breast_cancer()
X = cancer.data
y = cancer.target
feature_names = cancer.feature_names

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create base models
rf = RandomForestClassifier(n_estimators=100, random_state=42)
gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
lr = LogisticRegression(max_iter=1000, random_state=42)

# Train individual models for comparison
rf.fit(X_train, y_train)
gb.fit(X_train, y_train)
lr.fit(X_train, y_train)

# Create meta-learner
meta_model = LogisticRegression(random_state=42)

# Create stacking ensemble
ensemble = EnsembleModel(
    models=[('rf', rf), ('gb', gb), ('lr', lr)],
    method='stacking',
    meta_model=meta_model
)

# Train ensemble
ensemble.fit(X_train, y_train)

# Make predictions
rf_pred = rf.predict(X_test)
gb_pred = gb.predict(X_test)
lr_pred = lr.predict(X_test)
ensemble_pred = ensemble.predict(X_test)

# Compare performance
print(f"Random Forest Accuracy: {accuracy_score(y_test, rf_pred):.4f}")
print(f"Gradient Boosting Accuracy: {accuracy_score(y_test, gb_pred):.4f}")
print(f"Logistic Regression Accuracy: {accuracy_score(y_test, lr_pred):.4f}")
print(f"Ensemble Accuracy: {accuracy_score(y_test, ensemble_pred):.4f}")

# Detailed ensemble report
print("\nEnsemble Classification Report:")
print(classification_report(y_test, ensemble_pred))
```

This example demonstrates how ensemble methods can often achieve better performance than individual models.