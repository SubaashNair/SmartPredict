# Quick Start Tutorial

This tutorial provides a quick introduction to SmartPredict using scikit-learn datasets for classification and regression tasks.

## Classification Example

Let's start with a simple classification example using the Iris dataset:

```python
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from smartpredict import SmartClassifier

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize SmartClassifier with multiple models
classifier = SmartClassifier(
    models=['RandomForestClassifier', 'LogisticRegression', 'SVC'],
    verbose=1
)

# Train and evaluate the models
results = classifier.fit(X_train, X_test, y_train, y_test)

# Print the results
for model_name, metrics in results.items():
    print(f"{model_name}:")
    for metric_name, value in metrics.items():
        print(f"  {metric_name}: {value:.4f}")
    print()

# Make predictions with the best model
best_model_name = max(results, key=lambda x: results[x]['accuracy'])
predictions = classifier.predict(X_test)[best_model_name]

# Get feature importance for the best model
from smartpredict.explainability import ModelExplainer
explainer = ModelExplainer(
    model=classifier.trained_models[best_model_name],
    feature_names=iris.feature_names
)
explainer.set_training_data(X_train)
importance = explainer.get_feature_importance()
print("Feature Importance:")
print(importance)
```

## Regression Example

Now let's try a regression example using the Boston Housing dataset:

```python
from sklearn.datasets import fetch_california_housing
from smartpredict import SmartRegressor

# Load the California Housing dataset
housing = fetch_california_housing()
X = housing.data
y = housing.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize SmartRegressor with multiple models
regressor = SmartRegressor(
    models=['RandomForestRegressor', 'LinearRegression', 'SVR'],
    verbose=1
)

# Train and evaluate the models
results = regressor.fit(X_train, X_test, y_train, y_test)

# Print the results
for model_name, metrics in results.items():
    print(f"{model_name}:")
    for metric_name, value in metrics.items():
        print(f"  {metric_name}: {value:.4f}")
    print()

# Make predictions with the best model
best_model_name = max(results, key=lambda x: results[x]['r2'])
predictions = regressor.predict(X_test)[best_model_name]
```

## Feature Engineering Example

SmartPredict includes automated feature engineering capabilities:

```python
import pandas as pd
from smartpredict.feature_engineering import FeatureEngineer

# Convert data to a DataFrame
X_train_df = pd.DataFrame(X_train, columns=housing.feature_names)
X_test_df = pd.DataFrame(X_test, columns=housing.feature_names)

# Initialize feature engineer
feature_engineer = FeatureEngineer(
    numeric_features=housing.feature_names,
    scaler='standard',
    handle_missing='mean',
    create_interactions=True
)

# Fit and transform the data
X_train_engineered = feature_engineer.fit_transform(X_train_df)
X_test_engineered = feature_engineer.transform(X_test_df)

# Train a model with engineered features
regressor = SmartRegressor(models=['RandomForestRegressor'])
results = regressor.fit(X_train_engineered, X_test_engineered, y_train, y_test)
print(results)
```

## Hyperparameter Tuning Example

Optimize model parameters with SmartPredict's hyperparameter tuning:

```python
from sklearn.ensemble import RandomForestRegressor
from smartpredict.hyperparameter_tuning import tune_hyperparameters

# Define parameter space
param_space = {
    'n_estimators': (50, 200),
    'max_depth': (3, 15),
    'min_samples_split': (2, 10),
    'min_samples_leaf': (1, 5)
}

# Create base model
base_model = RandomForestRegressor(random_state=42)

# Tune hyperparameters
optimized_model = tune_hyperparameters(
    model=base_model,
    param_distributions=param_space,
    X=X_train,
    y=y_train,
    n_trials=50,
    scoring='r2',
    random_state=42
)

# Print optimized parameters
print("Optimized Parameters:")
print(optimized_model.get_params())
```

## Ensemble Methods Example

Create ensemble models with SmartPredict:

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from smartpredict.ensemble_methods import EnsembleModel

# Create base models
rf = RandomForestClassifier(n_estimators=100, random_state=42)
lr = LogisticRegression(max_iter=1000, random_state=42)
svc = SVC(probability=True, random_state=42)

# Create ensemble model
ensemble = EnsembleModel(
    models=[('rf', rf), ('lr', lr), ('svc', svc)],
    method='voting',
    weights=[2, 1, 1]  # Weighting the RandomForest more heavily
)

# Train the ensemble
ensemble.fit(X_train, y_train)

# Make predictions
ensemble_predictions = ensemble.predict(X_test)

# Calculate accuracy
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, ensemble_predictions)
print(f"Ensemble Accuracy: {accuracy:.4f}")
```

These examples demonstrate the core functionality of SmartPredict. Explore the rest of the documentation for more detailed information on each component.