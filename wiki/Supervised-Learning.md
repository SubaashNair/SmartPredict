# Supervised Learning

The Supervised Learning module in SmartPredict provides high-level interfaces for classification and regression tasks through the `SmartClassifier` and `SmartRegressor` classes.

## Overview

The Supervised Learning module offers:

- Training and evaluation of multiple models with a single function call
- Comprehensive performance metrics for each model
- Simplified model selection based on performance
- Integrated feature engineering and model explainability

## SmartClassifier

The `SmartClassifier` class is designed for classification tasks and provides an easy way to train and evaluate multiple classification models.

### Basic Usage

```python
from smartpredict import SmartClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

# Load the Breast Cancer dataset
data = load_breast_cancer()
X = data.data
y = data.target

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a SmartClassifier with multiple models
classifier = SmartClassifier(
    models=['RandomForestClassifier', 'LogisticRegression', 'GradientBoostingClassifier'],
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
```

### Parameters

The `SmartClassifier` class accepts the following parameters:

- `models`: List of model names to use (default: `['RandomForestClassifier', 'LogisticRegression']`)
- `custom_metric`: Custom scoring function with signature (y_true, y_pred)
- `verbose`: Verbosity level (0 = no output, 1 = basic output, 2 = detailed output)
- `ignore_warnings`: Whether to ignore warnings during model training
- Additional keyword arguments are passed to model constructors

### Available Models

The following classification models are available by default:

- `'RandomForestClassifier'`
- `'LogisticRegression'`
- `'DecisionTreeClassifier'`
- `'GradientBoostingClassifier'`
- `'KNeighborsClassifier'`
- `'SVC'`
- `'GaussianNB'`
- `'MLPClassifier'`
- `'AdaBoostClassifier'`
- `'XGBClassifier'` (if XGBoost is installed)
- `'LGBMClassifier'` (if LightGBM is installed)
- `'CatBoostClassifier'` (if CatBoost is installed)

### Methods

- `fit(X_train, X_test, y_train, y_test)`: Train models and evaluate on test data
- `predict(X)`: Make predictions with trained models
- `evaluate(X, y)`: Evaluate models on new data
- `score(X, y)`: Get performance scores for each model

## SmartRegressor

The `SmartRegressor` class is designed for regression tasks with an interface similar to `SmartClassifier`.

### Basic Usage

```python
from smartpredict import SmartRegressor
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

# Load the California Housing dataset
data = fetch_california_housing()
X = data.data
y = data.target

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a SmartRegressor with multiple models
regressor = SmartRegressor(
    models=['RandomForestRegressor', 'LinearRegression', 'GradientBoostingRegressor'],
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
```

### Parameters

The `SmartRegressor` class accepts the same parameters as `SmartClassifier`.

### Available Models

The following regression models are available by default:

- `'RandomForestRegressor'`
- `'LinearRegression'`
- `'Ridge'`
- `'Lasso'`
- `'ElasticNet'`
- `'SVR'`
- `'DecisionTreeRegressor'`
- `'GradientBoostingRegressor'`
- `'KNeighborsRegressor'`
- `'MLPRegressor'`
- `'XGBRegressor'` (if XGBoost is installed)
- `'LGBMRegressor'` (if LightGBM is installed)
- `'CatBoostRegressor'` (if CatBoost is installed)

### Methods

Same methods as `SmartClassifier` but with regression-specific metrics.

## Advanced Usage

### Custom Metrics

You can define custom metrics for model evaluation:

```python
from sklearn.metrics import f1_score
from smartpredict import SmartClassifier

# Define a custom metric function
def custom_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, average='macro')

# Use the custom metric
classifier = SmartClassifier(
    models=['RandomForestClassifier', 'LogisticRegression'],
    custom_metric=custom_f1,
    verbose=1
)
```

### Model Configuration

You can configure individual models by passing parameters as dictionary:

```python
classifier = SmartClassifier(
    models=['RandomForestClassifier', 'LogisticRegression'],
    RandomForestClassifier={
        'n_estimators': 200,
        'max_depth': 10,
        'random_state': 42
    },
    LogisticRegression={
        'C': 0.1,
        'max_iter': 1000,
        'random_state': 42
    }
)
```

### Integration with Other Modules

You can easily integrate supervised learning with other SmartPredict modules:

```python
from smartpredict import SmartClassifier
from smartpredict.feature_engineering import FeatureEngineer
from smartpredict.explainability import ModelExplainer

# Feature engineering
engineer = FeatureEngineer(scaler='standard', create_interactions=True)
X_train_transformed = engineer.fit_transform(X_train)
X_test_transformed = engineer.transform(X_test)

# Train classifier
classifier = SmartClassifier(models=['RandomForestClassifier'])
results = classifier.fit(X_train_transformed, X_test_transformed, y_train, y_test)

# Model explainability
best_model_name = max(results, key=lambda x: results[x]['accuracy'])
explainer = ModelExplainer(
    model=classifier.trained_models[best_model_name],
    feature_names=engineer.get_feature_names()
)
explainer.set_training_data(X_train_transformed)
importance = explainer.get_feature_importance()
```

This integration enables a complete machine learning workflow with minimal code.