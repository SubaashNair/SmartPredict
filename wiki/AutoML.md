# AutoML

The AutoML module in SmartPredict provides a fully automated machine learning pipeline that handles everything from target detection to model selection, training, and evaluation.

## Overview

AutoML is designed to make machine learning as simple as possible, requiring minimal code and configuration. It automatically:

- Detects the target variable and features
- Identifies the task type (classification or regression)
- Configures appropriate feature engineering
- Handles train/test splitting
- Selects and evaluates multiple models
- Returns the best performing model with explanations

## Basic Usage

Using AutoML is as simple as:

```python
from smartpredict import AutoML

# Initialize AutoML
auto_ml = AutoML()

# Fit on your dataset (features and target in one DataFrame or array)
result = auto_ml.fit(data)

# Make predictions with the best model
predictions = result.predict(new_data)
```

## Parameters

The `AutoML` class accepts the following parameters:

- `time_budget`: Maximum time in seconds for the AutoML process (optional)
- `optimize_for`: Metric to optimize ('accuracy', 'f1', etc. for classification; 'r2', 'mse', etc. for regression)
- `target_column`: Name (for DataFrame) or index (for array) of target column
- `verbose`: Verbosity level (0 = silent, 1 = basic info, 2 = detailed info)

## Examples

### Classification Example

```python
import pandas as pd
from sklearn.datasets import load_iris
from smartpredict import AutoML

# Load Iris dataset
iris = load_iris()
data = pd.DataFrame(
    data=iris.data,
    columns=iris.feature_names
)
data['target'] = iris.target

# Run AutoML
auto_ml = AutoML(verbose=1)
result = auto_ml.fit(data)

# Check results
print(f"Best model: {result.best_model_name}")
print(f"Accuracy: {result.metrics.get('accuracy', 'N/A'):.4f}")

# Get feature importance
importance = result.explain()['feature_importance']
print("Top features:")
print(importance.sort_values('importance', ascending=False).head())

# Make predictions
new_data = data.drop('target', axis=1).iloc[:1]
prediction = result.predict(new_data)
print(f"Predicted class: {prediction[0]}")
```

### Regression Example

```python
import pandas as pd
from sklearn.datasets import fetch_california_housing
from smartpredict import AutoML

# Load California Housing dataset
housing = fetch_california_housing()
data = pd.DataFrame(
    data=housing.data,
    columns=housing.feature_names
)
data['target'] = housing.target

# Run AutoML
auto_ml = AutoML(verbose=1)
result = auto_ml.fit(data)

# Check results
print(f"Best model: {result.best_model_name}")
print(f"R² score: {result.metrics.get('r2', 'N/A'):.4f}")
print(f"RMSE: {result.metrics.get('rmse', 'N/A'):.4f}")

# Make predictions
new_data = data.drop('target', axis=1).iloc[:1]
prediction = result.predict(new_data)
print(f"Predicted value: {prediction[0]:.4f}")
```

### Using NumPy Arrays

```python
import numpy as np
from smartpredict import AutoML

# Create synthetic data (last column is the target)
X = np.random.rand(100, 5)  # Features
y = 3*X[:, 0] - 2*X[:, 1] + np.random.normal(0, 0.1, 100)  # Target
data = np.column_stack((X, y))

# Run AutoML
auto_ml = AutoML()
result = auto_ml.fit(data)

# Make predictions
new_sample = X[0:1]
prediction = result.predict(new_sample)
print(f"Prediction: {prediction[0]:.4f}")
```

## Target Detection

AutoML automatically identifies the target variable:

- For pandas DataFrames:
  - Uses the specified `target_column` if provided
  - Otherwise, looks for columns named 'target', 'label', 'y', 'class', or 'outcome'
  - Falls back to using the last column if no match is found

- For NumPy arrays:
  - Uses the specified column index if `target_column` is provided
  - Otherwise, assumes the last column is the target

## Task Type Detection

AutoML automatically determines whether the problem is classification or regression:

- If the target has fewer than 10 unique values or is categorical, it's treated as classification
- Otherwise, it's treated as regression

## Feature Engineering

AutoML performs automatic feature engineering:

- Automatically detects numeric and categorical features
- Applies appropriate scaling and encoding
- Handles missing values
- Creates feature interactions
- Selects important features

## AutoMLResult Class

The `fit()` method returns an `AutoMLResult` object with the following properties and methods:

### Properties

- `best_model`: The best performing model
- `best_model_name`: Name of the best model
- `metrics`: Performance metrics for the best model
- `task_type`: Type of task ('classification' or 'regression')
- `feature_engineer`: The feature engineering pipeline
- `all_models`: Dictionary of all trained models
- `all_results`: Performance metrics for all models
- `explainer`: Model explainer for interpretability

### Methods

- `predict(X)`: Make predictions using the best model
- `predict_proba(X)`: Get probability predictions (classification only)
- `explain(X=None)`: Get model explanations
- `save(path)`: Save the AutoML result to disk
- `load(path)`: Static method to load a saved AutoML result

## Advanced Usage

### Optimizing for Different Metrics

```python
# Optimize for F1 score instead of accuracy
auto_ml = AutoML(optimize_for='f1')
result = auto_ml.fit(data)

# Optimize for Mean Absolute Error for regression
auto_ml = AutoML(optimize_for='mae')
result = auto_ml.fit(data)
```

### Time Budget

```python
# Limit AutoML runtime to 60 seconds
auto_ml = AutoML(time_budget=60)
result = auto_ml.fit(data)
```

### Saving and Loading Models

```python
# Save the model
result.save("my_model.pkl")

# Load the model later
from smartpredict import AutoMLResult
loaded_result = AutoMLResult.load("my_model.pkl")

# Use the loaded model
predictions = loaded_result.predict(new_data)
```

## Limitations

- Currently, AutoML focuses on tabular data
- For extremely large datasets, consider subsetting first
- Some advanced features like multi-target prediction are not yet supported

The AutoML module makes machine learning accessible to everyone regardless of expertise level while still providing the flexibility needed by experienced practitioners.