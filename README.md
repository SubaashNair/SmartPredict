
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
  - [Hyperparameter Tuning](#hyperparameter-tuning)
  - [Feature Engineering](#feature-engineering)
  - [Explainability](#explainability)
  - [Ensemble Methods](#ensemble-methods)
- [Model Assessment](#model-assessment)
- [Contributing](#contributing)
- [License](#license)

## Installation

You can install SmartPredict using pip:

```bash
pip install smartpredict
```

## Features

- **Advanced Model Selection**: Supports a wide range of models, including tree-based methods, neural networks, and more.
- **Automated Hyperparameter Tuning**: Uses Optuna for efficient hyperparameter optimization.
- **Feature Engineering**: Includes tools for automated feature creation and selection.
- **Ensemble Methods**: Implements stacking, blending, and voting techniques.
- **Model Explainability**: Provides SHAP and LIME for interpretability.
- **Parallel Processing**: Speeds up model training and evaluation.

## Quick Start

Here’s a quick example to get you started:

```python
from smartpredict import SmartClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

data = load_breast_cancer()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=123)

clf = SmartClassifier(verbose=1)
results = clf.fit(X_train, X_test, y_train, y_test)
print(results)
```

## Usage

### Classification

```python
from smartpredict import SmartClassifier

# Your code for loading and splitting data

clf = SmartClassifier(verbose=1)
results = clf.fit(X_train, X_test, y_train, y_test)
print(results)
```

### Regression

```python
from smartpredict import SmartRegressor

# Your code for loading and splitting data

reg = SmartRegressor(verbose=1)
results = reg.fit(X_train, X_test, y_train, y_test)
print(results)
```

## Advanced Features

### Hyperparameter Tuning

SmartPredict uses Optuna for hyperparameter optimization:

```python
clf = SmartClassifier(hyperparameter_tuning=True)
results = clf.fit(X_train, X_test, y_train, y_test)
print(results)
```

### Feature Engineering

Automated feature engineering to improve model performance:

```python
from smartpredict import SmartClassifier

clf = SmartClassifier(feature_engineering=True)
results = clf.fit(X_train, X_test, y_train, y_test)
print(results)
```

### Explainability

Model explainability with SHAP:

```python
clf = SmartClassifier(explainability=True)
results = clf.fit(X_train, X_test, y_train, y_test)
print(results)
```

### Ensemble Methods

Combine multiple models for better performance:

```python
clf = SmartClassifier(ensemble_methods=True)
results = clf.fit(X_train, X_test, y_train, y_test)
print(results)
```

## Model Assessment

SmartPredict provides comprehensive model assessment metrics to evaluate your machine learning models. Here is how you can use it:

```python
from smartpredict import ModelAssessment

# Assuming model, X_test, and y_test are defined
assessment = ModelAssessment(model, X_test, y_test)
results = assessment.summary()

print("Model Assessment Metrics:")
print(f"Accuracy: {results['accuracy']}")
print(f"Precision: {results['precision']}")
print(f"Recall: {results['recall']}")
print(f"F1 Score: {results['f1_score']}")
print(f"Confusion Matrix: {results['confusion_matrix']}")
print(f"ROC AUC: {results['roc_auc']}")
print("Classification Report:")
print(results['classification_report'])
```

## Contributing

We welcome contributions! Please read our [Contributing Guidelines](CONTRIBUTING.md) for more information.

## License

SmartPredict is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
