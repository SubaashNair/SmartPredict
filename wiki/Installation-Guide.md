# Installation Guide

This guide covers how to install SmartPredict and its dependencies.

## Prerequisites

SmartPredict requires:

- Python 3.8 or higher
- pip (Python package installer)

## Installing from PyPI

The simplest way to install SmartPredict is directly from PyPI:

```bash
pip install smartpredict
```

This will install SmartPredict and all its required dependencies.

## Installing from Source

To install the latest development version from GitHub:

1. Clone the repository:
```bash
git clone https://github.com/SubaashNair/smartpredict.git
```

2. Navigate to the project directory:
```bash
cd smartpredict
```

3. Install the package in development mode:
```bash
pip install -e .
```

## Dependencies

SmartPredict depends on several packages that will be automatically installed:

- **NumPy** & **Pandas**: For data manipulation
- **scikit-learn**: For machine learning algorithms
- **SHAP**: For model explainability
- **Optuna**: For hyperparameter optimization
- **Matplotlib** & **Seaborn**: For visualization

## Verifying Installation

You can verify that SmartPredict is correctly installed by importing it in a Python interpreter:

```python
import smartpredict

# Check version
print(smartpredict.__version__)
```

## Troubleshooting

If you encounter any issues during installation:

1. Make sure you're using a supported Python version (3.8+)
2. Try updating pip: `pip install --upgrade pip`
3. If you're behind a firewall, ensure you have the necessary permissions to download packages
4. For installation from source, check that you have the required build tools installed

If problems persist, please [open an issue](https://github.com/SubaashNair/smartpredict/issues) on our GitHub repository.