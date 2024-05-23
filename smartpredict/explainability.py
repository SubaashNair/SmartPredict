# smartpredict/explainability.py
"""
Explainability module for SmartPredict.
Provides functions for model explainability using SHAP.
"""

import shap

def explain_model(model, X_test, y_test):
    """
    Explain the model predictions using SHAP.

    Parameters:
    model (estimator): The trained machine learning model.
    X_test (array-like): Testing features.
    y_test (array-like): Testing labels.

    Returns:
    None
    """
    explainer = shap.Explainer(model)
    shap_values = explainer(X_test)
    shap.summary_plot(shap_values, X_test)