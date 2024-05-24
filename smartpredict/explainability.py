# smartpredict/explainability.py
"""
Explainability module for SmartPredict.
Provides functions for model explainability using SHAP.
"""

import shap

def explain_model(model, X_test, y_test,sample_size=100):
    """
    Explain the model predictions using SHAP.

    Parameters:
    model (estimator): The trained machine learning model. Must have a `predict` method.
    X_test (array-like or DataFrame): Testing features. This is the dataset used to explain the model's predictions.
    y_test (array-like or Series): Testing labels. This parameter is currently not used in the function but can be utilized for more advanced explanations or validations.
    sample_size (int): The number of samples to use for summarizing the background dataset. Reducing the size of the background data can speed up the SHAP calculations.

    Returns:
    None
    """
    # Summarize the background data
    background = shap.sample(X_test, sample_size)
    
    # Create a KernelExplainer with the summarized background data
    explainer = shap.KernelExplainer(model.predict,background)
    
    # Calculate SHAP values for the test dataset
    shap_values = explainer.shap_values(X_test)
    
    # Generate a summary plot of the SHAP values
    shap.summary_plot(shap_values, X_test)