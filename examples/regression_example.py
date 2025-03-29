#!/usr/bin/env python3
"""
Example demonstrating SmartPredict with regression datasets from scikit-learn.
This example shows how to properly use SmartRegressor with the correct model names.
"""

import pandas as pd
import numpy as np
from sklearn.datasets import load_boston, fetch_california_housing, load_diabetes
from sklearn.model_selection import train_test_split
from smartpredict import SmartRegressor

# List of available regression models in SmartPredict
AVAILABLE_MODELS = [
    'Linear Regression',
    'Ridge Regression',
    'Lasso Regression',
    'Random Forest',
    'Gradient Boosting',
    'AdaBoost',
    'Decision Tree',
    'Support Vector Machine',
    'K-Nearest Neighbors',
    'Neural Network',
    'XGBoost',
    'LightGBM',
    'CatBoost'
]

def boston_housing_example():
    """
    Demonstrates SmartPredict with the Boston Housing dataset.
    """
    try:
        print("\n--- Boston Housing Regression Example ---")
        
        # Handle deprecated dataset
        try:
            # Try to load Boston dataset (might be deprecated in newer sklearn)
            boston = load_boston()
            X = boston.data
            y = boston.target
            feature_names = boston.feature_names
        except:
            print("Boston Housing dataset is deprecated in newer scikit-learn versions.")
            print("Using California Housing dataset instead.")
            return california_housing_example()
        
        # Print dataset info
        print(f"Dataset shape: {X.shape}")
        print(f"Features: {', '.join(feature_names)}")
        print(f"Target range: {y.min():.2f} to {y.max():.2f}")
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        print(f"Training data: {X_train.shape[0]} samples")
        print(f"Test data: {X_test.shape[0]} samples")
        
        # Create a regressor with the CORRECT model names
        print("\nTraining models...")
        reg = SmartRegressor(
            models=['Random Forest', 'Linear Regression', 'Decision Tree'],
            # Optional model parameters
            Random_Forest={'n_estimators': 100, 'max_depth': 10},
            Linear_Regression={},  # Default parameters
            verbose=1
        )
        
        # Fit and evaluate the models
        results = reg.fit(X_train, X_test, y_train, y_test)
        
        # Print the results
        print("\nModel Results:")
        print(results)
        
        # Get the best model name
        best_model = reg.get_best_model_name()
        print(f"\nBest model: {best_model}")
        
        # Make predictions
        print("\nMaking predictions on test data...")
        predictions = reg.predict(X_test[:5])
        
        # Display sample predictions
        print("\nSample Predictions:")
        for i, pred in enumerate(predictions[:5]):
            actual = y_test[i]
            print(f"Sample {i+1}: Actual = {actual:.2f}, Predicted = {pred:.2f}, Diff = {abs(actual-pred):.2f}")
        
        print("\nBoston Housing example completed successfully!")
        return True
    except ValueError as e:
        print(f"Error: {e}")
        print("Please check the model names. Available regression models:")
        for model in AVAILABLE_MODELS:
            print(f"- '{model}'")
        return False

def california_housing_example():
    """
    Demonstrates SmartPredict with the California Housing dataset.
    """
    try:
        print("\n--- California Housing Regression Example ---")
        
        # Load California Housing dataset
        housing = fetch_california_housing()
        X = housing.data
        y = housing.target
        feature_names = housing.feature_names
        
        # Print dataset info
        print(f"Dataset shape: {X.shape}")
        print(f"Features: {', '.join(feature_names)}")
        print(f"Target range: {y.min():.2f} to {y.max():.2f}")
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        print(f"Training data: {X_train.shape[0]} samples")
        print(f"Test data: {X_test.shape[0]} samples")
        
        # Create regressor with multiple models
        print("\nTraining models with proper error handling...")
        try:
            # First attempt with potentially wrong model names
            reg = SmartRegressor(
                models=['Random Forest', 'Linear Regression', 'XGBoost'],
                Random_Forest={'n_estimators': 100, 'max_depth': 8},
                verbose=1
            )
        except ValueError as e:
            print(f"Error with model names: {e}")
            # Fallback to known working models
            print("Falling back to valid model names...")
            reg = SmartRegressor(
                models=['Random Forest', 'Linear Regression'],
                verbose=1
            )
        
        # Fit and evaluate models
        results = reg.fit(X_train, X_test, y_train, y_test)
        
        # Print results
        print("\nModel Results:")
        print(results)
        
        # Make predictions
        print("\nMaking predictions...")
        predictions = reg.predict(X_test[:5])
        
        # Display sample predictions
        print("\nSample Predictions:")
        for i, pred in enumerate(predictions[:5]):
            actual = y_test[i]
            print(f"Sample {i+1}: Actual = {actual:.2f}, Predicted = {pred:.2f}, Diff = {abs(actual-pred):.2f}")
        
        print("\nCalifornia Housing example completed successfully!")
        return True
    except Exception as e:
        print(f"Error in California Housing example: {str(e)}")
        print("Available regression models:")
        for model in AVAILABLE_MODELS:
            print(f"- '{model}'")
        return False

def diabetes_example():
    """
    Demonstrates SmartPredict with the Diabetes dataset.
    """
    try:
        print("\n--- Diabetes Regression Example ---")
        
        # Load Diabetes dataset
        diabetes = load_diabetes()
        X = diabetes.data
        y = diabetes.target
        feature_names = diabetes.feature_names
        
        # Print dataset info
        print(f"Dataset shape: {X.shape}")
        print(f"Features: {', '.join(feature_names)}")
        print(f"Target range: {y.min():.2f} to {y.max():.2f}")
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Create regressor with the correct model names
        print("\nTraining models...")
        reg = SmartRegressor(
            models=['Random Forest', 'Ridge Regression', 'Linear Regression'],
            Ridge_Regression={'alpha': 1.0},
            verbose=1
        )
        
        # Fit and evaluate models
        results = reg.fit(X_train, X_test, y_train, y_test)
        
        # Show results
        print("\nModel Results:")
        print(results)
        
        # Best model
        best_model = reg.get_best_model_name()
        print(f"\nBest model: {best_model}")
        
        # Make predictions
        predictions = reg.predict(X_test[:5])
        
        print("\nSample Predictions:")
        for i, pred in enumerate(predictions[:5]):
            actual = y_test[i]
            print(f"Sample {i+1}: Actual = {actual:.2f}, Predicted = {pred:.2f}, Diff = {abs(actual-pred):.2f}")
        
        print("\nDiabetes example completed successfully!")
        return True
    except Exception as e:
        print(f"Error in diabetes example: {str(e)}")
        return False

def multi_dataset_comparison():
    """
    Compares SmartPredict performance across multiple regression datasets.
    """
    try:
        print("\n--- Multi-Dataset Regression Comparison ---")
        
        # Define datasets to use
        datasets = []
        
        # Try to load Boston dataset (might be deprecated in newer sklearn)
        try:
            boston = load_boston()
            datasets.append(('Boston Housing', boston))
        except:
            print("Boston Housing dataset is deprecated in newer scikit-learn versions.")
        
        # Add California Housing dataset
        california = fetch_california_housing()
        datasets.append(('California Housing', california))
        
        # Add Diabetes dataset
        diabetes = load_diabetes()
        datasets.append(('Diabetes', diabetes))
        
        # Models to test
        models = ['Random Forest', 'Linear Regression', 'Ridge Regression']
        
        results = {}
        
        # Run comparison on each dataset
        for name, dataset in datasets:
            print(f"\nProcessing {name} dataset...")
            
            # Extract data
            X = dataset.data
            y = dataset.target
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42
            )
            
            # Create and train regressor
            reg = SmartRegressor(models=models, verbose=1)
            dataset_results = reg.fit(X_train, X_test, y_train, y_test)
            
            # Get best model and metrics
            best_model = reg.get_best_model_name()
            best_metrics = dataset_results[best_model]
            
            # Store results
            results[name] = {
                'best_model': best_model,
                'r2': best_metrics.get('r2', 0),
                'rmse': best_metrics.get('rmse', 0),
                'mae': best_metrics.get('mae', 0),
                'num_samples': X.shape[0],
                'num_features': X.shape[1]
            }
        
        # Display comparison results
        print("\n=== Dataset Comparison Results ===")
        print(f"{'Dataset':<20} {'Best Model':<20} {'R²':<10} {'RMSE':<10} {'MAE':<10} {'Samples':<10} {'Features':<10}")
        print("-" * 90)
        
        for name, result in results.items():
            print(f"{name:<20} {result['best_model']:<20} {result['r2']:<10.4f} {result['rmse']:<10.4f} {result['mae']:<10.4f} {result['num_samples']:<10} {result['num_features']:<10}")
        
        print("\nMulti-dataset comparison completed successfully!")
        return True
    except Exception as e:
        print(f"Error in multi-dataset comparison: {str(e)}")
        return False

if __name__ == "__main__":
    print("=== SmartPredict Regression Examples ===")
    
    # Print available models
    print("Available regression models in SmartPredict:")
    for model in AVAILABLE_MODELS:
        print(f"- '{model}'")
    
    # Run examples
    california_housing_example()
    diabetes_example()
    multi_dataset_comparison()
    
    print("\n=== All examples completed ===")