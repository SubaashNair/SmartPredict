#!/usr/bin/env python3
"""
Example demonstrating SmartPredict with classification datasets from scikit-learn.
This example shows how to properly use SmartClassifier with the correct model names.
"""

import pandas as pd
import numpy as np
from sklearn.datasets import load_iris, load_breast_cancer, load_wine, load_digits
from sklearn.model_selection import train_test_split
from smartpredict import SmartClassifier

# List of available classification models in SmartPredict
AVAILABLE_MODELS = [
    'Logistic Regression', 
    'Random Forest',
    'Gradient Boosting',
    'AdaBoost',
    'Decision Tree',
    'Support Vector Machine',
    'K-Nearest Neighbors',
    'Gaussian Naive Bayes',
    'Neural Network',
    'XGBoost',
    'LightGBM',
    'CatBoost'
]

def iris_example():
    """
    Demonstrates SmartPredict with the Iris dataset.
    """
    try:
        print("\n--- Iris Classification Example ---")
        
        # Load Iris dataset
        iris = load_iris()
        X = iris.data
        y = iris.target
        feature_names = iris.feature_names
        target_names = iris.target_names
        
        # Print dataset info
        print(f"Dataset shape: {X.shape}")
        print(f"Features: {', '.join(feature_names)}")
        print(f"Classes: {', '.join(target_names)}")
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        print(f"Training data: {X_train.shape[0]} samples")
        print(f"Test data: {X_test.shape[0]} samples")
        
        # Create a classifier with the CORRECT model names
        print("\nTraining models...")
        clf = SmartClassifier(
            models=['Random Forest', 'Decision Tree', 'Logistic Regression'],
            # Optional model parameters
            Random_Forest={'n_estimators': 100, 'max_depth': 5},
            Logistic_Regression={'C': 1.0, 'max_iter': 300},
            verbose=1
        )
        
        # Fit and evaluate the models
        results = clf.fit(X_train, X_test, y_train, y_test)
        
        # Print the results
        print("\nModel Results:")
        print(results)
        
        # Get the best model name
        best_model = clf.get_best_model_name()
        print(f"\nBest model: {best_model}")
        
        # Make predictions
        print("\nMaking predictions on test data...")
        predictions = clf.predict(X_test[:5])
        
        # Display sample predictions
        print("\nSample Predictions:")
        for i, pred in enumerate(predictions[:5]):
            true_class = target_names[y_test[i]]
            pred_class = target_names[pred]
            print(f"Sample {i+1}: True = {true_class}, Predicted = {pred_class}")
        
        print("\nIris example completed successfully!")
        return True
    except ValueError as e:
        print(f"Error: {e}")
        print("Please check the model names. Available classification models:")
        for model in AVAILABLE_MODELS:
            print(f"- '{model}'")
        return False

def breast_cancer_example():
    """
    Demonstrates SmartPredict with the Breast Cancer dataset.
    """
    try:
        print("\n--- Breast Cancer Classification Example ---")
        
        # Load Breast Cancer dataset
        cancer = load_breast_cancer()
        X = cancer.data
        y = cancer.target
        
        # Print dataset info
        print(f"Dataset shape: {X.shape}")
        print(f"Number of features: {X.shape[1]}")
        print(f"Classes: {cancer.target_names[0]} (0), {cancer.target_names[1]} (1)")
        print(f"Class distribution: {np.bincount(y)}")
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y
        )
        print(f"Training data: {X_train.shape[0]} samples")
        print(f"Test data: {X_test.shape[0]} samples")
        
        # Create classifier with multiple models
        print("\nTraining models with proper error handling...")
        try:
            # First attempt with potentially wrong model names
            clf = SmartClassifier(
                models=['Random Forest', 'Logistic Regression', 'XGBoost'],
                verbose=1
            )
        except ValueError as e:
            print(f"Error with model names: {e}")
            # Fallback to known working models
            print("Falling back to valid model names...")
            clf = SmartClassifier(
                models=['Random Forest', 'Logistic Regression'],
                verbose=1
            )
        
        # Fit and evaluate models
        results = clf.fit(X_train, X_test, y_train, y_test)
        
        # Print results
        print("\nModel Results:")
        print(results)
        
        # Make predictions
        print("\nMaking predictions...")
        predictions = clf.predict(X_test[:5])
        
        # Display sample predictions
        print("\nSample Predictions:")
        for i, pred in enumerate(predictions[:5]):
            true_class = cancer.target_names[y_test[i]]
            pred_class = cancer.target_names[pred]
            print(f"Sample {i+1}: True = {true_class}, Predicted = {pred_class}")
        
        print("\nBreast Cancer example completed successfully!")
        return True
    except Exception as e:
        print(f"Error in breast cancer example: {str(e)}")
        print("Available classification models:")
        for model in AVAILABLE_MODELS:
            print(f"- '{model}'")
        return False

def multi_dataset_comparison():
    """
    Compares SmartPredict performance across multiple datasets.
    """
    try:
        print("\n--- Multi-Dataset Comparison ---")
        
        # Define datasets to use
        datasets = [
            ('Iris', load_iris()),
            ('Wine', load_wine()),
            ('Digits', load_digits()),
            ('Breast Cancer', load_breast_cancer())
        ]
        
        # Models to test
        models = ['Random Forest', 'Logistic Regression']
        
        results = {}
        
        # Run comparison on each dataset
        for name, dataset in datasets:
            print(f"\nProcessing {name} dataset...")
            
            # Extract data
            X = dataset.data
            y = dataset.target
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42, stratify=y
            )
            
            # Create and train classifier
            clf = SmartClassifier(models=models, verbose=1)
            dataset_results = clf.fit(X_train, X_test, y_train, y_test)
            
            # Store results
            results[name] = {
                'best_model': clf.get_best_model_name(),
                'accuracy': dataset_results[clf.get_best_model_name()].get('accuracy', 0),
                'f1_score': dataset_results[clf.get_best_model_name()].get('f1', 0),
                'num_samples': X.shape[0],
                'num_features': X.shape[1],
                'num_classes': len(np.unique(y))
            }
        
        # Display comparison results
        print("\n=== Dataset Comparison Results ===")
        print(f"{'Dataset':<15} {'Best Model':<20} {'Accuracy':<10} {'F1 Score':<10} {'Samples':<10} {'Features':<10} {'Classes':<10}")
        print("-" * 85)
        
        for name, result in results.items():
            print(f"{name:<15} {result['best_model']:<20} {result['accuracy']:<10.4f} {result['f1_score']:<10.4f} {result['num_samples']:<10} {result['num_features']:<10} {result['num_classes']:<10}")
        
        print("\nMulti-dataset comparison completed successfully!")
        return True
    except Exception as e:
        print(f"Error in multi-dataset comparison: {str(e)}")
        return False

if __name__ == "__main__":
    print("=== SmartPredict Classification Examples ===")
    
    # Print available models
    print("Available classification models in SmartPredict:")
    for model in AVAILABLE_MODELS:
        print(f"- '{model}'")
    
    # Run examples
    iris_example()
    breast_cancer_example()
    multi_dataset_comparison()
    
    print("\n=== All examples completed ===")