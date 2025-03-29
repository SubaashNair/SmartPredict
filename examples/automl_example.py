#!/usr/bin/env python3
"""
Example demonstrating the AutoML functionality in SmartPredict.
This example shows how to use AutoML for classification and regression tasks.
"""

import pandas as pd
import numpy as np
from sklearn.datasets import load_iris, fetch_california_housing
from smartpredict import AutoML

def classification_example():
    """
    Demonstrates using AutoML on the Iris dataset for classification.
    """
    try:
        print("\n--- Classification Example (Iris Dataset) ---")
        
        # Load Iris dataset
        iris = load_iris()
        data = pd.DataFrame(
            data=iris.data,
            columns=iris.feature_names
        )
        data['target'] = iris.target

        # Print dataset info
        print(f"Dataset shape: {data.shape}")
        print(f"Features: {', '.join(iris.feature_names)}")
        print(f"Number of classes: {len(np.unique(iris.target))}")
        
        # Initialize and run AutoML
        print("\nRunning AutoML...")
        auto_ml = AutoML(verbose=1)
        result = auto_ml.fit(data)
        
        # Display results
        print("\nResults:")
        print(f"Best model: {result.best_model_name}")
        print(f"Accuracy: {result.metrics.get('accuracy', 'N/A'):.4f}")
        print(f"F1 Score: {result.metrics.get('f1', 'N/A'):.4f}")
        
        # Get feature importance
        importance = result.explain()['feature_importance']
        print("\nFeature Importance:")
        print(importance.sort_values('importance', ascending=False))
        
        # Make predictions
        print("\nMaking predictions for the first sample:")
        new_data = data.drop('target', axis=1).iloc[:1]
        prediction = result.predict(new_data)
        proba = result.predict_proba(new_data)
        print(f"Sample features: {new_data.values[0]}")
        print(f"Predicted class: {prediction[0]}")
        print(f"Class probabilities: {proba[0]}")
        
        # Save and load model
        model_path = "iris_automl_model.pkl"
        print(f"\nSaving model to {model_path}")
        result.save(model_path)
        
        print("\nClassification example completed successfully!")
        return True
    except Exception as e:
        print(f"Error in classification example: {str(e)}")
        return False

def regression_example():
    """
    Demonstrates using AutoML on the California Housing dataset for regression.
    """
    try:
        print("\n--- Regression Example (California Housing Dataset) ---")
        
        # Load California Housing dataset
        housing = fetch_california_housing()
        data = pd.DataFrame(
            data=housing.data,
            columns=housing.feature_names
        )
        data['target'] = housing.target
        
        # Print dataset info
        print(f"Dataset shape: {data.shape}")
        print(f"Features: {', '.join(housing.feature_names)}")
        print(f"Target range: {data['target'].min():.2f} to {data['target'].max():.2f}")
        
        # Initialize and run AutoML (with time budget to make it faster)
        print("\nRunning AutoML...")
        auto_ml = AutoML(verbose=1, time_budget=60)  # 60 second time budget
        result = auto_ml.fit(data)
        
        # Display results
        print("\nResults:")
        print(f"Best model: {result.best_model_name}")
        print(f"R² score: {result.metrics.get('r2', 'N/A'):.4f}")
        print(f"RMSE: {result.metrics.get('rmse', 'N/A'):.4f}")
        print(f"MAE: {result.metrics.get('mae', 'N/A'):.4f}")
        
        # Get feature importance
        importance = result.explain()['feature_importance']
        print("\nFeature Importance:")
        print(importance.sort_values('importance', ascending=False))
        
        # Make predictions
        print("\nMaking predictions for the first sample:")
        new_data = data.drop('target', axis=1).iloc[:1]
        prediction = result.predict(new_data)
        print(f"Sample features: {new_data.values[0]}")
        print(f"Actual value: {data['target'].iloc[0]:.4f}")
        print(f"Predicted value: {prediction[0]:.4f}")
        
        # Save model
        model_path = "housing_automl_model.pkl"
        print(f"\nSaving model to {model_path}")
        result.save(model_path)
        
        print("\nRegression example completed successfully!")
        return True
    except Exception as e:
        print(f"Error in regression example: {str(e)}")
        return False

def numpy_example():
    """
    Demonstrates using AutoML with NumPy arrays instead of pandas DataFrames.
    """
    try:
        print("\n--- NumPy Array Example (Synthetic Data) ---")
        
        # Create synthetic data (last column is the target)
        np.random.seed(42)  # For reproducibility
        n_samples = 100
        
        # Features
        X = np.random.rand(n_samples, 5)
        
        # Target (linear relationship with some noise)
        y = 3*X[:, 0] - 2*X[:, 1] + 0.5*X[:, 2] + np.random.normal(0, 0.1, n_samples)
        
        # Combine into a single array (last column is target)
        data = np.column_stack((X, y))
        
        print(f"Data shape: {data.shape}")
        print(f"First few rows:\n{data[:3]}")
        
        # Run AutoML
        print("\nRunning AutoML on NumPy array...")
        auto_ml = AutoML(verbose=1)
        result = auto_ml.fit(data)
        
        # Display results
        print("\nResults:")
        print(f"Best model: {result.best_model_name}")
        print(f"R² score: {result.metrics.get('r2', 'N/A'):.4f}")
        
        # Make predictions
        print("\nMaking predictions for the first sample:")
        new_sample = X[0:1]  # First sample features
        prediction = result.predict(new_sample)
        actual = y[0]
        print(f"Sample features: {new_sample[0]}")
        print(f"Actual value: {actual:.4f}")
        print(f"Predicted value: {prediction[0]:.4f}")
        
        print("\nNumPy example completed successfully!")
        return True
    except Exception as e:
        print(f"Error in NumPy example: {str(e)}")
        return False

if __name__ == "__main__":
    print("=== SmartPredict AutoML Examples ===")
    
    print("\nRunning examples...")
    
    # Run all examples
    classification_example()
    regression_example()
    numpy_example()
    
    print("\n=== All examples completed ===")