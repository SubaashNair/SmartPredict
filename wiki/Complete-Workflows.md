# Complete Workflows

This page demonstrates complete machine learning workflows using SmartPredict, showcasing how different components work together.

## Classification Workflow

This workflow demonstrates a complete pipeline for a classification task using the Breast Cancer dataset:

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

# Import SmartPredict components
from smartpredict import SmartClassifier
from smartpredict.feature_engineering import FeatureEngineer
from smartpredict.explainability import ModelExplainer
from smartpredict.hyperparameter_tuning import tune_hyperparameters
from smartpredict.ensemble_methods import EnsembleModel

# 1. Load and prepare data
print("Loading data...")
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

# Display dataset information
print(f"Dataset shape: {X.shape}")
print(f"Number of classes: {len(np.unique(y))}")
print(f"Class distribution: {np.bincount(y)}")

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")

# 2. Feature Engineering
print("\nApplying feature engineering...")
engineer = FeatureEngineer(
    numeric_features=data.feature_names,
    scaler='standard',
    handle_missing='mean',
    create_interactions=True,
    n_features_to_select=15  # Select top 15 features
)

# Apply feature engineering
X_train_engineered = engineer.fit_transform(X_train, y_train)
X_test_engineered = engineer.transform(X_test)

# Get the names of the selected features
selected_features = engineer.get_feature_names()
print(f"Selected {len(selected_features)} features")

# 3. Model Training and Evaluation
print("\nTraining and evaluating models...")
classifier = SmartClassifier(
    models=['RandomForestClassifier', 'LogisticRegression', 'GradientBoostingClassifier'],
    verbose=1
)

# Train and evaluate
results = classifier.fit(X_train_engineered, X_test_engineered, y_train, y_test)

# Print results
print("\nModel Performance:")
for model_name, metrics in results.items():
    print(f"{model_name}:")
    for metric_name, value in metrics.items():
        print(f"  {metric_name}: {value:.4f}")
    print()

# 4. Hyperparameter Tuning
print("\nPerforming hyperparameter tuning...")
# Get the best model from initial training
best_model_name = max(results, key=lambda x: results[x]['accuracy'])
print(f"Best model: {best_model_name}")

# Define parameter space for tuning
if best_model_name == 'RandomForestClassifier':
    param_space = {
        'n_estimators': (50, 200),
        'max_depth': (3, 15),
        'min_samples_split': (2, 10)
    }
elif best_model_name == 'LogisticRegression':
    param_space = {
        'C': (0.01, 10.0),
        'solver': ['liblinear', 'saga']
    }
else:  # GradientBoostingClassifier
    param_space = {
        'n_estimators': (50, 200),
        'learning_rate': (0.01, 0.3),
        'max_depth': (2, 6)
    }

# Clone the best model for tuning
best_model = classifier.trained_models[best_model_name]

# Perform hyperparameter tuning
optimized_model = tune_hyperparameters(
    model=best_model,
    param_distributions=param_space,
    X=X_train_engineered,
    y=y_train,
    n_trials=50,
    scoring='accuracy',
    random_state=42
)

# Evaluate optimized model
optimized_pred = optimized_model.predict(X_test_engineered)
from sklearn.metrics import accuracy_score
print(f"Optimized model accuracy: {accuracy_score(y_test, optimized_pred):.4f}")

# 5. Model Explainability
print("\nGenerating model explanations...")
explainer = ModelExplainer(
    model=optimized_model,
    feature_names=selected_features,
    task_type='classification'
)
explainer.set_training_data(X_train_engineered)

# Get feature importance
importance = explainer.get_feature_importance()
print("Top 5 features by importance:")
print(importance.sort_values('importance', ascending=False).head(5))

# 6. Create Ensemble Model
print("\nCreating ensemble model...")
# Prepare base models
rf = classifier.trained_models.get('RandomForestClassifier')
lr = classifier.trained_models.get('LogisticRegression')
gb = classifier.trained_models.get('GradientBoostingClassifier')
base_models = []

if rf is not None:
    base_models.append(('rf', rf))
if lr is not None:
    base_models.append(('lr', lr))
if gb is not None:
    base_models.append(('gb', gb))

# Add optimized model
base_models.append(('optimized', optimized_model))

# Create ensemble
ensemble = EnsembleModel(
    models=base_models,
    method='voting',
)

# Train and evaluate ensemble
ensemble.fit(X_train_engineered, y_train)
ensemble_pred = ensemble.predict(X_test_engineered)
ensemble_accuracy = accuracy_score(y_test, ensemble_pred)
print(f"Ensemble model accuracy: {ensemble_accuracy:.4f}")

# 7. Final Evaluation and Reporting
print("\nFinal Report:")
from sklearn.metrics import classification_report, confusion_matrix
print("Classification Report:")
print(classification_report(y_test, ensemble_pred))

print("Confusion Matrix:")
cm = confusion_matrix(y_test, ensemble_pred)
print(cm)

# 8. Save the final model
print("\nSaving models...")
import joblib
joblib.dump(ensemble, 'breast_cancer_ensemble_model.pkl')
joblib.dump(engineer, 'breast_cancer_feature_engineer.pkl')

print("Workflow complete!")
```

## Regression Workflow

This workflow demonstrates a complete pipeline for a regression task using the California Housing dataset:

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

# Import SmartPredict components
from smartpredict import SmartRegressor
from smartpredict.feature_engineering import FeatureEngineer
from smartpredict.explainability import ModelExplainer
from smartpredict.hyperparameter_tuning import tune_hyperparameters
from smartpredict.ensemble_methods import EnsembleModel

# 1. Load and prepare data
print("Loading data...")
data = fetch_california_housing()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

# Display dataset information
print(f"Dataset shape: {X.shape}")
print(f"Target mean: {y.mean():.4f}, std: {y.std():.4f}")
print(f"Target range: {y.min():.4f} to {y.max():.4f}")

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")

# 2. Feature Engineering
print("\nApplying feature engineering...")
engineer = FeatureEngineer(
    numeric_features=data.feature_names,
    scaler='standard',
    handle_missing='mean',
    create_interactions=True
)

# Apply feature engineering
X_train_engineered = engineer.fit_transform(X_train, y_train)
X_test_engineered = engineer.transform(X_test)

# Get feature names
engineered_features = engineer.get_feature_names()
print(f"Number of features after engineering: {len(engineered_features)}")

# 3. Model Training and Evaluation
print("\nTraining and evaluating models...")
regressor = SmartRegressor(
    models=['RandomForestRegressor', 'LinearRegression', 'GradientBoostingRegressor', 'SVR'],
    verbose=1
)

# Train and evaluate
results = regressor.fit(X_train_engineered, X_test_engineered, y_train, y_test)

# Print results
print("\nModel Performance:")
for model_name, metrics in results.items():
    print(f"{model_name}:")
    for metric_name, value in metrics.items():
        print(f"  {metric_name}: {value:.4f}")
    print()

# 4. Hyperparameter Tuning
print("\nPerforming hyperparameter tuning...")
# Get the best model from initial training
best_model_name = max(results, key=lambda x: results[x]['r2'])
print(f"Best model: {best_model_name}")

# Define parameter space for tuning
if best_model_name == 'RandomForestRegressor':
    param_space = {
        'n_estimators': (50, 200),
        'max_depth': (3, 15),
        'min_samples_split': (2, 10)
    }
elif best_model_name == 'LinearRegression':
    # LinearRegression has few parameters to tune
    param_space = {}
elif best_model_name == 'GradientBoostingRegressor':
    param_space = {
        'n_estimators': (50, 200),
        'learning_rate': (0.01, 0.3),
        'max_depth': (2, 6)
    }
else:  # SVR
    param_space = {
        'C': (0.1, 100),
        'gamma': (0.001, 1.0),
        'kernel': ['rbf', 'linear']
    }

# Clone the best model for tuning
best_model = regressor.trained_models[best_model_name]

# Skip tuning if there are no parameters to tune
if param_space:
    # Perform hyperparameter tuning
    optimized_model = tune_hyperparameters(
        model=best_model,
        param_distributions=param_space,
        X=X_train_engineered,
        y=y_train,
        n_trials=50,
        scoring='r2',
        random_state=42
    )

    # Evaluate optimized model
    optimized_pred = optimized_model.predict(X_test_engineered)
    from sklearn.metrics import r2_score, mean_squared_error
    print(f"Optimized model R²: {r2_score(y_test, optimized_pred):.4f}")
    print(f"Optimized model RMSE: {np.sqrt(mean_squared_error(y_test, optimized_pred)):.4f}")
else:
    optimized_model = best_model
    print("Skipping tuning for LinearRegression as it has few parameters.")

# 5. Model Explainability
print("\nGenerating model explanations...")
explainer = ModelExplainer(
    model=optimized_model,
    feature_names=engineered_features,
    task_type='regression'
)
explainer.set_training_data(X_train_engineered)

# Get feature importance
importance = explainer.get_feature_importance()
print("Top 5 features by importance:")
print(importance.sort_values('importance', ascending=False).head(5))

# 6. Create Ensemble Model
print("\nCreating ensemble model...")
# Prepare base models
rf = regressor.trained_models.get('RandomForestRegressor')
lr = regressor.trained_models.get('LinearRegression')
gb = regressor.trained_models.get('GradientBoostingRegressor')
svr = regressor.trained_models.get('SVR')
base_models = []

if rf is not None:
    base_models.append(('rf', rf))
if lr is not None:
    base_models.append(('lr', lr))
if gb is not None:
    base_models.append(('gb', gb))
if svr is not None:
    base_models.append(('svr', svr))

# Add optimized model if it's different
if optimized_model != best_model:
    base_models.append(('optimized', optimized_model))

# Create ensemble
ensemble = EnsembleModel(
    models=base_models,
    method='averaging',
)

# Train and evaluate ensemble
ensemble.fit(X_train_engineered, y_train)
ensemble_pred = ensemble.predict(X_test_engineered)
ensemble_r2 = r2_score(y_test, ensemble_pred)
ensemble_rmse = np.sqrt(mean_squared_error(y_test, ensemble_pred))
print(f"Ensemble model R²: {ensemble_r2:.4f}")
print(f"Ensemble model RMSE: {ensemble_rmse:.4f}")

# 7. Final Evaluation and Reporting
print("\nFinal Report:")
print(f"Best Single Model ({best_model_name}) R²: {results[best_model_name]['r2']:.4f}")
print(f"Optimized Model R²: {r2_score(y_test, optimized_pred):.4f}")
print(f"Ensemble Model R²: {ensemble_r2:.4f}")

# Visualize predictions vs actual
plt.figure(figsize=(10, 6))
plt.scatter(y_test, ensemble_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Ensemble Model: Predicted vs Actual Values')
plt.savefig('housing_predictions.png')
plt.close()

# 8. Save the final model
print("\nSaving models...")
import joblib
joblib.dump(ensemble, 'california_housing_ensemble_model.pkl')
joblib.dump(engineer, 'california_housing_feature_engineer.pkl')

print("Workflow complete!")
```

These workflows demonstrate how to combine all the components of SmartPredict into cohesive machine learning pipelines for both classification and regression tasks.