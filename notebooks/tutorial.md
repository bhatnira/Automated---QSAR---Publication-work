# Example Jupyter Notebook for AutoML QSAR

This notebook demonstrates the usage of AutoML QSAR.

## Installation

```python
# Install required packages
!pip install -r requirements.txt
```

## Import Libraries

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from automl_qsar import AutoMLQSAR

# Set style
sns.set_style('whitegrid')
```

## Load Data

```python
# Load your molecular data
data = pd.read_csv('your_data.csv')

# Display first few rows
data.head()
```

## Initialize AutoML QSAR

```python
automl = AutoMLQSAR(
    featurizers=['rdkit', 'ecfp', 'maccs'],
    models=['ridge', 'randomforest', 'xgboost', 'nn'],
    optimization_method='bayesian',
    cv_strategy='kfold',
    ensemble_method='topk',
    n_trials=50,
    random_state=42
)

# Load data
automl.load_data(data, smiles_col='SMILES', target_col='Activity')
```

## Run Pipeline

```python
# Run the automated pipeline
results = automl.fit()
```

## Evaluate Results

```python
# Display model performance
results_df = pd.DataFrame(results['model_performance'])
results_df.sort_values('rmse_mean')
```

## Make Predictions

```python
# Predict on new molecules
test_smiles = ['CCO', 'c1ccccc1', 'CC(C)O']
predictions = automl.predict(test_smiles)

print("Predictions:")
for smile, pred in zip(test_smiles, predictions):
    print(f"{smile}: {pred:.4f}")
```

## Uncertainty Estimation

```python
# Get predictions with uncertainty
predictions, uncertainties = automl.predict_with_uncertainty(test_smiles)

# Plot
plt.figure(figsize=(10, 6))
plt.errorbar(range(len(predictions)), predictions, yerr=uncertainties, 
             fmt='o', capsize=5, capthick=2)
plt.xlabel('Molecule Index')
plt.ylabel('Predicted Activity')
plt.title('Predictions with Uncertainty')
plt.xticks(range(len(test_smiles)), test_smiles, rotation=45)
plt.tight_layout()
plt.show()
```

## Model Interpretation

```python
# Get feature importance
interpretations = automl.interpret()

if 'feature_importance' in interpretations:
    importance = interpretations['feature_importance']
    
    # Plot top 20 features
    top_features = list(importance.items())[:20]
    features, scores = zip(*top_features)
    
    plt.figure(figsize=(10, 8))
    plt.barh(range(len(features)), scores)
    plt.yticks(range(len(features)), features)
    plt.xlabel('Importance')
    plt.title('Top 20 Most Important Features')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()
```

## Save Model

```python
# Save the trained pipeline
automl.save('./trained_qsar_model')
print("Model saved!")
```

## Load and Use Saved Model

```python
# Load a saved model
automl_loaded = AutoMLQSAR()
automl_loaded.load('./trained_qsar_model')

# Make predictions
new_predictions = automl_loaded.predict(test_smiles)
print(new_predictions)
```
