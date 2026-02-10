# AutoML QSAR Modeling Software

A comprehensive **agentic** automated machine learning pipeline for Quantitative Structure-Activity Relationship (QSAR) regression modeling.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## ✨ Features

### Agentic AutoML QSAR Capabilities

- **Data Validation & Quality Control**
  - SMILES validation and canonicalization
  - Outlier detection (IQR, Z-score, Isolation Forest)
  - Noise detection (residual analysis, k-NN consistency)
  - Duplicate molecule identification

- **Diverse Feature Space Exploration**
  - RDKit molecular descriptors
  - ECFP/Morgan fingerprints
  - MACCS keys
  - Multiple preprocessing strategies (StandardScaler, MinMax, Robust)

- **Multi-Model Family Exploration**
  - Linear models (Ridge, Lasso, ElasticNet)
  - Tree-based models (Random Forest, Gradient Boosting, XGBoost)
  - Kernel methods (SVR, Kernel Ridge)
  - Neural networks (Feedforward, Deep NN)

- **Iterative Hyperparameter Optimization**
  - Bayesian optimization (Optuna)
  - Genetic algorithms
  - Grid search

- **Robust Evaluation & Benchmarking**
  - Nested cross-validation
  - External validation
  - Q² (predictive R²), RMSE, MAE, R²

- **Automated Report Generation**
  - Markdown reports with model benchmarks
  - JSON data export for programmatic access
  - Feature importance analysis

## Architecture

The software follows a modular pipeline architecture:

1. **Featurization Module** - Molecular descriptor generation (RDKit, ECFP, Pharmacophore, Embeddings, Graph)
2. **Preprocessing Module** - Data scaling, dimensionality reduction, feature selection
3. **Model Selection Module** - Linear, Tree-based, Kernel, Neural Networks, GNN models
4. **Hyperparameter Optimization** - Bayesian, Genetic, Grid Search
5. **Evaluation Module** - Nested CV, LOSO, Q², RMSE, MAE metrics
6. **Ensemble Module** - Top-K pipeline consensus and uncertainty estimation
7. **Prediction & Interpretability** - Model predictions and feature importance

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### Regression-Focused Pipeline (Recommended)

```python
from automl_qsar import AutoMLQSARRegressor
import pandas as pd

# Load your data
df = pd.read_csv('molecules.csv')

# Initialize the AutoML QSAR regressor
regressor = AutoMLQSARRegressor(
    featurizers=['rdkit', 'ecfp', 'maccs'],
    models=['ridge', 'randomforest', 'xgboost', 'svr'],
    n_trials=50,           # HPO trials
    cv_folds=5,            # Cross-validation folds
    handle_outliers=True,  # Auto-detect and handle outliers
    detect_noise=True      # Detect noisy data points
)

# Load data
regressor.load_data(data=df, smiles_col='SMILES', target_col='pIC50')

# Run the automated pipeline
results = regressor.fit()

# Results include:
# - Data validation summary
# - Feature space exploration
# - Model benchmarking
# - HPO results
# - Final metrics (R², RMSE, MAE, Q²)

# Make predictions with uncertainty
predictions, uncertainty = regressor.predict_with_uncertainty(new_smiles)

# Save the model
regressor.save('my_qsar_model')

# Load and reuse
regressor.load('my_qsar_model')
```

### Classic Pipeline

```python
from automl_qsar import AutoMLQSAR

# Initialize the AutoML QSAR pipeline
automl = AutoMLQSAR()

# Load data (SMILES and activity values)
automl.load_data('molecules.csv', smiles_col='SMILES', target_col='Activity')

# Run the automated pipeline
results = automl.fit()

# Make predictions
predictions = automl.predict(new_smiles)

# Get model interpretation
interpretation = automl.interpret()
```

## Pipeline Components

### DataValidator
```python
from automl_qsar.pipeline.automl_regressor import DataValidator

validator = DataValidator()
valid_mask, info = validator.validate_smiles(smiles_list)
outlier_mask, info = validator.detect_outliers(values, method='iqr')
noise_mask, info = validator.detect_noise(X, y, method='residual')
```

### FeatureSpaceExplorer
```python
from automl_qsar.pipeline.automl_regressor import FeatureSpaceExplorer

explorer = FeatureSpaceExplorer()
results = explorer.explore_feature_space(smiles, y, featurizers=['rdkit', 'ecfp'])
best_config = results['best_configuration']
```

### ModelExplorer
```python
from automl_qsar.pipeline.automl_regressor import ModelExplorer

explorer = ModelExplorer()
results = explorer.explore_models(X, y, models=['ridge', 'randomforest', 'xgboost'])
best_model = results['best_model']
```

### HyperparameterOptimizer
```python
from automl_qsar.pipeline.automl_regressor import HyperparameterOptimizer

optimizer = HyperparameterOptimizer(method='bayesian', n_trials=50)
results = optimizer.optimize('randomforest', X, y, param_space)
```

## Project Structure

```
automl_qsar/
├── featurization/      # Molecular featurization methods
├── preprocessing/      # Data preprocessing utilities
├── models/            # ML model implementations
├── optimization/      # Hyperparameter optimization
├── evaluation/        # Model evaluation metrics
├── ensemble/          # Ensemble methods
├── interpretation/    # Model interpretability
└── pipeline/          # Main pipeline orchestration
    ├── automl_pipeline.py    # Classic AutoML pipeline
    └── automl_regressor.py   # Regression-focused agentic pipeline
```

## Example Results

Using the kinase IC50 dataset (40 compounds):

| Metric | Value |
|--------|-------|
| R² | 0.74 |
| RMSE | 0.55 |
| MAE | 0.43 |
| Q² | 0.74 |

## Requirements

- Python >= 3.8
- RDKit
- scikit-learn
- XGBoost
- Optuna (Bayesian optimization)
- pandas, numpy

Optional:
- PyTorch / PyTorch Geometric (for GNN)
- DEAP (Genetic algorithms)
- matplotlib, seaborn (visualization)

## License

MIT License
