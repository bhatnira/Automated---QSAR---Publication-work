# AutoML QSAR - Comprehensive Documentation

## Table of Contents
1. [Overview](#overview)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [Architecture](#architecture)
5. [Modules](#modules)
6. [Advanced Usage](#advanced-usage)
7. [API Reference](#api-reference)

## Overview

AutoML QSAR is a comprehensive automated machine learning framework for Quantitative Structure-Activity Relationship (QSAR) modeling. It automates the entire workflow from molecular featurization to model deployment.

### Key Features

- **Multiple Featurization Methods**: RDKit descriptors, ECFP, MACCS keys, pharmacophore features, embeddings, and graph representations
- **Diverse Model Types**: Linear models, tree-based models, kernel methods, neural networks, and graph neural networks
- **Automated Hyperparameter Optimization**: Bayesian, genetic, and grid search algorithms
- **Robust Evaluation**: Nested cross-validation, LOSO CV, multiple metrics (RMSE, MAE, R², Q²)
- **Ensemble Methods**: Voting, stacking, and top-K consensus
- **Model Interpretability**: Feature importance, SHAP, and LIME

## Installation

### Prerequisites

- Python >= 3.8
- RDKit (for molecular featurization)
- PyTorch (optional, for neural networks)
- PyTorch Geometric (optional, for GNNs)

### Basic Installation

```bash
git clone https://github.com/bhatnira/Automated---QSAR---Publication-work.git
cd Automated---QSAR---Publication-work
pip install -r requirements.txt
pip install -e .
```

### Installing RDKit

```bash
# Using conda (recommended)
conda install -c conda-forge rdkit

# Using pip
pip install rdkit
```

### Optional Dependencies

```bash
# For deep learning
pip install torch torch-geometric

# For additional features
pip install xgboost umap-learn boruta shap lime
```

## Quick Start

### Basic Example

```python
from automl_qsar import AutoMLQSAR
import pandas as pd

# Load data
data = pd.read_csv('molecules.csv')

# Initialize AutoML QSAR
automl = AutoMLQSAR(
    featurizers=['rdkit', 'ecfp'],
    models=['ridge', 'randomforest', 'xgboost'],
    optimization_method='bayesian',
    cv_strategy='kfold',
    ensemble_method='topk',
    n_trials=50
)

# Load and train
automl.load_data(data, smiles_col='SMILES', target_col='Activity')
results = automl.fit()

# Predict
test_smiles = ['CCO', 'c1ccccc1']
predictions = automl.predict(test_smiles)

# Save model
automl.save('./trained_model')
```

## Architecture

The AutoML QSAR pipeline consists of seven main modules:

### 1. Featurization Module

Converts molecular structures (SMILES/SDF) into numerical features.

**Available Featurizers:**
- `RDKitDescriptors`: Physicochemical properties (MW, LogP, TPSA, etc.)
- `ECFPFingerprints`: Extended Connectivity Fingerprints (Morgan)
- `MACCSFingerprints`: 166-bit structural keys
- `PharmacophoreFeatures`: Pharmacophoric features
- `MolecularEmbeddings`: Pre-trained molecular embeddings
- `GraphFeaturizer`: Graph representations for GNNs

**Example:**
```python
from automl_qsar.featurization import RDKitDescriptors, ECFPFingerprints

# RDKit descriptors
rdkit_feat = RDKitDescriptors()
X_rdkit = rdkit_feat.featurize(['CCO', 'c1ccccc1'])

# ECFP4 fingerprints
ecfp_feat = ECFPFingerprints(radius=2, n_bits=2048)
X_ecfp = ecfp_feat.featurize(['CCO', 'c1ccccc1'])
```

### 2. Preprocessing Module

Handles data scaling, feature selection, and dimensionality reduction.

**Available Methods:**
- **Scalers**: StandardScaler, MinMaxScaler, RobustScaler
- **Feature Selection**: VarianceThreshold, CorrelationSelector, BorutaSelector
- **Dimensionality Reduction**: PCA, UMAP

**Example:**
```python
from automl_qsar.preprocessing import StandardScaler, BorutaSelector

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Select features
selector = BorutaSelector(n_estimators=100)
X_selected = selector.fit_transform(X_scaled, y)
```

### 3. Model Selection Module

Provides various machine learning models.

**Available Models:**

**Linear Models:**
- LinearRegression
- Ridge
- Lasso
- ElasticNet

**Tree-Based Models:**
- RandomForest
- GradientBoosting
- XGBoost

**Kernel Methods:**
- SVR (Support Vector Regression)
- KernelRidge

**Neural Networks:**
- FeedForwardNN
- DeepNN
- GraphNeuralNetwork (GNN)

**Example:**
```python
from automl_qsar.models import RandomForest, DeepNN

# Random Forest
rf = RandomForest(n_estimators=100, max_depth=10)
rf.fit(X_train, y_train)
predictions = rf.predict(X_test)

# Deep Neural Network
dnn = DeepNN(hidden_layers=[256, 128, 64], epochs=100)
dnn.fit(X_train, y_train)
predictions = dnn.predict(X_test)
```

### 4. Hyperparameter Optimization Module

Automated hyperparameter tuning.

**Available Optimizers:**
- `BayesianOptimizer`: Optuna-based Bayesian optimization
- `GeneticOptimizer`: Genetic algorithm
- `GridSearchOptimizer`: Exhaustive grid search

**Example:**
```python
from automl_qsar.optimization import BayesianOptimizer

optimizer = BayesianOptimizer(n_trials=100)

def objective(params):
    model = RandomForest(**params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    return rmse(y_val, y_pred)

param_space = {
    'n_estimators': {'type': 'int', 'low': 50, 'high': 300},
    'max_depth': {'type': 'int', 'low': 3, 'high': 15},
}

best_params = optimizer.optimize(objective, param_space)
```

### 5. Evaluation Module

Model evaluation with various metrics and cross-validation strategies.

**Metrics:**
- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)
- R² (Coefficient of Determination)
- Q² (Cross-validated R²)
- CCC (Concordance Correlation Coefficient)

**Cross-Validation:**
- K-Fold CV
- Leave-One-Series-Out (LOSO) CV
- Nested CV

**Example:**
```python
from automl_qsar.evaluation import KFoldCV, Metrics

# K-Fold Cross-Validation
cv = KFoldCV(n_splits=5)

scores = []
for train_idx, test_idx in cv.split(X, y):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    rmse = Metrics.rmse(y_test, y_pred)
    scores.append(rmse)

print(f"Mean RMSE: {np.mean(scores):.4f} ± {np.std(scores):.4f}")
```

### 6. Ensemble Module

Combines multiple models for improved performance.

**Available Ensembles:**
- `VotingEnsemble`: Weighted average of predictions
- `StackingEnsemble`: Meta-model trained on base predictions
- `TopKEnsemble`: Selects best K models

**Example:**
```python
from automl_qsar.ensemble import TopKEnsemble

# Train multiple models
models = [model1, model2, model3, model4, model5]

# Create ensemble
ensemble = TopKEnsemble(k=3, method='weighted_mean')
ensemble.fit(models, X_val, y_val)

# Predict with uncertainty
predictions, uncertainties = ensemble.predict_with_uncertainty(X_test)
```

### 7. Interpretation Module

Model interpretability and explainability.

**Available Methods:**
- Feature Importance (tree-based, linear, permutation)
- SHAP (SHapley Additive exPlanations)
- LIME (Local Interpretable Model-agnostic Explanations)

**Example:**
```python
from automl_qsar.interpretation import FeatureImportance, SHAPInterpreter

# Feature importance
importance = FeatureImportance.get_tree_importance(model, feature_names)
FeatureImportance.plot_importance(importance, top_n=20)

# SHAP
shap_interp = SHAPInterpreter()
shap_results = shap_interp.explain_model(model, X, feature_names)
shap_interp.plot_summary(shap_results['shap_values'], X, feature_names)
```

## Advanced Usage

### Custom Pipeline Configuration

```python
automl = AutoMLQSAR(
    featurizers=['rdkit', 'ecfp4', 'maccs', 'pharmacophore'],
    models=['ridge', 'lasso', 'randomforest', 'xgboost', 'deepnn'],
    optimization_method='bayesian',
    cv_strategy='nested',
    ensemble_method='stacking',
    n_trials=100,
    random_state=42
)
```

### Using Pre-computed Features

```python
# If you already have computed features
X = compute_features(smiles)
y = get_activities()

automl = AutoMLQSAR()
results = automl.fit(X=X, y=y)
```

### Model Persistence

```python
# Save trained pipeline
automl.save('./my_qsar_model')

# Load later
automl_loaded = AutoMLQSAR()
automl_loaded.load('./my_qsar_model')
predictions = automl_loaded.predict(new_smiles)
```

### Batch Prediction

```python
# Predict on large dataset
large_smiles_list = [...]  # 10,000 molecules
predictions = automl.predict(large_smiles_list)
```

## API Reference

### AutoMLQSAR Class

```python
class AutoMLQSAR:
    def __init__(
        self,
        featurizers: List[str] = None,
        models: List[str] = None,
        optimization_method: str = 'bayesian',
        cv_strategy: str = 'kfold',
        ensemble_method: str = 'topk',
        n_trials: int = 50,
        random_state: int = 42
    )
    
    def load_data(
        self,
        data: pd.DataFrame = None,
        filepath: str = None,
        smiles_col: str = 'SMILES',
        target_col: str = 'Activity'
    ) -> 'AutoMLQSAR'
    
    def fit(
        self,
        X: np.ndarray = None,
        y: np.ndarray = None
    ) -> Dict[str, Any]
    
    def predict(
        self,
        smiles: List[str] = None,
        X: np.ndarray = None
    ) -> np.ndarray
    
    def predict_with_uncertainty(
        self,
        smiles: List[str] = None,
        X: np.ndarray = None
    ) -> Tuple[np.ndarray, np.ndarray]
    
    def interpret(self) -> Dict[str, Any]
    
    def save(self, filepath: str) -> None
    
    def load(self, filepath: str) -> None
```

## Best Practices

### 1. Data Quality
- Remove duplicate molecules
- Check for invalid SMILES
- Handle missing values
- Normalize activity values if needed

### 2. Feature Selection
- Use variance threshold for high-dimensional data
- Consider Boruta for small datasets
- PCA for dimensionality reduction when needed

### 3. Model Selection
- Start with simpler models (Ridge, RandomForest)
- Add complex models (XGBoost, DeepNN) if needed
- Use ensemble for final predictions

### 4. Hyperparameter Optimization
- Use Bayesian optimization for efficiency
- Nested CV for unbiased evaluation
- Monitor overfitting

### 5. Interpretation
- Always check feature importance
- Use SHAP for detailed analysis
- Validate predictions with domain knowledge

## Troubleshooting

### Common Issues

**Issue: RDKit not installed**
```bash
conda install -c conda-forge rdkit
```

**Issue: Out of memory with large datasets**
- Use batch processing
- Reduce fingerprint size
- Use PCA for dimensionality reduction

**Issue: Slow training**
- Reduce n_trials for HPO
- Use fewer models
- Enable parallel processing (n_jobs=-1)

## Contributing

Contributions are welcome! Please see CONTRIBUTING.md for guidelines.

## Citation

If you use this software in your research, please cite:

```bibtex
@software{automl_qsar,
  title={AutoML QSAR: Automated Machine Learning for QSAR Modeling},
  author={Your Name},
  year={2026},
  url={https://github.com/bhatnira/Automated---QSAR---Publication-work}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For questions and support:
- GitHub Issues: https://github.com/bhatnira/Automated---QSAR---Publication-work/issues
- Documentation: See this file and code comments
- Examples: Check the `examples/` directory
