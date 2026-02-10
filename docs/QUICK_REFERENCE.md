# AutoML QSAR - Quick Reference

## Installation

```bash
pip install -r requirements.txt
conda install -c conda-forge rdkit
pip install -e .
```

## Basic Usage

```python
from automl_qsar import AutoMLQSAR
import pandas as pd

# Load data
data = pd.read_csv('molecules.csv')

# Create and train model
automl = AutoMLQSAR()
automl.load_data(data, smiles_col='SMILES', target_col='Activity')
results = automl.fit()

# Predict
predictions = automl.predict(['CCO', 'c1ccccc1'])

# Save/Load
automl.save('./model')
automl.load('./model')
```

## Configuration Options

### Featurizers
- `'rdkit'` - RDKit descriptors
- `'ecfp'` / `'ecfp4'` - Morgan fingerprints (radius 2)
- `'ecfp6'` - Morgan fingerprints (radius 3)
- `'maccs'` - MACCS keys
- `'pharmacophore'` - Pharmacophore features
- `'graph'` - Graph representations

### Models
- `'linear'` - Linear regression
- `'ridge'` - Ridge regression
- `'lasso'` - Lasso regression
- `'elasticnet'` - ElasticNet
- `'randomforest'` / `'rf'` - Random Forest
- `'gradientboosting'` / `'gb'` - Gradient Boosting
- `'xgboost'` / `'xgb'` - XGBoost
- `'svr'` - Support Vector Regression
- `'nn'` - Neural Network
- `'deepnn'` - Deep Neural Network
- `'gnn'` - Graph Neural Network

### Optimization Methods
- `'bayesian'` - Bayesian optimization (default)
- `'genetic'` - Genetic algorithm
- `'grid'` - Grid search

### CV Strategies
- `'kfold'` - K-Fold CV (default)
- `'loso'` - Leave-One-Series-Out
- `'nested'` - Nested CV

### Ensemble Methods
- `'topk'` - Top-K models (default)
- `'voting'` - Voting ensemble
- `'stacking'` - Stacking ensemble

## Common Patterns

### Custom Configuration
```python
automl = AutoMLQSAR(
    featurizers=['rdkit', 'ecfp', 'maccs'],
    models=['ridge', 'randomforest', 'xgboost', 'nn'],
    optimization_method='bayesian',
    cv_strategy='kfold',
    ensemble_method='topk',
    n_trials=100,
    random_state=42
)
```

### Using Pre-computed Features
```python
X = your_features  # numpy array
y = your_targets   # numpy array

automl = AutoMLQSAR()
results = automl.fit(X=X, y=y)
```

### Prediction with Uncertainty
```python
predictions, uncertainties = automl.predict_with_uncertainty(smiles_list)

for smi, pred, unc in zip(smiles_list, predictions, uncertainties):
    print(f"{smi}: {pred:.3f} ± {unc:.3f}")
```

### Model Interpretation
```python
interpretations = automl.interpret()

if 'feature_importance' in interpretations:
    importance = interpretations['feature_importance']
    print("Top 10 features:")
    for feat, score in list(importance.items())[:10]:
        print(f"  {feat}: {score:.4f}")
```

## Module-Level Usage

### Featurization
```python
from automl_qsar.featurization import RDKitDescriptors, ECFPFingerprints

# RDKit descriptors
feat = RDKitDescriptors()
X = feat.featurize(['CCO', 'c1ccccc1'])

# ECFP fingerprints
feat = ECFPFingerprints(radius=2, n_bits=2048)
X = feat.featurize(['CCO', 'c1ccccc1'])
```

### Preprocessing
```python
from automl_qsar.preprocessing import StandardScaler, BorutaSelector

# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Select features
selector = BorutaSelector()
X_selected = selector.fit_transform(X_scaled, y)
```

### Models
```python
from automl_qsar.models import RandomForest, DeepNN

# Random Forest
rf = RandomForest(n_estimators=100)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

# Deep NN
dnn = DeepNN(hidden_layers=[256, 128, 64], epochs=100)
dnn.fit(X_train, y_train)
y_pred = dnn.predict(X_test)
```

### Evaluation
```python
from automl_qsar.evaluation import KFoldCV, Metrics

# Cross-validation
cv = KFoldCV(n_splits=5)
for train_idx, test_idx in cv.split(X, y):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    # Train and evaluate...

# Calculate metrics
rmse = Metrics.rmse(y_true, y_pred)
mae = Metrics.mae(y_true, y_pred)
r2 = Metrics.r2(y_true, y_pred)
q2 = Metrics.q2(y_true, y_pred)
```

### Ensemble
```python
from automl_qsar.ensemble import TopKEnsemble

ensemble = TopKEnsemble(k=5, method='weighted_mean')
ensemble.fit(models, X_val, y_val)
predictions = ensemble.predict(X_test)
```

## Metrics

- **RMSE**: Root Mean Squared Error (lower is better)
- **MAE**: Mean Absolute Error (lower is better)
- **R²**: Coefficient of Determination (higher is better, max 1.0)
- **Q²**: Cross-validated R² (higher is better)
- **CCC**: Concordance Correlation Coefficient (higher is better)

## Tips

1. **Start Simple**: Begin with `['rdkit', 'ecfp']` featurizers and `['ridge', 'randomforest']` models
2. **Scale Data**: Always scale features, especially for neural networks
3. **Feature Selection**: Use when n_features > 1000
4. **Cross-Validation**: Use nested CV for unbiased evaluation
5. **Ensemble**: Top-K ensemble usually gives best performance
6. **Uncertainty**: Always check prediction uncertainty
7. **Interpretation**: Use feature importance to understand models

## Troubleshooting

**ImportError: No module named 'rdkit'**
```bash
conda install -c conda-forge rdkit
```

**Memory Error**
- Reduce fingerprint size: `n_bits=1024` instead of 2048
- Use PCA for dimensionality reduction
- Process data in batches

**Slow Training**
- Reduce `n_trials` for HPO
- Use fewer models
- Enable parallelization: `n_jobs=-1`

**Poor Performance**
- Try more featurizers
- Increase number of models
- Use nested CV for tuning
- Check data quality (duplicates, outliers)

## Resources

- **Documentation**: See DOCUMENTATION.md
- **Examples**: Check examples/ directory
- **GitHub**: https://github.com/bhatnira/Automated---QSAR---Publication-work
- **Issues**: Report bugs on GitHub Issues

## Version

Current Version: 0.1.0
Python: >= 3.8
License: MIT
