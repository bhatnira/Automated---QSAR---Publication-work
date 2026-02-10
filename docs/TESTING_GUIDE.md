# AutoML QSAR - Testing Guide

Complete guide for testing all features with sample QSAR datasets.

## Quick Start

### 1. Install Dependencies

```bash
# Core dependencies
pip install -r requirements.txt

# Install RDKit (recommended via conda)
conda install -c conda-forge rdkit

# Or via pip (may have issues on some systems)
pip install rdkit
```

### 2. Run Quick Test (Essential Features Only)

```bash
python test_quick.py
```

This tests core functionality without heavy dependencies:
- ✓ Basic pipeline
- ✓ RDKit featurization
- ✓ Linear and tree-based models
- ✓ Ensemble methods
- ✓ Save/load functionality

**Expected Output:**
```
[1] Creating sample QSAR dataset...
✓ Created 29 molecules
✓ Data loaded
✓ Model training complete
✓ Predictions:
  CCO             → 5.234
  c1ccccc1        → 5.187
✓ BASIC PIPELINE TEST PASSED
```

### 3. Run Comprehensive Test (All Features)

```bash
python test_complete_pipeline.py
```

This tests all advanced features:
- ✓ All 6 featurizers
- ✓ All 12 model types
- ✓ HPO methods (Bayesian, Genetic, Grid)
- ✓ Cross-validation strategies
- ✓ All ensemble methods
- ✓ Uncertainty quantification
- ✓ Model interpretation (SHAP, LIME)
- ✓ Visualization

**Expected Output:**
```
[1] Creating sample QSAR dataset...
[2] Testing basic pipeline...
[3] Testing all featurizers...
[4] Testing all models...
[5] Testing HPO methods...
[6] Testing cross-validation...
[7] Testing ensemble methods...
[8] Testing uncertainty quantification...
[9] Testing interpretation...
[10] Testing save/load...
[11] Testing visualization...
[12] Running comprehensive end-to-end test...
```

## Detailed Testing Scenarios

### Test 1: Basic Usage with Real Data

```python
from automl_qsar import AutoMLQSAR
import pandas as pd

# Load your data
data = pd.read_csv('your_data.csv')
# Expected columns: 'SMILES', 'Activity'

# Create pipeline
automl = AutoMLQSAR(
    featurizers=['rdkit', 'ecfp'],
    models=['ridge', 'randomforest', 'xgboost'],
    n_trials=20,
    random_state=42
)

# Train
automl.load_data(data, smiles_col='SMILES', target_col='Activity')
results = automl.fit()

# Predict
test_smiles = ['CCO', 'c1ccccc1']
predictions = automl.predict(test_smiles)
```

### Test 2: Advanced Pipeline with HPO

```python
from automl_qsar import AutoMLQSAR

automl = AutoMLQSAR(
    featurizers=['rdkit', 'ecfp', 'maccs'],
    models=['ridge', 'randomforest', 'svr', 'mlp'],
    optimization_method='bayesian',
    n_trials=50,
    cv_folds=5,
    ensemble_method='stacking',
    random_state=42
)

automl.load_data(data, smiles_col='SMILES', target_col='Activity')
results = automl.fit()

# Get predictions with uncertainty
predictions, uncertainties = automl.predict_with_uncertainty(test_smiles)

# Interpret best model
importance = automl.interpret(method='shap')
```

### Test 3: Custom Featurization

```python
from automl_qsar.featurization import RDKitDescriptors, ECFPFingerprints
from automl_qsar import AutoMLQSAR

# Use specific featurizers
rdkit = RDKitDescriptors()
ecfp = ECFPFingerprints(radius=3, n_bits=2048)

# Manual featurization
smiles_list = data['SMILES'].tolist()
X_rdkit = rdkit.featurize(smiles_list)
X_ecfp = ecfp.featurize(smiles_list)

# Combine features
import numpy as np
X_combined = np.hstack([X_rdkit, X_ecfp])
```

### Test 4: Model Comparison

```python
from automl_qsar.models import Ridge, RandomForest, XGBoost, MLP
from automl_qsar.evaluation import Metrics, KFoldCV
from automl_qsar.featurization import RDKitDescriptors

# Prepare features
feat = RDKitDescriptors()
X = feat.featurize(smiles_list)
y = data['Activity'].values

# Initialize models
models = {
    'Ridge': Ridge(),
    'Random Forest': RandomForest(n_estimators=100),
    'XGBoost': XGBoost(n_estimators=100),
    'MLP': MLP(hidden_layers=[128, 64])
}

# Evaluate each model
cv = KFoldCV(n_splits=5)
results = {}

for name, model in models.items():
    cv_results = cv.cross_validate(model, X, y, metrics=['rmse', 'r2'])
    results[name] = cv_results
    print(f"{name}: RMSE = {cv_results['rmse_mean']:.4f} ± {cv_results['rmse_std']:.4f}")
```

### Test 5: Ensemble Methods

```python
from automl_qsar.ensemble import VotingEnsemble, StackingEnsemble, TopKEnsemble
from automl_qsar.models import Ridge, RandomForest, XGBoost

# Train individual models
models = []
for ModelClass in [Ridge, RandomForest, XGBoost]:
    model = ModelClass()
    model.fit(X_train, y_train)
    models.append(model)

# Voting ensemble
voting = VotingEnsemble(method='mean')
voting.fit(models, X_val, y_val)
pred_voting = voting.predict(X_test)

# Stacking ensemble
stacking = StackingEnsemble()
stacking.fit(models, X_val, y_val)
pred_stacking = stacking.predict(X_test)

# Top-K ensemble
topk = TopKEnsemble(k=2)
topk.fit(models, X_val, y_val)
pred_topk, unc_topk = topk.predict_with_uncertainty(X_test)
```

### Test 6: Hyperparameter Optimization

```python
from automl_qsar.optimization import BayesianOptimization, GeneticOptimization
from automl_qsar.models import RandomForest

# Bayesian optimization
optimizer = BayesianOptimization(
    model_class=RandomForest,
    n_trials=50,
    random_state=42
)

best_params = optimizer.optimize(X_train, y_train, X_val, y_val)
print(f"Best parameters: {best_params}")

# Genetic optimization
optimizer_ga = GeneticOptimization(
    model_class=RandomForest,
    population_size=20,
    n_generations=10
)

best_params_ga = optimizer_ga.optimize(X_train, y_train, X_val, y_val)
```

### Test 7: Cross-Validation

```python
from automl_qsar.evaluation import KFoldCV, LOSOCV, NestedCV

# K-Fold CV
kfold = KFoldCV(n_splits=5)
results = kfold.cross_validate(model, X, y, metrics=['rmse', 'mae', 'r2'])

# Leave-One-Structure-Out CV
loso = LOSOCV()
results = loso.cross_validate(model, X, y, groups, metrics=['rmse', 'r2'])

# Nested CV for unbiased performance estimation
nested = NestedCV(outer_cv=5, inner_cv=3)
results = nested.nested_cross_validate(model, X, y)
```

### Test 8: Model Interpretation

```python
from automl_qsar.interpretation import FeatureImportance, SHAPInterpreter, LIMEInterpreter

# Feature importance
fi = FeatureImportance(model)
importance = fi.get_importance(X)
top_features = fi.get_top_features(X, n=10)

# SHAP values
shap_interp = SHAPInterpreter(model)
shap_values = shap_interp.explain(X, feature_names)
shap_interp.plot_summary(X, feature_names)

# LIME explanations
lime_interp = LIMEInterpreter(model, X_train)
explanation = lime_interp.explain_instance(X_test[0], feature_names)
lime_interp.plot_explanation(explanation)
```

## Sample Datasets

### Creating Synthetic QSAR Data

```python
import pandas as pd

# Small druglike molecules with synthetic pIC50 values
sample_data = {
    'SMILES': [
        'CCO', 'CC(C)O', 'CC(C)(C)O', 'CCCCO', 'c1ccc(O)cc1',
        'c1ccccc1', 'Cc1ccccc1', 'c1ccc(C)cc1C', 'c1ccc2ccccc2c1',
        'CC(=O)O', 'CCC(=O)O', 'c1ccc(C(=O)O)cc1',
        'CCN', 'CC(C)N', 'c1ccc(N)cc1',
        'CCOC', 'COc1ccccc1', 'c1cccnc1', 'c1ccoc1',
        'CC(=O)N', 'CC(=O)NC',
    ],
    'pIC50': [
        5.2, 5.5, 5.8, 5.3, 7.1,
        5.1, 5.4, 5.6, 6.2,
        5.7, 6.0, 7.3,
        5.3, 5.6, 6.8,
        5.4, 6.5, 6.0, 5.8,
        5.9, 6.1
    ]
}

df = pd.DataFrame(sample_data)
df.to_csv('sample_qsar_dataset.csv', index=False)
```

### Using Public QSAR Datasets

```python
# From ChEMBL
from chembl_webresource_client.new_client import new_client

target = new_client.target
activities = new_client.activity

# Get activities for a specific target
target_query = target.filter(pref_name__iexact='Acetylcholinesterase')
target_chembl_id = target_query[0]['target_chembl_id']

activities = activities.filter(
    target_chembl_id=target_chembl_id,
    standard_type='IC50',
    pchembl_value__isnull=False
)

# Convert to dataframe
data = pd.DataFrame(activities)
data = data[['canonical_smiles', 'pchembl_value']]
data.columns = ['SMILES', 'pIC50']
```

## Troubleshooting

### Issue: RDKit not found
```bash
# Solution: Install via conda
conda install -c conda-forge rdkit
```

### Issue: CUDA errors with PyTorch
```bash
# Solution: Install CPU version
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### Issue: Memory errors with large datasets
```python
# Solution: Use feature selection or dimensionality reduction
automl = AutoMLQSAR(
    featurizers=['rdkit'],
    preprocessing={
        'feature_selection': {'method': 'variance', 'threshold': 0.01}
    }
)
```

### Issue: Slow training
```python
# Solution: Reduce hyperparameter search space
automl = AutoMLQSAR(
    models=['randomforest'],  # Use fewer models
    n_trials=10,  # Reduce trials
    cv_folds=3,   # Reduce CV folds
)
```

## Performance Benchmarks

Expected training times on a modern laptop (M1/M2 or Intel i7):

| Dataset Size | Featurizers | Models | HPO Trials | Time    |
|--------------|-------------|--------|------------|---------|
| 100 mols     | RDKit       | 2      | 10         | ~30s    |
| 500 mols     | RDKit+ECFP  | 4      | 20         | ~5 min  |
| 1000 mols    | All         | 8      | 50         | ~30 min |
| 5000 mols    | RDKit+ECFP  | 4      | 20         | ~2 hr   |

## Validation

To ensure your installation is working correctly:

```bash
# Run quick test (should complete in <1 min)
python test_quick.py

# Run comprehensive test (may take 10-30 min)
python test_complete_pipeline.py

# Check output files
ls -lh automl_qsar_test_results.png
ls -lh test_model_*/
```

## Next Steps

1. **Start with quick test**: `python test_quick.py`
2. **Try with your data**: Prepare CSV with SMILES and activity
3. **Optimize pipeline**: Adjust featurizers, models, and HPO settings
4. **Validate results**: Use nested CV for unbiased performance
5. **Interpret models**: Use SHAP/LIME to understand predictions
6. **Deploy**: Save model and use for new predictions

## Additional Resources

- **Documentation**: See `DOCUMENTATION.md` for detailed API reference
- **Quick Reference**: See `QUICK_REFERENCE.md` for common operations
- **Architecture**: See `ARCHITECTURE.md` for design details
- **Examples**: Check `examples/` directory for more scripts

## Support

If you encounter issues:
1. Check that all dependencies are installed
2. Verify your data format (SMILES column + activity column)
3. Try the quick test first
4. Review error messages carefully

Happy modeling! 🧪🤖
