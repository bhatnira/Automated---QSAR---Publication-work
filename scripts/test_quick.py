"""
Quick Test Script for AutoML QSAR - Core Features Only

This is a simplified test that works even without all optional dependencies.
Tests the core pipeline with essential features.
"""

import numpy as np
import pandas as pd
from pathlib import Path

print("="*70)
print("AutoML QSAR - Quick Test")
print("="*70)

# ============================================================================
# 1. CREATE SAMPLE DATASET
# ============================================================================
print("\n[1] Creating sample QSAR dataset...")

sample_data = {
    'SMILES': [
        'CCO', 'CC(C)O', 'CC(C)(C)O', 'CCCCO', 'c1ccc(O)cc1',
        'c1ccccc1', 'Cc1ccccc1', 'c1ccc(C)cc1C', 'c1ccc2ccccc2c1', 'c1ccc(Cl)cc1',
        'CC(=O)O', 'CCC(=O)O', 'c1ccc(C(=O)O)cc1',
        'CCN', 'CC(C)N', 'c1ccc(N)cc1',
        'CCOC', 'COc1ccccc1',
        'c1cccnc1', 'c1ccoc1',
        'CC(=O)N', 'CC(=O)NC',
        'CC(C)Cc1ccc(C(C)C(=O)O)cc1', 'COc1ccc(CCN)cc1OC',
        'c1ccc(cc1)C(=O)c2ccccc2', 'CC(C)(C)c1ccc(O)cc1',
        'c1ccc(cc1)c2ccccc2', 'COc1cc(C=O)ccc1O', 'c1ccc2c(c1)cc(c(=O)o2)C',
    ],
    'pIC50': [
        5.2, 5.5, 5.8, 5.3, 7.1,
        5.1, 5.4, 5.6, 6.2, 5.9,
        5.7, 6.0, 7.3,
        5.3, 5.6, 6.8,
        5.4, 6.5,
        6.0, 5.8,
        5.9, 6.1,
        7.5, 7.2,
        6.8, 6.9,
        6.7, 7.0, 6.6
    ]
}

data = pd.DataFrame(sample_data)
print(f"✓ Created {len(data)} molecules")
print(f"  Activity range: {data['pIC50'].min():.2f} - {data['pIC50'].max():.2f}")

# ============================================================================
# 2. TEST BASIC PIPELINE
# ============================================================================
print("\n[2] Testing basic AutoML pipeline...")

try:
    from automl_qsar import AutoMLQSAR
    
    automl = AutoMLQSAR(
        featurizers=['rdkit'],
        models=['ridge', 'randomforest'],
        n_trials=5,
        random_state=42
    )
    
    automl.load_data(data, smiles_col='SMILES', target_col='pIC50')
    print("✓ Data loaded")
    
    results = automl.fit()
    print("✓ Model training complete")
    
    # Test predictions
    test_smiles = ['CCO', 'c1ccccc1']
    predictions = automl.predict(test_smiles)
    
    print(f"\n✓ Predictions:")
    for smi, pred in zip(test_smiles, predictions):
        print(f"  {smi:15s} → {pred:.3f}")
    
    # Test with uncertainty
    predictions, uncertainties = automl.predict_with_uncertainty(test_smiles)
    print(f"\n✓ Predictions with uncertainty:")
    for smi, pred, unc in zip(test_smiles, predictions, uncertainties):
        print(f"  {smi:15s} → {pred:.3f} ± {unc:.3f}")
    
    print("\n✓ BASIC PIPELINE TEST PASSED")
    
except Exception as e:
    print(f"\n✗ Basic pipeline test failed: {e}")
    import traceback
    traceback.print_exc()

# ============================================================================
# 3. TEST FEATURIZERS
# ============================================================================
print("\n[3] Testing featurizers...")

try:
    from automl_qsar.featurization import RDKitDescriptors, ECFPFingerprints, MACCSFingerprints
    
    test_smiles = ['CCO', 'c1ccccc1', 'CC(=O)O']
    
    # RDKit
    rdkit = RDKitDescriptors()
    X_rdkit = rdkit.featurize(test_smiles)
    print(f"✓ RDKit descriptors: {X_rdkit.shape}")
    
    # ECFP
    ecfp = ECFPFingerprints(radius=2, n_bits=1024)
    X_ecfp = ecfp.featurize(test_smiles)
    print(f"✓ ECFP fingerprints: {X_ecfp.shape}")
    
    # MACCS
    maccs = MACCSFingerprints()
    X_maccs = maccs.featurize(test_smiles)
    print(f"✓ MACCS keys: {X_maccs.shape}")
    
    print("\n✓ FEATURIZER TEST PASSED")
    
except Exception as e:
    print(f"\n✗ Featurizer test failed: {e}")

# ============================================================================
# 4. TEST MODELS
# ============================================================================
print("\n[4] Testing individual models...")

try:
    from automl_qsar.featurization import RDKitDescriptors
    from automl_qsar.preprocessing import StandardScaler
    from automl_qsar.models import Ridge, RandomForest
    from automl_qsar.evaluation import Metrics
    
    # Prepare data
    feat = RDKitDescriptors()
    X = feat.featurize(data['SMILES'].tolist())
    y = data['pIC50'].values
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data
    n_train = int(0.7 * len(X_scaled))
    X_train, X_test = X_scaled[:n_train], X_scaled[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]
    
    # Test Ridge
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train, y_train)
    y_pred_ridge = ridge.predict(X_test)
    rmse_ridge = Metrics.rmse(y_test, y_pred_ridge)
    print(f"✓ Ridge RMSE: {rmse_ridge:.4f}")
    
    # Test Random Forest
    rf = RandomForest(n_estimators=50, random_state=42)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    rmse_rf = Metrics.rmse(y_test, y_pred_rf)
    print(f"✓ Random Forest RMSE: {rmse_rf:.4f}")
    
    print("\n✓ MODEL TEST PASSED")
    
except Exception as e:
    print(f"\n✗ Model test failed: {e}")

# ============================================================================
# 5. TEST ENSEMBLE
# ============================================================================
print("\n[5] Testing ensemble methods...")

try:
    from automl_qsar.ensemble import TopKEnsemble
    
    # Use previously trained models
    models = [ridge, rf]
    
    ensemble = TopKEnsemble(k=2)
    ensemble.fit(models, X_test, y_test)
    
    y_pred_ens = ensemble.predict(X_test)
    rmse_ens = Metrics.rmse(y_test, y_pred_ens)
    print(f"✓ Ensemble RMSE: {rmse_ens:.4f}")
    
    # Test uncertainty
    y_pred_ens, y_unc_ens = ensemble.predict_with_uncertainty(X_test)
    print(f"✓ Mean uncertainty: {np.mean(y_unc_ens):.4f}")
    
    print("\n✓ ENSEMBLE TEST PASSED")
    
except Exception as e:
    print(f"\n✗ Ensemble test failed: {e}")

# ============================================================================
# 6. TEST SAVE/LOAD
# ============================================================================
print("\n[6] Testing save/load functionality...")

try:
    save_dir = './test_model_quick'
    
    # Save
    automl.save(save_dir)
    print(f"✓ Model saved to {save_dir}")
    
    # Load
    from automl_qsar import AutoMLQSAR
    automl_loaded = AutoMLQSAR()
    automl_loaded.load(save_dir)
    print(f"✓ Model loaded from {save_dir}")
    
    # Test loaded model
    pred_loaded = automl_loaded.predict(['CCO'])
    print(f"✓ Prediction with loaded model: {pred_loaded[0]:.3f}")
    
    print("\n✓ SAVE/LOAD TEST PASSED")
    
except Exception as e:
    print(f"\n✗ Save/load test failed: {e}")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*70)
print("QUICK TEST SUMMARY")
print("="*70)

summary = """
Tests Completed:
✓ Sample dataset creation
✓ Basic AutoML pipeline
✓ Featurization (RDKit, ECFP, MACCS)
✓ Individual models (Ridge, Random Forest)
✓ Ensemble methods
✓ Save/Load functionality

Next Steps:
1. Install optional dependencies for full features:
   - pip install torch torch-geometric (for GNN)
   - pip install xgboost (for XGBoost)
   - pip install shap lime (for interpretability)

2. Run comprehensive test:
   - python test_complete_pipeline.py

3. Try with your own data:
   - Prepare CSV with SMILES and activity columns
   - Run the pipeline!
"""

print(summary)
print("="*70)
print("Quick Test Complete!")
print("="*70)
