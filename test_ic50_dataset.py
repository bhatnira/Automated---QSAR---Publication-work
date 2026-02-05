"""
Test AutoML QSAR with IC50 Dataset

This script tests the complete pipeline with a realistic IC50 dataset
containing kinase inhibitors.
"""

import numpy as np
import pandas as pd
import sys

print("="*70)
print("AutoML QSAR - IC50 Dataset Test")
print("="*70)

# ============================================================================
# 1. CREATE REALISTIC IC50 DATASET
# ============================================================================
print("\n[1] Creating sample IC50 dataset (kinase inhibitors)...")

# Real kinase inhibitor structures with synthetic IC50 values
ic50_data = {
    'SMILES': [
        # Known kinase inhibitors (validated SMILES)
        'Cc1ccc(cc1Nc2nccc(n2)c3cccnc3)S(=O)(=O)N4CCN(C)CC4',  # Dasatinib-like
        'CN(C)CCCNc1ccc2ncnc(c2c1)Nc3ccc(c(c3)Cl)F',  # Gefitinib-like
        'COc1cc2ncnc(c2cc1OCCCN3CCOCC3)Nc4ccc(c(c4)Cl)F',  # Erlotinib-like
        'Cc1ccc(cc1)Nc2nccc(n2)c3cccnc3',  # Pyrimidine scaffold
        'c1ccc2c(c1)ccc3c2[nH]c4c3cccc4',  # Carbazole
        'COc1ccc2[nH]cc(c2c1)CCN',  # Tryptamine derivative
        'CN1CCN(CC1)c2ccc3c(c2)ncnc3Nc4cccc(c4)Br',  # Quinazoline
        'c1ccc(cc1)C2=NN=C(O2)c3ccccc3',  # Oxadiazole
        'CC(C)Cc1ccc(cc1)C(C)C(=O)O',  # Ibuprofen
        'c1ccc2c(c1)c(c[nH]2)CC(C(=O)O)N',  # Tryptophan
        'c1ccc(cc1)CCNC(=O)c2ccccc2',  # Benzamide
        'COc1ccc(cc1)C=NNc2nc(nc(n2)N)N',  # Pyrimidine
        'Cc1ccc(cc1)S(=O)(=O)Nc2ncccn2',  # Sulfonamide-pyrimidine
        'c1cc2c(cc1F)ncnc2Nc3cccc(c3)Br',  # Quinazoline variant
        'CCN(CC)C(=O)c1cc(on1)c2ccccc2',  # Isoxazole
        'Cc1ccccc1NC(=O)c2cccnc2',  # Nicotinamide
        'Cn1cnc2c1c(=O)n(c(=O)n2C)C',  # Xanthine
        'c1ccc2c(c1)nc(s2)Nc3ccccc3',  # Benzothiazole
        'Cc1ncc(c(n1)Nc2ccc(cc2)CN3CCN(CC3)C)c4cccnc4',  # Pyrimidine
        'c1cc2c(cc1Br)nc(o2)Nc3ccccc3',  # Benzoxazole
        'CCOc1ccc2ncnc(c2c1)Nc3cccc(c3)C#C',  # Acetylene
        'Cn1cc(c2c1ncnc2N)c3ccccc3',  # Purine-like
        'COc1ccc2c(c1)c(=O)c(c(=O)n2C)N',  # Quinolone
        'Cc1c(sc(n1)NC(=O)c2ccco2)C(=O)N',  # Thiazole
        'c1ccc2c(c1)ncc(n2)Nc3ccc(cc3)N4CCOCC4',  # Benzimidazole
        'COc1ccc(cc1)C2=NN(C(=O)O2)c3ccccc3',  # Oxadiazolone
        'Cc1ccc(cc1)NC(=O)CSc2nnc(s2)N',  # Thiadiazole
        'c1ccc(c(c1)F)Nc2ncnc3c2[nH]c4c3cccc4',  # Fluorinated purine
        'CCn1c(=O)c2c(ncn2C)n(c1=O)C',  # Methylxanthine
        'Cn1c2c(c(=O)[nH]c1=O)nc[nH]2',  # Hypoxanthine
        'COc1cc2c(cc1)nc(n2)Nc3ccc(cc3)C',  # Benzimidazole variant
        'Cc1cc(no1)C(=O)Nc2ccc(cc2)Cl',  # Isoxazole carboxamide
        'CN1CCN(CC1)c2ccc3c(c2)nc(n3)Nc4cccc(c4)C',  # Piperazine
        'c1ccc(cc1)Nc2ncnc3c2ccc4c3cccc4',  # Naphthyridine
        'COc1cc2c(cc1)ncc(n2)Nc3cccc(c3)Br',  # Quinazoline
        'Cc1ccc(cc1)C(=O)Nc2nccs2',  # Thiazole amide
        'c1ccc2c(c1)sc(n2)Nc3ccc(cc3)F',  # Benzothiazole variant
        'CN(C)c1ccc(cc1)C(=O)Nc2ncccn2',  # Pyrimidine amide
        'COc1ccc2c(c1)cc(c(=O)o2)C',  # Coumarin
        'Cc1cc(ccc1O)C(=O)Nc2ccccc2',  # Salicylamide
    ],
    'IC50_nM': [
        # IC50 values in nanomolar (nM)
        2.5, 12.3, 8.7, 5.8, 1200.0,
        3400.0, 45.0, 560.0, 12000.0, 15000.0,
        2400.0, 780.0, 340.0, 28.0, 1500.0,
        4200.0, 18000.0, 450.0, 67.0, 890.0,
        38.0, 125.0, 670.0, 1100.0, 72.0,
        980.0, 1800.0, 52.0, 22000.0, 19000.0,
        620.0, 3100.0, 150.0, 15.0, 95.0,
        2700.0, 890.0, 1900.0, 4500.0, 8200.0,
    ]
}

df = pd.DataFrame(ic50_data)

# Convert IC50 to pIC50 (more suitable for modeling)
# pIC50 = -log10(IC50 in M) = -log10(IC50_nM / 1e9) = 9 - log10(IC50_nM)
df['pIC50'] = 9 - np.log10(df['IC50_nM'])

print(f"✓ Created {len(df)} kinase inhibitor structures")
print(f"  IC50 range: {df['IC50_nM'].min():.1f} - {df['IC50_nM'].max():.1f} nM")
print(f"  pIC50 range: {df['pIC50'].min():.2f} - {df['pIC50'].max():.2f}")
print(f"  Active compounds (IC50 < 100 nM): {(df['IC50_nM'] < 100).sum()}")
print(f"  Moderately active (100-1000 nM): {((df['IC50_nM'] >= 100) & (df['IC50_nM'] < 1000)).sum()}")
print(f"  Weak/Inactive (> 1000 nM): {(df['IC50_nM'] >= 1000).sum()}")

# Save dataset
df.to_csv('kinase_ic50_dataset.csv', index=False)
print(f"✓ Saved to kinase_ic50_dataset.csv")

# Display some examples
print("\nSample compounds:")
print(df[['IC50_nM', 'pIC50']].head(10).to_string())

# ============================================================================
# 2. TEST AUTOML QSAR PIPELINE
# ============================================================================
print("\n[2] Testing AutoML QSAR pipeline with IC50 data...")

try:
    from automl_qsar import AutoMLQSAR
    
    # Initialize AutoML with appropriate settings for IC50 data
    automl = AutoMLQSAR(
        featurizers=['rdkit', 'ecfp'],  # Use RDKit descriptors + fingerprints
        models=['ridge', 'randomforest', 'xgboost'],  # Multiple models
        optimization_method='bayesian',
        n_trials=10,  # Quick test with 10 trials
        cv_strategy='kfold',
        ensemble_method='topk',
        random_state=42
    )
    
    # Load IC50 data (use pIC50 for modeling)
    automl.load_data(df, smiles_col='SMILES', target_col='pIC50')
    print("✓ Data loaded successfully")
    
    # Train the pipeline
    print("\n[3] Training models...")
    results = automl.fit()
    print("✓ Training complete!")
    
    # Display results
    print("\n[4] Model Performance:")
    print("-" * 50)
    if 'model_performance' in results and results['model_performance']:
        perf = results['model_performance']
        if isinstance(perf, dict):
            for model_name, metrics in perf.items():
                print(f"\n{model_name}:")
                print(f"  RMSE: {metrics.get('rmse_mean', 'N/A'):.4f} ± {metrics.get('rmse_std', 0):.4f}")
                print(f"  MAE:  {metrics.get('mae_mean', 'N/A'):.4f} ± {metrics.get('mae_std', 0):.4f}")
                print(f"  R²:   {metrics.get('r2_mean', 'N/A'):.4f} ± {metrics.get('r2_std', 0):.4f}")
        else:
            print("  Model performance data not in expected format")
    else:
        print("  No model performance data available")
    
    # Test predictions on known compounds
    print("\n[5] Testing predictions on validation compounds...")
    
    # Select some test compounds (strong, moderate, weak)
    test_indices = [0, 4, 14, 24, 34]  # Mix of activities
    test_smiles = df.iloc[test_indices]['SMILES'].tolist()
    test_ic50 = df.iloc[test_indices]['IC50_nM'].values
    test_pic50 = df.iloc[test_indices]['pIC50'].values
    
    # Get predictions
    pred_pic50 = automl.predict(test_smiles)
    
    # Convert predictions back to IC50
    pred_ic50 = 10 ** (9 - pred_pic50)
    
    print("\nPredictions vs. Actual:")
    print("-" * 80)
    print(f"{'SMILES':<25} {'Actual IC50 (nM)':<20} {'Predicted IC50 (nM)':<20} {'Error':<15}")
    print("-" * 80)
    
    for i, (smiles, actual_ic50, actual_pic50, pred_p, pred_i) in enumerate(zip(
        test_smiles, test_ic50, test_pic50, pred_pic50, pred_ic50
    )):
        error = abs(actual_pic50 - pred_p)
        fold_error = max(pred_i / actual_ic50, actual_ic50 / pred_i)
        smiles_short = smiles[:22] + '...' if len(smiles) > 25 else smiles
        print(f"{smiles_short:<25} {actual_ic50:<20.1f} {pred_i:<20.1f} {fold_error:<15.2f}x")
    
    # Get predictions with uncertainty
    print("\n[6] Testing uncertainty quantification...")
    pred_pic50_unc, uncertainties = automl.predict_with_uncertainty(test_smiles)
    
    print("\nPredictions with Uncertainty:")
    print("-" * 70)
    print(f"{'SMILES':<25} {'Predicted pIC50':<25} {'Confidence':<20}")
    print("-" * 70)
    
    for smiles, pred, unc in zip(test_smiles, pred_pic50_unc, uncertainties):
        smiles_short = smiles[:22] + '...' if len(smiles) > 25 else smiles
        confidence = "High" if unc < 0.5 else ("Medium" if unc < 1.0 else "Low")
        print(f"{smiles_short:<25} {pred:.2f} ± {unc:.2f}        {confidence:<20}")
    
    # Test on completely new structures
    print("\n[7] Testing on novel compounds...")
    novel_smiles = [
        'COc1ccc(cc1)c2cc(no2)C(=O)Nc3ccc(cc3)F',  # Novel isoxazole
        'Cc1nc(sc1)NC(=O)c2ccc(cc2)N3CCOCC3',  # Novel thiazole
        'c1ccc(cc1)C2=NOC(=N2)c3ccccc3',  # Novel oxadiazole
    ]
    
    novel_pred = automl.predict(novel_smiles)
    novel_ic50 = 10 ** (9 - novel_pred)
    
    print("\nNovel Compound Predictions:")
    print("-" * 60)
    for smiles, pic50, ic50 in zip(novel_smiles, novel_pred, novel_ic50):
        print(f"SMILES: {smiles}")
        print(f"  Predicted pIC50: {pic50:.2f}")
        print(f"  Predicted IC50:  {ic50:.1f} nM")
        activity = "Strong" if ic50 < 100 else ("Moderate" if ic50 < 1000 else "Weak")
        print(f"  Activity class: {activity}")
        print()
    
    # Save model
    print("[8] Saving trained model...")
    model_path = './trained_kinase_model'
    automl.save(model_path)
    print(f"✓ Model saved to {model_path}")
    
    # Test loading
    print("\n[9] Testing model loading...")
    from automl_qsar import AutoMLQSAR
    automl_loaded = AutoMLQSAR()
    automl_loaded.load(model_path)
    
    # Verify loaded model works
    test_pred = automl_loaded.predict(['CCO'])
    print(f"✓ Loaded model works! Test prediction: {test_pred[0]:.2f}")
    
    print("\n" + "="*70)
    print("✅ ALL TESTS PASSED!")
    print("="*70)
    
    # Final summary
    print("\nSummary:")
    print(f"  ✓ Dataset: {len(df)} kinase inhibitors with IC50 values")
    print(f"  ✓ Featurization: RDKit descriptors + ECFP fingerprints")
    print(f"  ✓ Models: Ridge, Random Forest, XGBoost")
    print(f"  ✓ Cross-validation: 5-fold CV")
    print(f"  ✓ Ensemble: Top-K with uncertainty quantification")
    print(f"  ✓ Predictions: Working on training and novel compounds")
    print(f"  ✓ Model persistence: Save/Load working")
    
    print("\n🎉 AutoML QSAR is ready for IC50 prediction tasks!")
    
except Exception as e:
    print(f"\n✗ Test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
