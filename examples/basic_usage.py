"""
Example usage of AutoML QSAR
"""

import numpy as np
import pandas as pd
from automl_qsar import AutoMLQSAR

# Example: Create synthetic QSAR data
def generate_example_data():
    """Generate example molecular data"""
    # Example SMILES strings (drug-like molecules)
    smiles = [
        'CCO',  # Ethanol
        'CC(C)O',  # Isopropanol
        'CCCC',  # Butane
        'c1ccccc1',  # Benzene
        'CC(=O)O',  # Acetic acid
        'CCN',  # Ethylamine
        'CCOC',  # Diethyl ether
        'CC(C)C',  # Isobutane
        'c1ccc(O)cc1',  # Phenol
        'CC(C)(C)O',  # tert-Butanol
    ]
    
    # Synthetic activity values
    activity = np.random.randn(len(smiles)) * 2 + 5
    
    return pd.DataFrame({
        'SMILES': smiles,
        'Activity': activity
    })

def main():
    """Main example"""
    print("AutoML QSAR Example")
    print("="*50)
    
    # Generate example data
    data = generate_example_data()
    print(f"\nGenerated {len(data)} example molecules")
    print(data.head())
    
    # Initialize AutoML QSAR
    automl = AutoMLQSAR(
        featurizers=['rdkit', 'ecfp'],
        models=['ridge', 'randomforest'],
        optimization_method='bayesian',
        cv_strategy='kfold',
        ensemble_method='topk',
        n_trials=10,
        random_state=42
    )
    
    # Load data
    automl.load_data(data, smiles_col='SMILES', target_col='Activity')
    
    # Run pipeline
    results = automl.fit()
    
    # Make predictions
    test_smiles = ['CCO', 'c1ccccc1']
    predictions = automl.predict(test_smiles)
    
    print(f"\nPredictions for test molecules:")
    for smile, pred in zip(test_smiles, predictions):
        print(f"  {smile}: {pred:.4f}")
    
    # Get uncertainty estimates
    predictions, uncertainties = automl.predict_with_uncertainty(test_smiles)
    print(f"\nPredictions with uncertainty:")
    for smile, pred, unc in zip(test_smiles, predictions, uncertainties):
        print(f"  {smile}: {pred:.4f} ± {unc:.4f}")
    
    # Interpret model
    interpretations = automl.interpret()
    
    # Save model
    automl.save('./trained_model')
    print("\nModel saved to ./trained_model")

if __name__ == '__main__':
    main()
