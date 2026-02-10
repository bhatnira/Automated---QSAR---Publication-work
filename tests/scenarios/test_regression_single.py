"""
Scenario Test: Single-Target Regression

Tests the complete pipeline for IC50 prediction with a single endpoint.
"""

import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


@pytest.fixture
def regression_dataset():
    """Generate single-target regression dataset."""
    from tests.datasets.generate_datasets import generate_regression_single_dataset
    return generate_regression_single_dataset(n_samples=30, seed=42)


class TestSingleTargetRegression:
    """Test suite for single-target regression scenarios."""
    
    def test_data_curation(self, regression_dataset):
        """Test data curation for regression."""
        from automl_qsar.preprocessing import QSARDataCurationAgent
        
        df = regression_dataset
        agent = QSARDataCurationAgent(
            df=df,
            smiles_col='SMILES',
            target_cols='IC50_nM',
            units='nM',
            verbose=False
        )
        
        cleaned_df = agent.run()
        stats = agent.get_statistics()
        
        assert stats['overall_task'] == 'regression_single'
        assert len(cleaned_df) > 0
    
    def test_featurization(self, regression_dataset):
        """Test molecular featurization."""
        from automl_qsar.featurization import RDKitDescriptors, ECFPFingerprints
        
        # Use fewer SMILES and filter to valid ones first
        smiles = regression_dataset['SMILES'].tolist()[:10]
        
        # Test RDKit descriptors - may filter invalid molecules
        rdkit_feat = RDKitDescriptors()
        X_rdkit = rdkit_feat.featurize(smiles)
        assert X_rdkit.shape[0] > 0  # At least some molecules featurized
        assert X_rdkit.shape[1] > 0
        
        # Test ECFP fingerprints
        ecfp_feat = ECFPFingerprints()
        X_ecfp = ecfp_feat.featurize(smiles)
        assert X_ecfp.shape[0] > 0  # At least some molecules featurized
    
    def test_pipeline_integration(self, regression_dataset):
        """Test full pipeline with data curation and featurization."""
        from automl_qsar.preprocessing import QSARDataCurationAgent
        from automl_qsar.featurization import ECFPFingerprints
        from sklearn.ensemble import RandomForestRegressor
        
        # Curate data
        agent = QSARDataCurationAgent(
            df=regression_dataset,
            smiles_col='SMILES',
            target_cols='IC50_nM',
            verbose=False
        )
        cleaned_df = agent.run()
        
        # Featurize
        featurizer = ECFPFingerprints()
        X = featurizer.featurize(cleaned_df['SMILES'].tolist())
        y = cleaned_df['IC50_nM'].values
        
        # Train model
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(X, y)
        
        # Predict
        predictions = model.predict(X)
        assert len(predictions) == len(y)
    
    def test_pIC50_transformation(self, regression_dataset):
        """Test pIC50 transformation for regression via units parameter."""
        from automl_qsar.preprocessing import QSARDataCurationAgent
        
        agent = QSARDataCurationAgent(
            df=regression_dataset,
            smiles_col='SMILES',
            target_cols='IC50_nM',
            units='nM',  # Setting units triggers pIC50 handling
            verbose=False
        )
        
        cleaned_df = agent.run()
        
        # Check target column exists and has valid values
        target_col = [c for c in cleaned_df.columns if c != 'SMILES'][0]
        values = cleaned_df[target_col].values
        
        # All values should be finite
        assert np.all(np.isfinite(values))
        assert len(values) > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
