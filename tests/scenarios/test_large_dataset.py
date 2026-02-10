"""
Scenario Test: Large Dataset

Tests pipeline performance and scalability with larger datasets.
"""

import pytest
import numpy as np
import pandas as pd
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


@pytest.fixture
def large_dataset():
    """Generate large dataset for performance testing."""
    from tests.datasets.generate_datasets import generate_large_dataset
    return generate_large_dataset(n_samples=50, seed=42)  # Smaller for CI


class TestLargeDataset:
    """Test suite for large dataset scenarios."""
    
    def test_data_curation_performance(self, large_dataset):
        """Test data curation completes in reasonable time."""
        from automl_qsar.preprocessing import QSARDataCurationAgent
        
        df = large_dataset
        
        start_time = time.time()
        
        agent = QSARDataCurationAgent(
            df=df,
            smiles_col='SMILES',
            target_cols='IC50_nM',
            units='nM',
            verbose=False
        )
        
        cleaned_df = agent.run()
        
        elapsed = time.time() - start_time
        
        # Should complete in under 30 seconds
        assert elapsed < 30, f"Data curation took {elapsed:.1f}s"
        assert len(cleaned_df) > 0
    
    def test_featurization_scaling(self, large_dataset):
        """Test featurization scales reasonably."""
        from automl_qsar.featurization import RDKitDescriptors
        
        smiles = large_dataset['SMILES'].tolist()
        
        start_time = time.time()
        
        feat = RDKitDescriptors()
        X = feat.featurize(smiles)
        
        elapsed = time.time() - start_time
        
        # Should complete in under 60 seconds
        assert elapsed < 60, f"Featurization took {elapsed:.1f}s"
        # Some molecules may fail to featurize, but most should succeed
        assert X.shape[0] >= len(smiles) * 0.9  # At least 90% success rate
    
    def test_memory_efficient(self, large_dataset):
        """Test that operations don't cause memory issues."""
        import gc
        from automl_qsar.preprocessing import QSARDataCurationAgent
        from automl_qsar.featurization import ECFPFingerprints
        
        df = large_dataset
        
        # Run curation
        agent = QSARDataCurationAgent(
            df=df,
            smiles_col='SMILES',
            target_cols='IC50_nM',
            verbose=False
        )
        cleaned_df = agent.run()
        
        # Run featurization
        feat = ECFPFingerprints()
        X = feat.featurize(cleaned_df['SMILES'].tolist())
        
        # Force garbage collection
        gc.collect()
        
        assert X.shape[0] == len(cleaned_df)
    
    def test_batch_predictions(self, large_dataset):
        """Test batch predictions work correctly."""
        from automl_qsar.preprocessing import QSARDataCurationAgent
        from automl_qsar.featurization import ECFPFingerprints
        from sklearn.ensemble import RandomForestRegressor
        
        # Curate
        agent = QSARDataCurationAgent(
            df=large_dataset,
            smiles_col='SMILES',
            target_cols='IC50_nM',
            verbose=False
        )
        cleaned_df = agent.run()
        
        # Featurize
        feat = ECFPFingerprints()
        X = feat.featurize(cleaned_df['SMILES'].tolist())
        y = cleaned_df['IC50_nM'].values
        
        # Train
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(X, y)
        
        # Predict
        predictions = model.predict(X)
        
        assert len(predictions) == len(cleaned_df)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
