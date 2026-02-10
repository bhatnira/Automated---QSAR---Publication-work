"""
Integration Tests for Full QSAR Pipeline
"""

import pytest
import numpy as np
import pandas as pd
import sys
import tempfile
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


@pytest.fixture
def regression_dataset():
    """Create a simple regression dataset."""
    smiles = [
        'CCO', 'c1ccccc1', 'CC(C)O', 'CCN', 'CCCC', 'c1ccc(O)cc1',
        'CC(=O)O', 'CCO', 'CCCN', 'c1ccc(N)cc1', 'CCCO', 'CCCCO',
        'CC(C)(C)O', 'c1ccc(C)cc1', 'CC=O', 'CCC=O', 'CCCC=O'
    ]
    np.random.seed(42)
    ic50 = np.random.uniform(1e-9, 1e-5, len(smiles))
    
    return pd.DataFrame({
        'SMILES': smiles,
        'IC50': ic50
    })


@pytest.fixture
def classification_dataset():
    """Create a simple classification dataset."""
    smiles = [
        'CCO', 'c1ccccc1', 'CC(C)O', 'CCN', 'CCCC', 'c1ccc(O)cc1',
        'CC(=O)O', 'CCO', 'CCCN', 'c1ccc(N)cc1', 'CCCO', 'CCCCO',
        'CC(C)(C)O', 'c1ccc(C)cc1', 'CC=O', 'CCC=O', 'CCCC=O'
    ]
    np.random.seed(42)
    activity = np.random.choice(['Active', 'Inactive'], len(smiles))
    
    return pd.DataFrame({
        'SMILES': smiles,
        'Activity': activity
    })


class TestDataCurationIntegration:
    """Integration tests for data curation pipeline."""
    
    def test_curation_with_regression(self, regression_dataset):
        """Test data curation for regression task."""
        from automl_qsar.preprocessing.data_curation import QSARDataCurationAgent
        
        agent = QSARDataCurationAgent(regression_dataset)
        result = agent.run()
        
        assert result is not None
        assert len(result) > 0
    
    def test_curation_with_classification(self, classification_dataset):
        """Test data curation for classification task."""
        from automl_qsar.preprocessing.data_curation import QSARDataCurationAgent
        
        agent = QSARDataCurationAgent(classification_dataset)
        result = agent.run()
        
        assert result is not None
        assert len(result) > 0


class TestFeaturizationIntegration:
    """Integration tests for featurization pipeline."""
    
    def test_rdkit_descriptors_pipeline(self, regression_dataset):
        """Test RDKit descriptors in full pipeline."""
        from automl_qsar.preprocessing.data_curation import QSARDataCurationAgent
        from automl_qsar.featurization import RDKitDescriptors
        
        # Curate data
        agent = QSARDataCurationAgent(regression_dataset)
        result = agent.run()
        
        # Featurize
        featurizer = RDKitDescriptors()
        smiles = result['SMILES'].tolist()
        X = featurizer.featurize(smiles)
        
        assert X.shape[0] == len(smiles)
        assert X.shape[1] > 0
    
    def test_fingerprint_pipeline(self, regression_dataset):
        """Test fingerprints in full pipeline."""
        from automl_qsar.preprocessing.data_curation import QSARDataCurationAgent
        from automl_qsar.featurization import ECFPFingerprints
        
        # Curate data
        agent = QSARDataCurationAgent(regression_dataset)
        result = agent.run()
        
        # Featurize
        featurizer = ECFPFingerprints(radius=2, n_bits=1024)
        smiles = result['SMILES'].tolist()
        X = featurizer.featurize(smiles)
        
        assert X.shape == (len(smiles), 1024)


class TestModelTrainingIntegration:
    """Integration tests for model training pipeline."""
    
    def test_train_regression_model(self, regression_dataset):
        """Test full regression model training."""
        from automl_qsar.preprocessing.data_curation import QSARDataCurationAgent
        from automl_qsar.featurization import RDKitDescriptors
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.model_selection import cross_val_score
        
        # Curate
        agent = QSARDataCurationAgent(regression_dataset)
        df = agent.run()
        
        # Featurize
        featurizer = RDKitDescriptors()
        X = featurizer.featurize(df['SMILES'].tolist())
        
        # Get target
        target_col = [c for c in df.columns if c != 'SMILES'][0]
        y = df[target_col].values
        
        # Handle NaN features
        X = np.nan_to_num(X, nan=0.0)
        
        # Train model
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        
        # Cross-validate
        if len(y) >= 5:
            scores = cross_val_score(model, X, y, cv=min(3, len(y)), scoring='r2')
            assert len(scores) == min(3, len(y))
    
    def test_train_classification_model(self, classification_dataset):
        """Test full classification model training."""
        from automl_qsar.preprocessing.data_curation import QSARDataCurationAgent
        from automl_qsar.featurization import ECFPFingerprints
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import cross_val_score
        
        # Curate
        agent = QSARDataCurationAgent(classification_dataset)
        df = agent.run()
        
        # Featurize
        featurizer = ECFPFingerprints(radius=2, n_bits=1024)
        X = featurizer.featurize(df['SMILES'].tolist())
        
        # Get target
        target_col = [c for c in df.columns if c != 'SMILES'][0]
        y = df[target_col].values
        
        # Train model
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        
        # Cross-validate
        if len(y) >= 5:
            scores = cross_val_score(model, X, y, cv=min(3, len(y)), scoring='accuracy')
            assert len(scores) == min(3, len(y))


class TestModelPersistenceIntegration:
    """Integration tests for model saving and loading."""
    
    def test_save_and_load_model(self, regression_dataset):
        """Test model persistence."""
        from automl_qsar.preprocessing.data_curation import QSARDataCurationAgent
        from automl_qsar.featurization import RDKitDescriptors
        from sklearn.ensemble import RandomForestRegressor
        import pickle
        
        # Curate and featurize
        agent = QSARDataCurationAgent(regression_dataset)
        df = agent.run()
        
        featurizer = RDKitDescriptors()
        X = featurizer.featurize(df['SMILES'].tolist())
        X = np.nan_to_num(X, nan=0.0)
        
        target_col = [c for c in df.columns if c != 'SMILES'][0]
        y = df[target_col].values
        
        # Train model
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(X, y)
        
        # Save and load
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.pkl', delete=False) as f:
            pickle.dump(model, f)
            temp_path = f.name
        
        try:
            with open(temp_path, 'rb') as f:
                loaded_model = pickle.load(f)
            
            # Verify predictions match
            original_pred = model.predict(X)
            loaded_pred = loaded_model.predict(X)
            
            np.testing.assert_array_almost_equal(original_pred, loaded_pred)
        finally:
            os.unlink(temp_path)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
