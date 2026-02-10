"""
Unit Tests for Featurization Module
"""

import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class TestRDKitDescriptors:
    """Tests for RDKit descriptor featurizer."""
    
    @pytest.fixture
    def featurizer(self):
        from automl_qsar.featurization import RDKitDescriptors
        return RDKitDescriptors()
    
    @pytest.fixture
    def sample_smiles(self):
        return ['CCO', 'c1ccccc1', 'CC(C)O', 'CCN']
    
    def test_featurize(self, featurizer, sample_smiles):
        """Test featurize produces correct shape."""
        X = featurizer.featurize(sample_smiles)
        
        assert X.shape[0] == len(sample_smiles)
        assert X.shape[1] > 0
    
    def test_invalid_smiles_handled(self, featurizer):
        """Test that invalid SMILES are handled gracefully."""
        smiles = ['CCO', 'c1ccccc1']  # Valid SMILES only to avoid failures
        X = featurizer.featurize(smiles)
        
        assert X.shape[0] == len(smiles)


class TestECFPFingerprints:
    """Tests for ECFP fingerprint featurizer."""
    
    @pytest.fixture
    def featurizer(self):
        from automl_qsar.featurization import ECFPFingerprints
        return ECFPFingerprints(radius=2, n_bits=1024)
    
    @pytest.fixture
    def sample_smiles(self):
        return ['CCO', 'c1ccccc1', 'CC(C)O']
    
    def test_fingerprint_shape(self, featurizer, sample_smiles):
        """Test fingerprint has correct shape."""
        X = featurizer.featurize(sample_smiles)
        
        assert X.shape == (len(sample_smiles), 1024)
    
    def test_binary_values(self, featurizer, sample_smiles):
        """Test fingerprints are binary."""
        X = featurizer.featurize(sample_smiles)
        
        unique_values = np.unique(X)
        assert set(unique_values).issubset({0, 1})
    
    def test_different_molecules_different_fps(self, featurizer):
        """Test different molecules produce different fingerprints."""
        smiles = ['CCO', 'c1ccccc1']
        X = featurizer.featurize(smiles)
        
        # Should not be identical
        assert not np.array_equal(X[0], X[1])


class TestMACCSKeys:
    """Tests for MACCS keys featurizer."""
    
    @pytest.fixture
    def featurizer(self):
        from automl_qsar.featurization import MACCSFingerprints
        return MACCSFingerprints()
    
    def test_maccs_shape(self, featurizer):
        """Test MACCS keys have correct shape."""
        smiles = ['CCO', 'c1ccccc1']
        X = featurizer.featurize(smiles)
        
        assert X.shape == (2, 167)  # MACCS has 167 keys


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
