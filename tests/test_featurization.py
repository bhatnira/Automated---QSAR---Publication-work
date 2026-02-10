"""
Unit tests for featurization module
"""

import unittest
import numpy as np
from automl_qsar.featurization import (
    RDKitDescriptors, ECFPFingerprints, MACCSFingerprints
)


class TestFeaturizers(unittest.TestCase):
    """Test molecular featurizers"""
    
    def setUp(self):
        """Set up test data"""
        self.test_smiles = ['CCO', 'c1ccccc1', 'CC(=O)O']
    
    def test_rdkit_descriptors(self):
        """Test RDKit descriptors"""
        featurizer = RDKitDescriptors()
        X = featurizer.featurize(self.test_smiles)
        
        self.assertEqual(X.shape[0], len(self.test_smiles))
        self.assertGreater(X.shape[1], 0)
        self.assertFalse(np.any(np.isnan(X)))
    
    def test_ecfp_fingerprints(self):
        """Test ECFP fingerprints"""
        featurizer = ECFPFingerprints(radius=2, n_bits=1024)
        X = featurizer.featurize(self.test_smiles)
        
        self.assertEqual(X.shape, (len(self.test_smiles), 1024))
        self.assertTrue(np.all((X == 0) | (X == 1)))
    
    def test_maccs_fingerprints(self):
        """Test MACCS fingerprints"""
        featurizer = MACCSFingerprints()
        X = featurizer.featurize(self.test_smiles)
        
        self.assertEqual(X.shape, (len(self.test_smiles), 167))  # MACCS has 167 keys
        self.assertTrue(np.all((X == 0) | (X == 1)))


if __name__ == '__main__':
    unittest.main()
