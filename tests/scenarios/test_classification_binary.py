"""
Scenario Test: Binary Classification

Tests the pipeline for binary activity classification (Active/Inactive).
"""

import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


@pytest.fixture
def binary_dataset():
    """Generate binary classification dataset."""
    from tests.datasets.generate_datasets import generate_classification_binary_dataset
    return generate_classification_binary_dataset(n_samples=50, seed=42)


class TestBinaryClassification:
    """Test suite for binary classification scenarios."""
    
    def test_task_inference(self, binary_dataset):
        """Test that binary classification is correctly inferred."""
        from automl_qsar.preprocessing import QSARDataCurationAgent
        
        df = binary_dataset
        
        agent = QSARDataCurationAgent(
            df=df,
            smiles_col='SMILES',
            target_cols='Activity',
            units=None,
            verbose=False
        )
        
        cleaned_df = agent.run()
        stats = agent.get_statistics()
        
        assert stats['overall_task'] == 'classification_binary'
    
    def test_label_encoding(self, binary_dataset):
        """Test that string labels are encoded correctly."""
        from automl_qsar.preprocessing import QSARDataCurationAgent
        
        df = binary_dataset
        
        agent = QSARDataCurationAgent(
            df=df,
            smiles_col='SMILES',
            target_cols='Activity',
            units=None,
            verbose=False
        )
        
        cleaned_df = agent.run()
        encoders = agent.get_label_encoders()
        
        # Should have label encoder
        assert 'Activity' in encoders
        
        # Encoded values should be 0 and 1
        unique_vals = cleaned_df['Activity'].unique()
        assert set(unique_vals) == {0, 1}
    
    def test_class_balance(self, binary_dataset):
        """Test handling class balance information."""
        from automl_qsar.preprocessing import QSARDataCurationAgent
        
        df = binary_dataset
        
        agent = QSARDataCurationAgent(
            df=df,
            smiles_col='SMILES',
            target_cols='Activity',
            units=None,
            verbose=False
        )
        
        agent.run()
        configs = agent.get_target_configs()
        
        config = configs['Activity']
        assert 'class_counts' in config.statistics
        assert config.n_classes == 2
    
    def test_inverse_transform(self, binary_dataset):
        """Test that predictions can be decoded back to labels."""
        from automl_qsar.preprocessing import QSARDataCurationAgent
        
        df = binary_dataset
        
        agent = QSARDataCurationAgent(
            df=df,
            smiles_col='SMILES',
            target_cols='Activity',
            units=None,
            verbose=False
        )
        
        agent.run()
        encoders = agent.get_label_encoders()
        
        le = encoders['Activity']
        
        # Test inverse transform
        encoded = [0, 1, 0, 1]
        decoded = le.inverse_transform(encoded)
        
        assert all(d in ['Active', 'Inactive'] for d in decoded)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
