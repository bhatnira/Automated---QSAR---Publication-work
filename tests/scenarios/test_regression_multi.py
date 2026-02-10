"""
Scenario Test: Multi-Target Regression

Tests the pipeline for multi-endpoint IC50 prediction.
"""

import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


@pytest.fixture
def multi_target_dataset():
    """Generate multi-target regression dataset."""
    from tests.datasets.generate_datasets import generate_regression_multi_dataset
    return generate_regression_multi_dataset(n_samples=30, n_targets=3, seed=42)


class TestMultiTargetRegression:
    """Test suite for multi-target regression scenarios."""
    
    def test_data_curation_multi_target(self, multi_target_dataset):
        """Test data curation identifies multi-target regression."""
        from automl_qsar.preprocessing import QSARDataCurationAgent
        
        df = multi_target_dataset
        target_cols = [c for c in df.columns if c != 'SMILES']
        
        agent = QSARDataCurationAgent(
            df=df,
            smiles_col='SMILES',
            target_cols=target_cols,
            units='nM',
            verbose=False
        )
        
        cleaned_df = agent.run()
        stats = agent.get_statistics()
        
        assert stats['overall_task'] == 'regression_multi'
        assert len(agent.target_configs) == 3
    
    def test_per_target_transformation(self, multi_target_dataset):
        """Test that each target gets appropriate transformation."""
        from automl_qsar.preprocessing import QSARDataCurationAgent
        
        df = multi_target_dataset
        target_cols = [c for c in df.columns if c != 'SMILES']
        
        agent = QSARDataCurationAgent(
            df=df,
            smiles_col='SMILES',
            target_cols=target_cols,
            units='nM',
            verbose=False
        )
        
        agent.run()
        configs = agent.get_target_configs()
        
        # Each target should have its own config
        for col in target_cols:
            assert col in configs
            config = configs[col]
            assert config.transform in ['none', 'pIC50']
    
    def test_different_units_per_target(self):
        """Test handling different units for each target."""
        from automl_qsar.preprocessing import QSARDataCurationAgent
        from tests.datasets.generate_datasets import generate_regression_multi_dataset
        
        df = generate_regression_multi_dataset(n_samples=20, n_targets=2, seed=42)
        
        # Assign different units
        units = {
            'IC50_TargetA': 'nM',
            'IC50_TargetB': 'uM'
        }
        
        agent = QSARDataCurationAgent(
            df=df,
            smiles_col='SMILES',
            target_cols=['IC50_TargetA', 'IC50_TargetB'],
            units=units,
            verbose=False
        )
        
        cleaned_df = agent.run()
        assert len(cleaned_df) > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
