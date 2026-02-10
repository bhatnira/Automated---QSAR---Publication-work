"""
Scenario Test: Mixed Tasks

Tests the pipeline for datasets with both regression and classification targets.
"""

import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


@pytest.fixture
def mixed_dataset():
    """Generate mixed tasks dataset."""
    from tests.datasets.generate_datasets import generate_mixed_tasks_dataset
    return generate_mixed_tasks_dataset(n_samples=40, seed=42)


class TestMixedTasks:
    """Test suite for mixed task scenarios."""
    
    def test_task_inference_mixed(self, mixed_dataset):
        """Test that mixed task type is correctly inferred."""
        from automl_qsar.preprocessing import QSARDataCurationAgent
        
        df = mixed_dataset
        target_cols = ['IC50_nM', 'Toxicity', 'Selectivity']
        
        agent = QSARDataCurationAgent(
            df=df,
            smiles_col='SMILES',
            target_cols=target_cols,
            units={'IC50_nM': 'nM', 'Toxicity': None, 'Selectivity': None},
            verbose=False
        )
        
        cleaned_df = agent.run()
        stats = agent.get_statistics()
        
        assert stats['overall_task'] == 'mixed'
    
    def test_individual_target_types(self, mixed_dataset):
        """Test that each target gets correct task type."""
        from automl_qsar.preprocessing import QSARDataCurationAgent
        from automl_qsar.preprocessing.data_curation import TaskType
        
        df = mixed_dataset
        target_cols = ['IC50_nM', 'Toxicity', 'Selectivity']
        
        agent = QSARDataCurationAgent(
            df=df,
            smiles_col='SMILES',
            target_cols=target_cols,
            units={'IC50_nM': 'nM', 'Toxicity': None, 'Selectivity': None},
            verbose=False
        )
        
        agent.run()
        configs = agent.get_target_configs()
        
        # IC50 should be regression
        assert configs['IC50_nM'].task_type == TaskType.REGRESSION_SINGLE
        
        # Toxicity should be binary classification
        assert configs['Toxicity'].task_type == TaskType.CLASSIFICATION_BINARY
        
        # Selectivity should be multiclass
        assert configs['Selectivity'].task_type == TaskType.CLASSIFICATION_MULTICLASS
    
    def test_appropriate_transforms(self, mixed_dataset):
        """Test that appropriate transforms are applied."""
        from automl_qsar.preprocessing import QSARDataCurationAgent
        
        df = mixed_dataset
        target_cols = ['IC50_nM', 'Toxicity', 'Selectivity']
        
        agent = QSARDataCurationAgent(
            df=df,
            smiles_col='SMILES',
            target_cols=target_cols,
            units={'IC50_nM': 'nM', 'Toxicity': None, 'Selectivity': None},
            verbose=False
        )
        
        agent.run()
        configs = agent.get_target_configs()
        
        # Regression: pIC50 or none
        assert configs['IC50_nM'].transform in ['pIC50', 'none']
        
        # Classification: label_encode
        assert configs['Toxicity'].transform == 'label_encode'
        assert configs['Selectivity'].transform == 'label_encode'
    
    def test_label_encoders_only_for_classification(self, mixed_dataset):
        """Test that label encoders are created only for classification."""
        from automl_qsar.preprocessing import QSARDataCurationAgent
        
        df = mixed_dataset
        target_cols = ['IC50_nM', 'Toxicity', 'Selectivity']
        
        agent = QSARDataCurationAgent(
            df=df,
            smiles_col='SMILES',
            target_cols=target_cols,
            units={'IC50_nM': 'nM', 'Toxicity': None, 'Selectivity': None},
            verbose=False
        )
        
        agent.run()
        encoders = agent.get_label_encoders()
        
        # Should have encoders for classification targets
        assert 'Toxicity' in encoders
        assert 'Selectivity' in encoders
        
        # Should NOT have encoder for regression target
        assert 'IC50_nM' not in encoders
    
    def test_with_solubility_regression(self, mixed_dataset):
        """Test handling multiple regression + classification."""
        from automl_qsar.preprocessing import QSARDataCurationAgent
        
        df = mixed_dataset
        target_cols = ['IC50_nM', 'Solubility_logS', 'Toxicity']
        
        agent = QSARDataCurationAgent(
            df=df,
            smiles_col='SMILES',
            target_cols=target_cols,
            units={'IC50_nM': 'nM', 'Solubility_logS': None, 'Toxicity': None},
            verbose=False
        )
        
        agent.run()
        stats = agent.get_statistics()
        
        assert stats['overall_task'] == 'mixed'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
