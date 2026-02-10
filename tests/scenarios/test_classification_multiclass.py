"""
Scenario Test: Multiclass Classification

Tests the pipeline for multiclass potency classification (High/Medium/Low).
"""

import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


@pytest.fixture
def multiclass_dataset():
    """Generate multiclass classification dataset."""
    from tests.datasets.generate_datasets import generate_classification_multiclass_dataset
    return generate_classification_multiclass_dataset(n_samples=60, n_classes=3, seed=42)


class TestMulticlassClassification:
    """Test suite for multiclass classification scenarios."""
    
    def test_task_inference(self, multiclass_dataset):
        """Test that multiclass classification is correctly inferred."""
        from automl_qsar.preprocessing import QSARDataCurationAgent
        
        df = multiclass_dataset
        
        agent = QSARDataCurationAgent(
            df=df,
            smiles_col='SMILES',
            target_cols='Potency',
            units=None,
            verbose=False
        )
        
        cleaned_df = agent.run()
        stats = agent.get_statistics()
        
        assert stats['overall_task'] == 'classification_multiclass'
    
    def test_multiple_classes_encoded(self, multiclass_dataset):
        """Test that all classes are properly encoded."""
        from automl_qsar.preprocessing import QSARDataCurationAgent
        
        df = multiclass_dataset
        
        agent = QSARDataCurationAgent(
            df=df,
            smiles_col='SMILES',
            target_cols='Potency',
            units=None,
            verbose=False
        )
        
        cleaned_df = agent.run()
        configs = agent.get_target_configs()
        
        config = configs['Potency']
        assert config.n_classes == 3
        assert set(config.classes) == {'High', 'Low', 'Medium'}
    
    def test_class_distribution(self, multiclass_dataset):
        """Test class distribution is captured."""
        from automl_qsar.preprocessing import QSARDataCurationAgent
        
        df = multiclass_dataset
        
        agent = QSARDataCurationAgent(
            df=df,
            smiles_col='SMILES',
            target_cols='Potency',
            units=None,
            verbose=False
        )
        
        agent.run()
        configs = agent.get_target_configs()
        
        config = configs['Potency']
        class_counts = config.statistics.get('class_counts', {})
        
        # All classes should have some samples
        assert len(class_counts) == 3
        assert all(count > 0 for count in class_counts.values())
    
    def test_four_class_scenario(self):
        """Test handling 4+ class scenario."""
        from automl_qsar.preprocessing import QSARDataCurationAgent
        from tests.datasets.generate_datasets import generate_classification_multiclass_dataset
        
        df = generate_classification_multiclass_dataset(n_samples=40, n_classes=4, seed=42)
        
        agent = QSARDataCurationAgent(
            df=df,
            smiles_col='SMILES',
            target_cols='Potency',
            units=None,
            verbose=False
        )
        
        agent.run()
        configs = agent.get_target_configs()
        
        config = configs['Potency']
        assert config.n_classes == 4


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
