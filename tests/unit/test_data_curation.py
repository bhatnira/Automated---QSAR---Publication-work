"""
Unit Tests for Data Curation Module

Tests individual functions and classes in the data curation module.
"""

import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from automl_qsar.preprocessing.data_curation import (
    TaskType,
    TargetConfig,
    infer_task_type,
    QSARDataCurationAgent,
    curate_qsar_data
)


class TestTaskType:
    """Tests for TaskType enum."""
    
    def test_enum_values(self):
        """Test that all task types have correct values."""
        assert TaskType.REGRESSION_SINGLE.value == 'regression_single'
        assert TaskType.REGRESSION_MULTI.value == 'regression_multi'
        assert TaskType.CLASSIFICATION_BINARY.value == 'classification_binary'
        assert TaskType.CLASSIFICATION_MULTICLASS.value == 'classification_multiclass'
        assert TaskType.MIXED.value == 'mixed'


class TestInferTaskType:
    """Tests for task type inference function."""
    
    def test_regression_continuous(self):
        """Continuous float values should be regression."""
        series = pd.Series([1.5, 2.3, 4.7, 8.9, 12.1, 100.5, 250.0, 1000.0])
        task = infer_task_type(series)
        assert task == TaskType.REGRESSION_SINGLE
    
    def test_classification_binary_strings(self):
        """Two string labels should be binary classification."""
        series = pd.Series(['Active', 'Inactive', 'Active', 'Inactive'])
        task = infer_task_type(series)
        assert task == TaskType.CLASSIFICATION_BINARY
    
    def test_classification_multiclass_strings(self):
        """Multiple string labels should be multiclass."""
        series = pd.Series(['High', 'Medium', 'Low', 'High', 'Low'])
        task = infer_task_type(series)
        assert task == TaskType.CLASSIFICATION_MULTICLASS
    
    def test_classification_binary_integers(self):
        """Small integer labels (0,1) should be binary."""
        series = pd.Series([0, 1, 0, 1, 1, 0])
        task = infer_task_type(series)
        assert task == TaskType.CLASSIFICATION_BINARY
    
    def test_empty_series(self):
        """Empty series should return unknown."""
        series = pd.Series([], dtype=float)
        task = infer_task_type(series)
        assert task == TaskType.UNKNOWN
    
    def test_all_nan(self):
        """All NaN series should return unknown."""
        series = pd.Series([np.nan, np.nan, np.nan])
        task = infer_task_type(series)
        assert task == TaskType.UNKNOWN


class TestTargetConfig:
    """Tests for TargetConfig dataclass."""
    
    def test_default_values(self):
        """Test default values are set correctly."""
        config = TargetConfig(name='test')
        
        assert config.name == 'test'
        assert config.task_type == TaskType.UNKNOWN
        assert config.unit == 'nM'
        assert config.transform == 'none'
        assert config.n_classes == 0
        assert config.classes == []
    
    def test_custom_values(self):
        """Test custom values are set correctly."""
        config = TargetConfig(
            name='IC50',
            task_type=TaskType.REGRESSION_SINGLE,
            unit='uM',
            transform='pIC50',
            n_classes=0,
            classes=[]
        )
        
        assert config.name == 'IC50'
        assert config.task_type == TaskType.REGRESSION_SINGLE
        assert config.unit == 'uM'
        assert config.transform == 'pIC50'


class TestQSARDataCurationAgent:
    """Tests for QSARDataCurationAgent class."""
    
    @pytest.fixture
    def sample_df(self):
        """Create sample DataFrame for testing."""
        return pd.DataFrame({
            'SMILES': ['CCO', 'c1ccccc1', 'CC(C)O', 'CCN', 'CCC'],
            'IC50_nM': [100, 500, 200, 1000, 50]
        })
    
    def test_initialization(self, sample_df):
        """Test agent initialization."""
        agent = QSARDataCurationAgent(
            df=sample_df,
            smiles_col='SMILES',
            target_cols='IC50_nM',
            units='nM',
            verbose=False
        )
        
        assert agent.smiles_col == 'SMILES'
        assert agent.target_cols == ['IC50_nM']
        assert agent.units == {'IC50_nM': 'nM'}
    
    def test_list_target_cols(self, sample_df):
        """Test with list of target columns."""
        sample_df['Activity'] = ['Active', 'Inactive', 'Active', 'Inactive', 'Active']
        
        agent = QSARDataCurationAgent(
            df=sample_df,
            smiles_col='SMILES',
            target_cols=['IC50_nM', 'Activity'],
            verbose=False
        )
        
        assert agent.target_cols == ['IC50_nM', 'Activity']
    
    def test_run_returns_dataframe(self, sample_df):
        """Test that run() returns a DataFrame."""
        agent = QSARDataCurationAgent(
            df=sample_df,
            smiles_col='SMILES',
            target_cols='IC50_nM',
            verbose=False
        )
        
        result = agent.run()
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0
    
    def test_get_statistics(self, sample_df):
        """Test statistics retrieval."""
        agent = QSARDataCurationAgent(
            df=sample_df,
            smiles_col='SMILES',
            target_cols='IC50_nM',
            verbose=False
        )
        
        agent.run()
        stats = agent.get_statistics()
        
        assert isinstance(stats, dict)
        assert 'overall_task' in stats
    
    def test_get_target_configs(self, sample_df):
        """Test target config retrieval."""
        agent = QSARDataCurationAgent(
            df=sample_df,
            smiles_col='SMILES',
            target_cols='IC50_nM',
            verbose=False
        )
        
        agent.run()
        configs = agent.get_target_configs()
        
        assert isinstance(configs, dict)
        assert 'IC50_nM' in configs


class TestCurateQSARData:
    """Tests for convenience function."""
    
    def test_convenience_function(self):
        """Test curate_qsar_data convenience function."""
        df = pd.DataFrame({
            'SMILES': ['CCO', 'c1ccccc1', 'CC(C)O'],
            'IC50': [100, 500, 200]
        })
        
        cleaned_df, stats = curate_qsar_data(
            df,
            smiles_col='SMILES',
            target_cols='IC50',
            units='nM',
            verbose=False
        )
        
        assert isinstance(cleaned_df, pd.DataFrame)
        assert isinstance(stats, dict)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
