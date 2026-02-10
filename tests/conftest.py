"""
Pytest Configuration and Fixtures for QSAR Tests
"""

import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Ensure the package is importable
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture(scope="session")
def sample_smiles():
    """Provide sample SMILES strings for testing."""
    return [
        'CCO',                  # Ethanol
        'c1ccccc1',             # Benzene
        'CC(C)O',               # Isopropanol
        'CCN',                  # Ethylamine
        'CCCC',                 # Butane
        'c1ccc(O)cc1',          # Phenol
        'CC(=O)O',              # Acetic acid
        'CCCN',                 # Propylamine
        'c1ccc(N)cc1',          # Aniline
        'CCCO',                 # Propanol
    ]


@pytest.fixture(scope="session")
def extended_smiles():
    """Provide extended SMILES set for larger tests."""
    return [
        'CCO', 'c1ccccc1', 'CC(C)O', 'CCN', 'CCCC', 'c1ccc(O)cc1',
        'CC(=O)O', 'CCCN', 'c1ccc(N)cc1', 'CCCO', 'CCCCO', 'CC(C)(C)O',
        'c1ccc(C)cc1', 'CC=O', 'CCC=O', 'CCCC=O', 'c1ccc(CC)cc1',
        'CC(=O)C', 'CC(=O)CC', 'CCOC', 'CCOCC', 'c1ccc(OC)cc1',
        'CC#N', 'CCC#N', 'c1ccc(C#N)cc1', 'CC(C)N', 'CC(C)CC',
        'c1ccc(Cl)cc1', 'c1ccc(F)cc1', 'c1ccc(Br)cc1',
    ]


@pytest.fixture
def regression_dataframe(sample_smiles):
    """Create a regression dataset for testing."""
    np.random.seed(42)
    ic50 = np.random.uniform(1e-9, 1e-5, len(sample_smiles))
    
    return pd.DataFrame({
        'SMILES': sample_smiles,
        'IC50': ic50
    })


@pytest.fixture
def multiregression_dataframe(sample_smiles):
    """Create a multi-target regression dataset."""
    np.random.seed(42)
    
    return pd.DataFrame({
        'SMILES': sample_smiles,
        'IC50_Target1': np.random.uniform(1e-9, 1e-5, len(sample_smiles)),
        'IC50_Target2': np.random.uniform(1e-8, 1e-6, len(sample_smiles)),
        'Ki': np.random.uniform(1e-9, 1e-7, len(sample_smiles)),
    })


@pytest.fixture
def classification_dataframe(sample_smiles):
    """Create a binary classification dataset."""
    np.random.seed(42)
    labels = np.random.choice(['Active', 'Inactive'], len(sample_smiles))
    
    return pd.DataFrame({
        'SMILES': sample_smiles,
        'Activity': labels
    })


@pytest.fixture
def multiclass_dataframe(sample_smiles):
    """Create a multiclass classification dataset."""
    np.random.seed(42)
    labels = np.random.choice(['High', 'Medium', 'Low'], len(sample_smiles))
    
    return pd.DataFrame({
        'SMILES': sample_smiles,
        'Potency': labels
    })


@pytest.fixture
def mixed_task_dataframe(sample_smiles):
    """Create a mixed task dataset."""
    np.random.seed(42)
    
    return pd.DataFrame({
        'SMILES': sample_smiles,
        'IC50': np.random.uniform(1e-9, 1e-5, len(sample_smiles)),
        'Activity': np.random.choice(['Active', 'Inactive'], len(sample_smiles)),
    })


@pytest.fixture
def invalid_smiles_dataframe():
    """Create a dataset with some invalid SMILES."""
    return pd.DataFrame({
        'SMILES': ['CCO', 'INVALID', 'c1ccccc1', 'NOT_A_SMILES', 'CC(C)O'],
        'IC50': [1e-9, 1e-8, 1e-7, 1e-6, 1e-5]
    })


@pytest.fixture
def empty_dataframe():
    """Create an empty dataset."""
    return pd.DataFrame(columns=['SMILES', 'IC50'])


# Markers for slow and integration tests
def pytest_configure(config):
    """Configure custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "requires_rdkit: marks tests that require RDKit"
    )
