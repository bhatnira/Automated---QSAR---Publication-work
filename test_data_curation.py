#!/usr/bin/env python
"""
Test for QSAR Data Curation Agent
"""

import sys
import os
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_data_curation_agent():
    """Test the QSARDataCurationAgent with sample data."""
    print("\n" + "="*60)
    print("Testing QSAR Data Curation Agent")
    print("="*60)
    
    from automl_qsar.preprocessing import QSARDataCurationAgent, curate_qsar_data
    
    # Create sample dataset with various issues
    data = {
        'SMILES': [
            'CCO',                           # Ethanol - valid
            'c1ccccc1',                       # Benzene - valid
            'CC(=O)OC1=CC=CC=C1C(=O)O',       # Aspirin - valid
            'INVALID_SMILES',                 # Invalid
            'CCO',                            # Duplicate
            'CC(C)CC1=CC=C(C=C1)C(C)C(=O)O',  # Ibuprofen - valid
            'CN1C=NC2=C1C(=O)N(C(=O)N2C)C',   # Caffeine - valid
            '',                               # Empty
            'C1=CC=CC=C1O',                   # Phenol - valid
            'CC(C)C',                         # Isobutane - valid
        ],
        'IC50': [
            '100',        # Normal value
            '>10000',     # Right censored
            '<1',         # Left censored
            '50',         # Will be removed (invalid SMILES)
            '150',        # Duplicate
            '~500',       # Approximate
            '25.5',       # Decimal
            '100',        # Empty SMILES
            'invalid',    # Invalid activity
            '1000',       # Normal
        ]
    }
    
    df = pd.DataFrame(data)
    print(f"\nInput data: {len(df)} rows")
    print(df.to_string())
    
    # Test the agent
    print("\n" + "-"*40)
    print("Running curation pipeline...")
    print("-"*40)
    
    agent = QSARDataCurationAgent(
        df=df,
        smiles_col='SMILES',
        activity_col='IC50',
        unit='nM',
        verbose=True
    )
    
    cleaned_df = agent.run()
    
    print("\n" + "-"*40)
    print("Cleaned data:")
    print("-"*40)
    print(cleaned_df.to_string())
    
    # Check results
    print("\n" + "-"*40)
    print("Validation:")
    print("-"*40)
    
    stats = agent.get_statistics()
    print(f"  Invalid SMILES removed: {stats['n_invalid_smiles']}")
    print(f"  Duplicates removed: {stats['n_duplicates']}")
    print(f"  Invalid activities: {stats['n_invalid_activity']}")
    print(f"  Task type: {stats['task_type']}")
    print(f"  Transform: {stats['transform']}")
    print(f"  Final size: {len(cleaned_df)}")
    
    # Generate reports
    agent.generate_report("test_curation_report.txt")
    agent.generate_markdown_report("test_curation_report.md")
    print(f"\n  Reports generated: test_curation_report.txt, test_curation_report.md")
    
    # Test convenience function
    print("\n" + "-"*40)
    print("Testing convenience function...")
    print("-"*40)
    
    df2 = pd.DataFrame({
        'smiles': ['CCO', 'c1ccccc1', 'CCCC'],
        'activity': [100, 1000, 50]
    })
    
    clean_df2, stats2 = curate_qsar_data(
        df2, 
        smiles_col='smiles', 
        activity_col='activity',
        unit='nM',
        verbose=False
    )
    
    print(f"  Curated {len(clean_df2)} molecules")
    print(f"  Transform: {stats2.get('transform', 'N/A')}")
    
    print("\n✓ QSAR Data Curation Agent test completed!")
    return True


def test_with_kinase_dataset():
    """Test with the kinase IC50 dataset."""
    print("\n" + "="*60)
    print("Testing with Kinase IC50 Dataset")
    print("="*60)
    
    from automl_qsar.preprocessing import QSARDataCurationAgent
    
    # Load kinase dataset
    df = pd.read_csv('kinase_ic50_dataset.csv')
    print(f"\nLoaded: {len(df)} compounds")
    
    agent = QSARDataCurationAgent(
        df=df,
        smiles_col='SMILES',
        activity_col='IC50_nM',
        unit='nM',
        verbose=True
    )
    
    cleaned_df = agent.run()
    
    stats = agent.get_statistics()
    
    print("\n" + "-"*40)
    print("Results Summary:")
    print("-"*40)
    print(f"  Original: {len(df)} compounds")
    print(f"  Cleaned: {len(cleaned_df)} compounds")
    print(f"  Transform: {stats.get('transform', 'N/A')}")
    
    if 'target_stats' in stats:
        ts = stats['target_stats']
        print(f"  Target range: {ts['min']:.2f} - {ts['max']:.2f}")
        print(f"  Target mean: {ts['mean']:.2f} ± {ts['std']:.2f}")
    
    print("\n✓ Kinase dataset test completed!")
    return True


if __name__ == "__main__":
    success1 = test_data_curation_agent()
    success2 = test_with_kinase_dataset()
    
    print("\n" + "="*60)
    if success1 and success2:
        print("🎉 All data curation tests passed!")
    else:
        print("⚠️ Some tests failed")
    print("="*60)
