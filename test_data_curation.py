#!/usr/bin/env python
"""
Test for QSAR Data Curation Agent - Multi-Task Support

Tests:
1. Single-target regression (IC50)
2. Multi-target regression (multiple endpoints)
3. Binary classification (Active/Inactive)
4. Multiclass classification (High/Medium/Low)
5. Mixed tasks (regression + classification)
6. String label handling
"""

import sys
import os
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_single_target_regression():
    """Test single-target regression (IC50 values)."""
    print("\n" + "="*60)
    print("Test 1: Single-Target Regression")
    print("="*60)
    
    from automl_qsar.preprocessing import QSARDataCurationAgent
    
    # Create sample IC50 dataset
    data = {
        'SMILES': [
            'CCO', 'c1ccccc1', 'CC(=O)OC1=CC=CC=C1C(=O)O',
            'CC(C)CC1=CC=C(C=C1)C(C)C(=O)O', 'CN1C=NC2=C1C(=O)N(C(=O)N2C)C',
            'CC(C)C', 'CCCC', 'c1ccc2ccccc2c1'
        ],
        'IC50_nM': [100, 1000, 50, 500, 25, 10000, 5000, 200]
    }
    
    df = pd.DataFrame(data)
    print(f"Input: {len(df)} compounds with IC50 values")
    
    agent = QSARDataCurationAgent(
        df=df,
        smiles_col='SMILES',
        target_cols='IC50_nM',
        units='nM',
        verbose=True
    )
    
    cleaned_df = agent.run()
    
    print(f"\nResult:")
    print(cleaned_df.head())
    
    stats = agent.get_statistics()
    print(f"\nOverall task: {stats['overall_task']}")
    
    config = agent.get_target_configs()['IC50_nM']
    print(f"Transform applied: {config.transform}")
    
    assert stats['overall_task'] == 'regression_single'
    print("\n✓ Single-target regression test passed!")
    return True


def test_multi_target_regression():
    """Test multi-target regression (multiple IC50 endpoints)."""
    print("\n" + "="*60)
    print("Test 2: Multi-Target Regression")
    print("="*60)
    
    from automl_qsar.preprocessing import QSARDataCurationAgent
    
    # Create multi-target dataset
    data = {
        'SMILES': [
            'CCO', 'c1ccccc1', 'CC(=O)OC1=CC=CC=C1C(=O)O',
            'CC(C)CC1=CC=C(C=C1)C(C)C(=O)O', 'CN1C=NC2=C1C(=O)N(C(=O)N2C)C',
        ],
        'IC50_TargetA': [100, 1000, 50, 500, 25],
        'IC50_TargetB': [200, 500, 100, 1000, 50],
        'Ki_TargetC': [50, 200, 25, 250, 10]
    }
    
    df = pd.DataFrame(data)
    print(f"Input: {len(df)} compounds with 3 targets")
    
    agent = QSARDataCurationAgent(
        df=df,
        smiles_col='SMILES',
        target_cols=['IC50_TargetA', 'IC50_TargetB', 'Ki_TargetC'],
        units={'IC50_TargetA': 'nM', 'IC50_TargetB': 'nM', 'Ki_TargetC': 'nM'},
        verbose=True
    )
    
    cleaned_df = agent.run()
    
    print(f"\nResult:")
    print(cleaned_df.head())
    
    stats = agent.get_statistics()
    print(f"\nOverall task: {stats['overall_task']}")
    
    assert stats['overall_task'] == 'regression_multi'
    print("\n✓ Multi-target regression test passed!")
    return True


def test_binary_classification():
    """Test binary classification with string labels."""
    print("\n" + "="*60)
    print("Test 3: Binary Classification (String Labels)")
    print("="*60)
    
    from automl_qsar.preprocessing import QSARDataCurationAgent
    
    # Create binary classification dataset
    data = {
        'SMILES': [
            'CCO', 'c1ccccc1', 'CC(=O)OC1=CC=CC=C1C(=O)O',
            'CC(C)CC1=CC=C(C=C1)C(C)C(=O)O', 'CN1C=NC2=C1C(=O)N(C(=O)N2C)C',
            'CC(C)C', 'CCCC', 'c1ccc2ccccc2c1'
        ],
        'Activity': ['Active', 'Inactive', 'Active', 'Active', 
                     'Inactive', 'Inactive', 'Active', 'Inactive']
    }
    
    df = pd.DataFrame(data)
    print(f"Input: {len(df)} compounds with Activity labels")
    
    agent = QSARDataCurationAgent(
        df=df,
        smiles_col='SMILES',
        target_cols='Activity',
        units=None,
        verbose=True
    )
    
    cleaned_df = agent.run()
    
    print(f"\nResult:")
    print(cleaned_df.head())
    
    stats = agent.get_statistics()
    print(f"\nOverall task: {stats['overall_task']}")
    
    config = agent.get_target_configs()['Activity']
    print(f"Classes: {config.classes}")
    print(f"Transform: {config.transform}")
    
    # Get label encoder
    encoders = agent.get_label_encoders()
    if 'Activity' in encoders:
        print(f"Label encoder classes: {list(encoders['Activity'].classes_)}")
    
    assert stats['overall_task'] == 'classification_binary'
    print("\n✓ Binary classification test passed!")
    return True


def test_multiclass_classification():
    """Test multiclass classification with string labels."""
    print("\n" + "="*60)
    print("Test 4: Multiclass Classification")
    print("="*60)
    
    from automl_qsar.preprocessing import QSARDataCurationAgent
    
    # Create multiclass dataset
    data = {
        'SMILES': [
            'CCO', 'c1ccccc1', 'CC(=O)OC1=CC=CC=C1C(=O)O',
            'CC(C)CC1=CC=C(C=C1)C(C)C(=O)O', 'CN1C=NC2=C1C(=O)N(C(=O)N2C)C',
            'CC(C)C', 'CCCC', 'c1ccc2ccccc2c1', 'CCN', 'CCOCC'
        ],
        'Potency': ['High', 'Low', 'High', 'Medium', 'High',
                    'Low', 'Medium', 'Medium', 'Low', 'High']
    }
    
    df = pd.DataFrame(data)
    print(f"Input: {len(df)} compounds with Potency labels")
    
    agent = QSARDataCurationAgent(
        df=df,
        smiles_col='SMILES',
        target_cols='Potency',
        units=None,
        verbose=True
    )
    
    cleaned_df = agent.run()
    
    print(f"\nResult:")
    print(cleaned_df.head())
    
    stats = agent.get_statistics()
    print(f"\nOverall task: {stats['overall_task']}")
    
    config = agent.get_target_configs()['Potency']
    print(f"Classes: {config.classes}")
    print(f"Class counts: {config.statistics.get('class_counts', {})}")
    
    assert stats['overall_task'] == 'classification_multiclass'
    print("\n✓ Multiclass classification test passed!")
    return True


def test_mixed_tasks():
    """Test mixed regression and classification tasks."""
    print("\n" + "="*60)
    print("Test 5: Mixed Tasks (Regression + Classification)")
    print("="*60)
    
    from automl_qsar.preprocessing import QSARDataCurationAgent
    
    # Create mixed task dataset
    data = {
        'SMILES': [
            'CCO', 'c1ccccc1', 'CC(=O)OC1=CC=CC=C1C(=O)O',
            'CC(C)CC1=CC=C(C=C1)C(C)C(=O)O', 'CN1C=NC2=C1C(=O)N(C(=O)N2C)C',
        ],
        'IC50_nM': [100, 1000, 50, 500, 25],  # Regression
        'Toxicity': ['Toxic', 'Non-toxic', 'Non-toxic', 'Toxic', 'Non-toxic'],  # Binary
        'Selectivity': ['High', 'Low', 'Medium', 'High', 'Medium']  # Multiclass
    }
    
    df = pd.DataFrame(data)
    print(f"Input: {len(df)} compounds with mixed targets")
    
    agent = QSARDataCurationAgent(
        df=df,
        smiles_col='SMILES',
        target_cols=['IC50_nM', 'Toxicity', 'Selectivity'],
        units={'IC50_nM': 'nM', 'Toxicity': None, 'Selectivity': None},
        verbose=True
    )
    
    cleaned_df = agent.run()
    
    print(f"\nResult:")
    print(cleaned_df)
    
    stats = agent.get_statistics()
    print(f"\nOverall task: {stats['overall_task']}")
    
    # Check each target type
    configs = agent.get_target_configs()
    for name, config in configs.items():
        print(f"  {name}: {config.task_type.value}")
    
    assert stats['overall_task'] == 'mixed'
    print("\n✓ Mixed tasks test passed!")
    return True


def test_multi_target_classification():
    """Test multi-target classification (multiple label columns)."""
    print("\n" + "="*60)
    print("Test 6: Multi-Target Classification")
    print("="*60)
    
    from automl_qsar.preprocessing import QSARDataCurationAgent
    
    # Create multi-label classification dataset
    data = {
        'SMILES': [
            'CCO', 'c1ccccc1', 'CC(=O)OC1=CC=CC=C1C(=O)O',
            'CC(C)CC1=CC=C(C=C1)C(C)C(=O)O', 'CN1C=NC2=C1C(=O)N(C(=O)N2C)C',
        ],
        'Target_A': ['Active', 'Inactive', 'Active', 'Inactive', 'Active'],
        'Target_B': ['Positive', 'Negative', 'Positive', 'Positive', 'Negative'],
    }
    
    df = pd.DataFrame(data)
    print(f"Input: {len(df)} compounds with 2 classification targets")
    
    agent = QSARDataCurationAgent(
        df=df,
        smiles_col='SMILES',
        target_cols=['Target_A', 'Target_B'],
        units=None,
        verbose=True
    )
    
    cleaned_df = agent.run()
    
    print(f"\nResult:")
    print(cleaned_df)
    
    stats = agent.get_statistics()
    print(f"\nOverall task: {stats['overall_task']}")
    
    assert stats['overall_task'] == 'classification_multilabel'
    print("\n✓ Multi-target classification test passed!")
    return True


def test_with_kinase_dataset():
    """Test with the real kinase IC50 dataset."""
    print("\n" + "="*60)
    print("Test 7: Real Kinase IC50 Dataset")
    print("="*60)
    
    from automl_qsar.preprocessing import QSARDataCurationAgent
    
    # Load kinase dataset
    df = pd.read_csv('kinase_ic50_dataset.csv')
    print(f"Loaded: {len(df)} compounds")
    
    agent = QSARDataCurationAgent(
        df=df,
        smiles_col='SMILES',
        target_cols='IC50_nM',
        units='nM',
        verbose=True
    )
    
    cleaned_df = agent.run()
    
    stats = agent.get_statistics()
    config = agent.get_target_configs()['IC50_nM']
    
    print(f"\nResults:")
    print(f"  Original: {len(df)} compounds")
    print(f"  Cleaned: {len(cleaned_df)} compounds")
    print(f"  Task: {stats['overall_task']}")
    print(f"  Transform: {config.transform}")
    
    if 'mean' in config.statistics:
        print(f"  pIC50 range: {config.statistics['min']:.2f} - {config.statistics['max']:.2f}")
    
    print("\n✓ Kinase dataset test passed!")
    return True


def test_convenience_function():
    """Test the curate_qsar_data convenience function."""
    print("\n" + "="*60)
    print("Test 8: Convenience Function")
    print("="*60)
    
    from automl_qsar.preprocessing import curate_qsar_data
    
    # Simple usage
    data = {
        'SMILES': ['CCO', 'c1ccccc1', 'CCCC', 'CC(C)C'],
        'IC50': [100, 1000, 500, 50]
    }
    df = pd.DataFrame(data)
    
    # Single target
    clean_df, stats = curate_qsar_data(
        df, 'SMILES', 'IC50', units='nM', verbose=False
    )
    print(f"Single target: {len(clean_df)} molecules, task={stats['overall_task']}")
    
    # Multi-target
    data2 = {
        'SMILES': ['CCO', 'c1ccccc1', 'CCCC'],
        'IC50_A': [100, 1000, 500],
        'IC50_B': [200, 500, 250]
    }
    df2 = pd.DataFrame(data2)
    
    clean_df2, stats2 = curate_qsar_data(
        df2, 'SMILES', ['IC50_A', 'IC50_B'], units='nM', verbose=False
    )
    print(f"Multi target: {len(clean_df2)} molecules, task={stats2['overall_task']}")
    
    # Classification
    data3 = {
        'SMILES': ['CCO', 'c1ccccc1', 'CCCC'],
        'Activity': ['Active', 'Inactive', 'Active']
    }
    df3 = pd.DataFrame(data3)
    
    clean_df3, stats3 = curate_qsar_data(
        df3, 'SMILES', 'Activity', units=None, verbose=False
    )
    print(f"Classification: {len(clean_df3)} molecules, task={stats3['overall_task']}")
    
    print("\n✓ Convenience function test passed!")
    return True


if __name__ == "__main__":
    tests = [
        ("Single-target Regression", test_single_target_regression),
        ("Multi-target Regression", test_multi_target_regression),
        ("Binary Classification", test_binary_classification),
        ("Multiclass Classification", test_multiclass_classification),
        ("Mixed Tasks", test_mixed_tasks),
        ("Multi-target Classification", test_multi_target_classification),
        ("Kinase Dataset", test_with_kinase_dataset),
        ("Convenience Function", test_convenience_function),
    ]
    
    results = []
    for name, test_fn in tests:
        try:
            success = test_fn()
            results.append((name, success))
        except Exception as e:
            print(f"\n❌ {name} failed: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    all_passed = True
    for name, success in results:
        status = "✓" if success else "❌"
        print(f"  {status} {name}")
        if not success:
            all_passed = False
    
    print("\n" + "="*60)
    if all_passed:
        print("🎉 All data curation tests passed!")
    else:
        print("⚠️ Some tests failed")
    print("="*60)
