"""
RDKit molecular descriptors featurizer
"""

from typing import List, Union
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, Crippen, MolSurf, GraphDescriptors
from .base import BaseFeaturizer


class RDKitDescriptors(BaseFeaturizer):
    """
    Compute RDKit molecular descriptors
    
    Includes physicochemical properties, Lipinski descriptors, and molecular properties.
    """
    
    def __init__(self):
        super().__init__(name="RDKitDescriptors")
        self._setup_descriptors()
    
    def _setup_descriptors(self):
        """Setup list of descriptor functions to compute"""
        descriptor_candidates = [
            # Molecular properties
            ('MolWt', Descriptors.MolWt),
            ('MolLogP', Descriptors.MolLogP),
            ('TPSA', Descriptors.TPSA),
            ('NumHAcceptors', Descriptors.NumHAcceptors),
            ('NumHDonors', Descriptors.NumHDonors),
            ('NumRotatableBonds', Descriptors.NumRotatableBonds),
            ('NumAromaticRings', Descriptors.NumAromaticRings),
            ('NumAliphaticRings', Descriptors.NumAliphaticRings),
            
            # Lipinski descriptors
            ('NumHeteroatoms', Lipinski.NumHeteroatoms),
            ('NumAromaticHeterocycles', Lipinski.NumAromaticHeterocycles),
            ('NumSaturatedRings', Lipinski.NumSaturatedRings),
            ('RingCount', Lipinski.RingCount),
            
            # Crippen descriptors
            ('MolMR', Crippen.MolMR),
            
            # Surface area
            ('LabuteASA', MolSurf.LabuteASA),
            
            # Graph descriptors
            ('BertzCT', GraphDescriptors.BertzCT),
            ('Chi0v', GraphDescriptors.Chi0v),
            ('Chi1v', GraphDescriptors.Chi1v),
            ('Kappa1', GraphDescriptors.Kappa1),
            ('Kappa2', GraphDescriptors.Kappa2),
            ('Kappa3', GraphDescriptors.Kappa3),
            
            # Additional descriptors
            ('HeavyAtomCount', Descriptors.HeavyAtomCount),
            ('NumValenceElectrons', Descriptors.NumValenceElectrons),
        ]
        
        # Try to add FractionCsp3 if available (not in all RDKit versions)
        if hasattr(Descriptors, 'FractionCsp3'):
            descriptor_candidates.append(('FractionCsp3', Descriptors.FractionCsp3))
        
        self.descriptor_funcs = descriptor_candidates
        self.feature_names = [name for name, _ in self.descriptor_funcs]
    
    def featurize(self, molecules: Union[List[str], List[Chem.Mol]]) -> np.ndarray:
        """
        Compute RDKit descriptors for molecules
        
        Args:
            molecules: List of SMILES strings or RDKit Mol objects
            
        Returns:
            Feature matrix of shape (n_molecules, n_descriptors)
        """
        mol_objects = self._prepare_molecules(molecules)
        
        features = []
        for mol in mol_objects:
            mol_features = []
            for _, func in self.descriptor_funcs:
                try:
                    value = func(mol)
                    # Handle NaN or inf values
                    if np.isnan(value) or np.isinf(value):
                        value = 0.0
                    mol_features.append(value)
                except Exception as e:
                    print(f"Error computing descriptor: {e}")
                    mol_features.append(0.0)
            
            features.append(mol_features)
        
        return np.array(features, dtype=np.float32)
