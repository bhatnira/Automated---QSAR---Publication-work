"""
Base class for molecular featurizers
"""

from abc import ABC, abstractmethod
from typing import List, Union, Optional
import numpy as np
from rdkit import Chem


class BaseFeaturizer(ABC):
    """Abstract base class for all featurizers"""
    
    def __init__(self, name: str = "BaseFeaturizer"):
        self.name = name
        self.feature_names = []
    
    @abstractmethod
    def featurize(self, molecules: Union[List[str], List[Chem.Mol]]) -> np.ndarray:
        """
        Convert molecules to feature vectors
        
        Args:
            molecules: List of SMILES strings or RDKit Mol objects
            
        Returns:
            Feature matrix of shape (n_molecules, n_features)
        """
        pass
    
    def _smiles_to_mol(self, smiles: str) -> Optional[Chem.Mol]:
        """Convert SMILES string to RDKit Mol object"""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            return mol
        except Exception as e:
            print(f"Error converting SMILES {smiles}: {e}")
            return None
    
    def _prepare_molecules(self, molecules: Union[List[str], List[Chem.Mol]]) -> List[Chem.Mol]:
        """Prepare molecules for featurization"""
        if molecules is None or len(molecules) == 0:
            raise ValueError("Empty molecule list provided")
        
        # Check if first element is string (SMILES) or Mol object
        if isinstance(molecules[0], str):
            mol_objects = [self._smiles_to_mol(smi) for smi in molecules]
        else:
            mol_objects = molecules
        
        # Filter out None values
        mol_objects = [mol for mol in mol_objects if mol is not None]
        
        if not mol_objects:
            raise ValueError("No valid molecules after conversion")
        
        return mol_objects
    
    def get_feature_names(self) -> List[str]:
        """Get names of features"""
        return self.feature_names
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"
