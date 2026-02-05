"""
Molecular fingerprint featurizers
"""

from typing import List, Union
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys
from .base import BaseFeaturizer


class ECFPFingerprints(BaseFeaturizer):
    """
    Extended Connectivity Fingerprints (ECFP) / Morgan Fingerprints
    
    Args:
        radius: Radius for ECFP (ECFP4 = radius 2, ECFP6 = radius 3)
        n_bits: Number of bits in the fingerprint
        use_features: Use feature-based invariants instead of atom types
    """
    
    def __init__(self, radius: int = 2, n_bits: int = 2048, use_features: bool = False):
        super().__init__(name=f"ECFP{radius*2}")
        self.radius = radius
        self.n_bits = n_bits
        self.use_features = use_features
        self.feature_names = [f"ECFP_bit_{i}" for i in range(n_bits)]
    
    def featurize(self, molecules: Union[List[str], List[Chem.Mol]]) -> np.ndarray:
        """
        Generate ECFP fingerprints
        
        Args:
            molecules: List of SMILES strings or RDKit Mol objects
            
        Returns:
            Binary fingerprint matrix of shape (n_molecules, n_bits)
        """
        mol_objects = self._prepare_molecules(molecules)
        
        features = []
        for mol in mol_objects:
            try:
                fp = AllChem.GetMorganFingerprintAsBitVect(
                    mol,
                    radius=self.radius,
                    nBits=self.n_bits,
                    useFeatures=self.use_features
                )
                arr = np.zeros((self.n_bits,), dtype=np.int8)
                AllChem.DataStructs.ConvertToNumpyArray(fp, arr)
                features.append(arr)
            except Exception as e:
                print(f"Error generating ECFP: {e}")
                features.append(np.zeros(self.n_bits, dtype=np.int8))
        
        return np.array(features, dtype=np.int8)


class MACCSFingerprints(BaseFeaturizer):
    """
    MACCS Keys - 166 bit structural key descriptors
    """
    
    def __init__(self):
        super().__init__(name="MACCS")
        self.n_bits = 166
        self.feature_names = [f"MACCS_key_{i}" for i in range(self.n_bits)]
    
    def featurize(self, molecules: Union[List[str], List[Chem.Mol]]) -> np.ndarray:
        """
        Generate MACCS fingerprints
        
        Args:
            molecules: List of SMILES strings or RDKit Mol objects
            
        Returns:
            Binary fingerprint matrix of shape (n_molecules, 166)
        """
        mol_objects = self._prepare_molecules(molecules)
        
        features = []
        for mol in mol_objects:
            try:
                fp = MACCSkeys.GenMACCSKeys(mol)
                arr = np.zeros((self.n_bits,), dtype=np.int8)
                AllChem.DataStructs.ConvertToNumpyArray(fp, arr)
                features.append(arr)
            except Exception as e:
                print(f"Error generating MACCS: {e}")
                features.append(np.zeros(self.n_bits, dtype=np.int8))
        
        return np.array(features, dtype=np.int8)


class AtomPairFingerprints(BaseFeaturizer):
    """
    Atom Pair Fingerprints
    
    Args:
        n_bits: Number of bits in the fingerprint
    """
    
    def __init__(self, n_bits: int = 2048):
        super().__init__(name="AtomPair")
        self.n_bits = n_bits
        self.feature_names = [f"AtomPair_bit_{i}" for i in range(n_bits)]
    
    def featurize(self, molecules: Union[List[str], List[Chem.Mol]]) -> np.ndarray:
        """
        Generate Atom Pair fingerprints
        
        Args:
            molecules: List of SMILES strings or RDKit Mol objects
            
        Returns:
            Binary fingerprint matrix of shape (n_molecules, n_bits)
        """
        mol_objects = self._prepare_molecules(molecules)
        
        features = []
        for mol in mol_objects:
            try:
                fp = AllChem.GetHashedAtomPairFingerprintAsBitVect(mol, nBits=self.n_bits)
                arr = np.zeros((self.n_bits,), dtype=np.int8)
                AllChem.DataStructs.ConvertToNumpyArray(fp, arr)
                features.append(arr)
            except Exception as e:
                print(f"Error generating Atom Pair FP: {e}")
                features.append(np.zeros(self.n_bits, dtype=np.int8))
        
        return np.array(features, dtype=np.int8)
