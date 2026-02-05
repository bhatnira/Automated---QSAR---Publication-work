"""
Pharmacophore-based features
"""

from typing import List, Union
import numpy as np
from rdkit import Chem
try:
    from rdkit.Chem import Pharmacophore, ChemicalFeatures
    from rdkit import RDConfig
    PHARMACOPHORE_AVAILABLE = True
except ImportError:
    # Fallback for older RDKit versions
    try:
        from rdkit.Chem.Pharm2D import Pharmacophore
        from rdkit.Chem import ChemicalFeatures
        from rdkit import RDConfig
        PHARMACOPHORE_AVAILABLE = True
    except ImportError:
        PHARMACOPHORE_AVAILABLE = False
import os
from .base import BaseFeaturizer


class PharmacophoreFeatures(BaseFeaturizer):
    """
    Pharmacophore feature fingerprints
    
    Identifies pharmacophoric features such as:
    - Hydrogen bond donors/acceptors
    - Aromatic rings
    - Hydrophobic regions
    - Positive/negative ionizable groups
    """
    
    def __init__(self):
        super().__init__(name="Pharmacophore")
        if not PHARMACOPHORE_AVAILABLE:
            raise ImportError(
                "Pharmacophore features require full RDKit installation. "
                "Install with: conda install -c conda-forge rdkit"
            )
        self._setup_factory()
        self._setup_feature_names()
    
    def _setup_factory(self):
        """Setup RDKit pharmacophore feature factory"""
        fdefName = os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
        self.factory = ChemicalFeatures.BuildFeatureFactory(fdefName)
        
        # Define pharmacophore feature types
        self.feature_types = [
            'Donor',
            'Acceptor',
            'NegIonizable',
            'PosIonizable',
            'Aromatic',
            'Hydrophobe',
            'LumpedHydrophobe'
        ]
    
    def _setup_feature_names(self):
        """Setup feature names"""
        self.feature_names = []
        for ftype in self.feature_types:
            self.feature_names.append(f"Pharm_Count_{ftype}")
        
        # Add distance-based features between feature types
        for i, ftype1 in enumerate(self.feature_types):
            for ftype2 in self.feature_types[i+1:]:
                self.feature_names.append(f"Pharm_MinDist_{ftype1}_{ftype2}")
    
    def _extract_features(self, mol: Chem.Mol) -> np.ndarray:
        """Extract pharmacophore features from a molecule"""
        rawFeats = self.factory.GetFeaturesForMol(mol)
        
        # Count features by type
        feature_counts = {ftype: 0 for ftype in self.feature_types}
        feature_positions = {ftype: [] for ftype in self.feature_types}
        
        for feat in rawFeats:
            ftype = feat.GetFamily()
            if ftype in feature_counts:
                feature_counts[ftype] += 1
                feature_positions[ftype].append(feat.GetPos())
        
        # Build feature vector
        features = []
        
        # Add counts
        for ftype in self.feature_types:
            features.append(feature_counts[ftype])
        
        # Add minimum distances between feature types
        for i, ftype1 in enumerate(self.feature_types):
            for ftype2 in self.feature_types[i+1:]:
                min_dist = self._min_distance(
                    feature_positions[ftype1],
                    feature_positions[ftype2]
                )
                features.append(min_dist)
        
        return np.array(features, dtype=np.float32)
    
    def _min_distance(self, positions1: List, positions2: List) -> float:
        """Calculate minimum distance between two sets of 3D positions"""
        if not positions1 or not positions2:
            return 0.0
        
        min_dist = float('inf')
        for pos1 in positions1:
            for pos2 in positions2:
                dist = pos1.Distance(pos2)
                if dist < min_dist:
                    min_dist = dist
        
        return min_dist if min_dist != float('inf') else 0.0
    
    def featurize(self, molecules: Union[List[str], List[Chem.Mol]]) -> np.ndarray:
        """
        Generate pharmacophore features
        
        Args:
            molecules: List of SMILES strings or RDKit Mol objects
            
        Returns:
            Feature matrix of shape (n_molecules, n_features)
        """
        mol_objects = self._prepare_molecules(molecules)
        
        features = []
        for mol in mol_objects:
            try:
                # Add 3D coordinates if not present
                if mol.GetNumConformers() == 0:
                    from rdkit.Chem import AllChem
                    AllChem.EmbedMolecule(mol, randomSeed=42)
                    AllChem.MMFFOptimizeMolecule(mol)
                
                mol_features = self._extract_features(mol)
                features.append(mol_features)
            except Exception as e:
                print(f"Error extracting pharmacophore features: {e}")
                features.append(np.zeros(len(self.feature_names), dtype=np.float32))
        
        return np.array(features, dtype=np.float32)
