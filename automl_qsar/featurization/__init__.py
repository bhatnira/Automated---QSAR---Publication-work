"""
Molecular Featurization Module

Provides various methods for converting molecular structures into numerical features.
"""

from .base import BaseFeaturizer
from .rdkit_descriptors import RDKitDescriptors
from .fingerprints import ECFPFingerprints, MACCSFingerprints

try:
    from .pharmacophore import PharmacophoreFeatures
    PHARMACOPHORE_AVAILABLE = True
except ImportError:
    PharmacophoreFeatures = None
    PHARMACOPHORE_AVAILABLE = False

from .embeddings import MolecularEmbeddings
from .graph_features import GraphFeaturizer

__all__ = [
    "BaseFeaturizer",
    "RDKitDescriptors",
    "ECFPFingerprints",
    "MACCSFingerprints",
    "PharmacophoreFeatures",
    "MolecularEmbeddings",
    "GraphFeaturizer",
]
