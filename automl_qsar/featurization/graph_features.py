"""
Graph-based molecular features
"""

from __future__ import annotations
from typing import List, Union, Tuple, TYPE_CHECKING, Any
import numpy as np
from rdkit import Chem
from .base import BaseFeaturizer

try:
    import torch
    from torch_geometric.data import Data
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    Data = None  # Placeholder for type hints


class GraphFeaturizer(BaseFeaturizer):
    """
    Convert molecules to graph representations for GNN models
    
    Creates node features (atoms) and edge features (bonds) suitable for
    Graph Neural Networks.
    """
    
    def __init__(self):
        super().__init__(name="Graph")
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch and PyTorch Geometric are required for graph features")
        
        self.atom_features_dim = 44  # Size of atom feature vector
        self.bond_features_dim = 6   # Size of bond feature vector
    
    def _atom_features(self, atom: Chem.Atom) -> List[float]:
        """
        Generate feature vector for an atom
        
        Features include:
        - Atom type (one-hot)
        - Degree
        - Formal charge
        - Hybridization
        - Aromaticity
        - etc.
        """
        features = []
        
        # Atom type (one-hot encoding for common elements)
        atom_types = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'I', 'Unknown']
        atom_symbol = atom.GetSymbol()
        features.extend([1 if atom_symbol == t else 0 for t in atom_types[:-1]])
        features.append(1 if atom_symbol not in atom_types[:-1] else 0)
        
        # Degree (one-hot, 0-5+)
        degree = atom.GetDegree()
        features.extend([1 if degree == i else 0 for i in range(6)])
        features.append(1 if degree > 5 else 0)
        
        # Formal charge (one-hot, -2 to +2)
        charge = atom.GetFormalCharge()
        features.extend([1 if charge == i else 0 for i in range(-2, 3)])
        
        # Hybridization (one-hot)
        hybridizations = [
            Chem.HybridizationType.SP,
            Chem.HybridizationType.SP2,
            Chem.HybridizationType.SP3,
            Chem.HybridizationType.SP3D,
            Chem.HybridizationType.SP3D2
        ]
        hyb = atom.GetHybridization()
        features.extend([1 if hyb == h else 0 for h in hybridizations])
        features.append(1 if hyb not in hybridizations else 0)
        
        # Additional features
        features.append(1 if atom.GetIsAromatic() else 0)
        features.append(atom.GetNumRadicalElectrons())
        features.append(atom.GetTotalNumHs())
        features.append(1 if atom.IsInRing() else 0)
        
        return features
    
    def _bond_features(self, bond: Chem.Bond) -> List[float]:
        """Generate feature vector for a bond"""
        features = []
        
        # Bond type (one-hot)
        bond_types = [
            Chem.BondType.SINGLE,
            Chem.BondType.DOUBLE,
            Chem.BondType.TRIPLE,
            Chem.BondType.AROMATIC
        ]
        bt = bond.GetBondType()
        features.extend([1 if bt == t else 0 for t in bond_types])
        
        # Additional features
        features.append(1 if bond.GetIsConjugated() else 0)
        features.append(1 if bond.IsInRing() else 0)
        
        return features
    
    def mol_to_graph(self, mol: Chem.Mol) -> Any:
        """
        Convert RDKit molecule to PyTorch Geometric Data object
        
        Args:
            mol: RDKit molecule
            
        Returns:
            PyTorch Geometric Data object with node and edge features
        """
        # Node features
        node_features = []
        for atom in mol.GetAtoms():
            node_features.append(self._atom_features(atom))
        x = torch.tensor(node_features, dtype=torch.float)
        
        # Edge indices and features
        edge_indices = []
        edge_features = []
        
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            
            # Add both directions for undirected graph
            edge_indices.append([i, j])
            edge_indices.append([j, i])
            
            bond_feat = self._bond_features(bond)
            edge_features.append(bond_feat)
            edge_features.append(bond_feat)
        
        if edge_indices:
            edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_features, dtype=torch.float)
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_attr = torch.zeros((0, self.bond_features_dim), dtype=torch.float)
        
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    
    def featurize(self, molecules: Union[List[str], List[Chem.Mol]]) -> List[Any]:
        """
        Convert molecules to graph representations
        
        Args:
            molecules: List of SMILES strings or RDKit Mol objects
            
        Returns:
            List of PyTorch Geometric Data objects
        """
        mol_objects = self._prepare_molecules(molecules)
        
        graphs = []
        for mol in mol_objects:
            try:
                graph = self.mol_to_graph(mol)
                graphs.append(graph)
            except Exception as e:
                print(f"Error converting molecule to graph: {e}")
                # Create empty graph as placeholder
                x = torch.zeros((1, self.atom_features_dim), dtype=torch.float)
                edge_index = torch.zeros((2, 0), dtype=torch.long)
                edge_attr = torch.zeros((0, self.bond_features_dim), dtype=torch.float)
                graphs.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr))
        
        return graphs
