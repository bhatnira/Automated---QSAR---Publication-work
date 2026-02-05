"""
Dimensionality reduction methods
"""

import numpy as np
from sklearn.decomposition import PCA


class PCAReducer:
    """
    Principal Component Analysis for dimensionality reduction
    
    Args:
        n_components: Number of components to keep
        variance_threshold: Alternatively, keep components explaining this much variance
    """
    
    def __init__(self, n_components: int = None, variance_threshold: float = None):
        if n_components is not None:
            self.pca = PCA(n_components=n_components)
        elif variance_threshold is not None:
            self.pca = PCA(n_components=variance_threshold)
        else:
            self.pca = PCA(n_components=0.95)  # Default: keep 95% of variance
        
        self.name = "PCA"
        self.n_components = n_components
        self.variance_threshold = variance_threshold
    
    def fit(self, X: np.ndarray, y: np.ndarray = None):
        """Fit PCA"""
        self.pca.fit(X)
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform data"""
        return self.pca.transform(X)
    
    def fit_transform(self, X: np.ndarray, y: np.ndarray = None) -> np.ndarray:
        """Fit and transform"""
        return self.pca.fit_transform(X)
    
    def get_explained_variance_ratio(self) -> np.ndarray:
        """Get explained variance ratio for each component"""
        return self.pca.explained_variance_ratio_
    
    def get_n_components(self) -> int:
        """Get number of components"""
        return self.pca.n_components_


class UMAPReducer:
    """
    UMAP (Uniform Manifold Approximation and Projection) for dimensionality reduction
    
    Args:
        n_components: Number of components to keep
        n_neighbors: Number of neighbors for UMAP
        min_dist: Minimum distance for UMAP
    """
    
    def __init__(self, n_components: int = 50, n_neighbors: int = 15, min_dist: float = 0.1):
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.name = "UMAP"
        
        try:
            import umap
            self.reducer = umap.UMAP(
                n_components=n_components,
                n_neighbors=n_neighbors,
                min_dist=min_dist,
                random_state=42
            )
            self.umap_available = True
        except ImportError:
            print("Warning: UMAP not available. Will use PCA instead.")
            self.reducer = PCA(n_components=n_components)
            self.umap_available = False
    
    def fit(self, X: np.ndarray, y: np.ndarray = None):
        """Fit UMAP"""
        self.reducer.fit(X, y)
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform data"""
        return self.reducer.transform(X)
    
    def fit_transform(self, X: np.ndarray, y: np.ndarray = None) -> np.ndarray:
        """Fit and transform"""
        return self.reducer.fit_transform(X, y)
