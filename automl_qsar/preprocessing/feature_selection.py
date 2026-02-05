"""
Feature selection methods
"""

import numpy as np
from typing import Optional
from sklearn.feature_selection import VarianceThreshold as SKVarianceThreshold
from sklearn.ensemble import RandomForestRegressor


class VarianceThreshold:
    """
    Remove features with low variance
    
    Args:
        threshold: Variance threshold below which features are removed
    """
    
    def __init__(self, threshold: float = 0.0):
        self.selector = SKVarianceThreshold(threshold=threshold)
        self.name = "VarianceThreshold"
        self.threshold = threshold
    
    def fit(self, X: np.ndarray, y: np.ndarray = None):
        """Fit feature selector"""
        self.selector.fit(X)
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform data"""
        return self.selector.transform(X)
    
    def fit_transform(self, X: np.ndarray, y: np.ndarray = None) -> np.ndarray:
        """Fit and transform"""
        return self.selector.fit_transform(X)
    
    def get_support(self) -> np.ndarray:
        """Get mask of selected features"""
        return self.selector.get_support()


class CorrelationSelector:
    """
    Remove highly correlated features
    
    Args:
        threshold: Correlation threshold above which features are removed
    """
    
    def __init__(self, threshold: float = 0.95):
        self.threshold = threshold
        self.name = "CorrelationSelector"
        self.selected_features_ = None
    
    def fit(self, X: np.ndarray, y: np.ndarray = None):
        """Fit feature selector"""
        # Calculate correlation matrix
        corr_matrix = np.corrcoef(X.T)
        
        # Find features to remove
        features_to_remove = set()
        n_features = X.shape[1]
        
        for i in range(n_features):
            if i in features_to_remove:
                continue
            for j in range(i + 1, n_features):
                if j in features_to_remove:
                    continue
                if abs(corr_matrix[i, j]) > self.threshold:
                    features_to_remove.add(j)
        
        # Create mask of selected features
        self.selected_features_ = np.array([
            i not in features_to_remove for i in range(n_features)
        ])
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform data"""
        if self.selected_features_ is None:
            raise ValueError("Selector has not been fitted yet")
        return X[:, self.selected_features_]
    
    def fit_transform(self, X: np.ndarray, y: np.ndarray = None) -> np.ndarray:
        """Fit and transform"""
        self.fit(X, y)
        return self.transform(X)
    
    def get_support(self) -> np.ndarray:
        """Get mask of selected features"""
        return self.selected_features_


class BorutaSelector:
    """
    Boruta feature selection algorithm
    
    Uses Random Forest to identify relevant features
    
    Args:
        n_estimators: Number of trees in Random Forest
        max_iter: Maximum number of iterations
        random_state: Random state for reproducibility
    """
    
    def __init__(self, n_estimators: int = 100, max_iter: int = 100, random_state: int = 42):
        self.n_estimators = n_estimators
        self.max_iter = max_iter
        self.random_state = random_state
        self.name = "Boruta"
        self.selected_features_ = None
        
        try:
            from boruta import BorutaPy
            self.boruta_available = True
        except ImportError:
            self.boruta_available = False
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit feature selector"""
        if not self.boruta_available:
            print("Warning: Boruta package not available. Using Random Forest feature importance instead.")
            return self._fit_rf_importance(X, y)
        
        from boruta import BorutaPy
        
        # Create Random Forest model
        rf = RandomForestRegressor(
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            n_jobs=-1
        )
        
        # Create Boruta selector
        boruta_selector = BorutaPy(
            rf,
            n_estimators='auto',
            max_iter=self.max_iter,
            random_state=self.random_state
        )
        
        # Fit selector
        boruta_selector.fit(X, y)
        self.selected_features_ = boruta_selector.support_
        
        return self
    
    def _fit_rf_importance(self, X: np.ndarray, y: np.ndarray):
        """Fallback: Use Random Forest feature importance"""
        rf = RandomForestRegressor(
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            n_jobs=-1
        )
        rf.fit(X, y)
        
        # Select top 50% of features
        importances = rf.feature_importances_
        threshold = np.median(importances)
        self.selected_features_ = importances >= threshold
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform data"""
        if self.selected_features_ is None:
            raise ValueError("Selector has not been fitted yet")
        return X[:, self.selected_features_]
    
    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Fit and transform"""
        self.fit(X, y)
        return self.transform(X)
    
    def get_support(self) -> np.ndarray:
        """Get mask of selected features"""
        return self.selected_features_
