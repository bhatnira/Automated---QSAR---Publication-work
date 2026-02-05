"""
Data scaling methods
"""

import numpy as np
from sklearn.preprocessing import StandardScaler as SKStandardScaler
from sklearn.preprocessing import MinMaxScaler as SKMinMaxScaler
from sklearn.preprocessing import RobustScaler as SKRobustScaler


class StandardScaler:
    """
    Standardize features by removing mean and scaling to unit variance
    
    Z = (X - μ) / σ
    """
    
    def __init__(self):
        self.scaler = SKStandardScaler()
        self.name = "StandardScaler"
    
    def fit(self, X: np.ndarray, y: np.ndarray = None):
        """Fit scaler to training data"""
        self.scaler.fit(X)
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform data"""
        return self.scaler.transform(X)
    
    def fit_transform(self, X: np.ndarray, y: np.ndarray = None) -> np.ndarray:
        """Fit and transform"""
        return self.scaler.fit_transform(X)
    
    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Inverse transform"""
        return self.scaler.inverse_transform(X)


class MinMaxScaler:
    """
    Scale features to a given range (default [0, 1])
    
    X_scaled = (X - X_min) / (X_max - X_min)
    """
    
    def __init__(self, feature_range: tuple = (0, 1)):
        self.scaler = SKMinMaxScaler(feature_range=feature_range)
        self.name = "MinMaxScaler"
        self.feature_range = feature_range
    
    def fit(self, X: np.ndarray, y: np.ndarray = None):
        """Fit scaler to training data"""
        self.scaler.fit(X)
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform data"""
        return self.scaler.transform(X)
    
    def fit_transform(self, X: np.ndarray, y: np.ndarray = None) -> np.ndarray:
        """Fit and transform"""
        return self.scaler.fit_transform(X)
    
    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Inverse transform"""
        return self.scaler.inverse_transform(X)


class RobustScaler:
    """
    Scale features using statistics that are robust to outliers
    
    Uses median and interquartile range instead of mean and standard deviation
    """
    
    def __init__(self):
        self.scaler = SKRobustScaler()
        self.name = "RobustScaler"
    
    def fit(self, X: np.ndarray, y: np.ndarray = None):
        """Fit scaler to training data"""
        self.scaler.fit(X)
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform data"""
        return self.scaler.transform(X)
    
    def fit_transform(self, X: np.ndarray, y: np.ndarray = None) -> np.ndarray:
        """Fit and transform"""
        return self.scaler.fit_transform(X)
    
    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Inverse transform"""
        return self.scaler.inverse_transform(X)
