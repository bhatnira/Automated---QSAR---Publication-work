"""
Base class for ensemble methods
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import List, Any


class BaseEnsemble(ABC):
    """Abstract base class for ensemble methods"""
    
    def __init__(self, name: str = "BaseEnsemble"):
        self.name = name
        self.models = []
        self.weights = None
    
    @abstractmethod
    def fit(self, models: List[Any], X: np.ndarray, y: np.ndarray):
        """Fit ensemble"""
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        pass
    
    def predict_with_uncertainty(self, X: np.ndarray) -> tuple:
        """
        Make predictions with uncertainty estimates
        
        Returns:
            predictions, uncertainties
        """
        all_predictions = []
        
        for model in self.models:
            pred = model.predict(X)
            all_predictions.append(pred)
        
        all_predictions = np.array(all_predictions)
        
        # Mean prediction
        predictions = np.mean(all_predictions, axis=0)
        
        # Uncertainty as standard deviation
        uncertainties = np.std(all_predictions, axis=0)
        
        return predictions, uncertainties
