"""
Voting ensemble methods
"""

import numpy as np
from typing import List, Any
from .base_ensemble import BaseEnsemble


class VotingEnsemble(BaseEnsemble):
    """
    Voting ensemble that averages predictions from multiple models
    
    Args:
        weights: Optional weights for each model
        method: 'mean' or 'median' averaging
    """
    
    def __init__(self, weights: List[float] = None, method: str = 'mean'):
        super().__init__(name="VotingEnsemble")
        self.weights = weights
        self.method = method
    
    def fit(self, models: List[Any], X: np.ndarray = None, y: np.ndarray = None):
        """
        Fit ensemble (just stores models)
        
        Args:
            models: List of trained models
            X: Not used (models already trained)
            y: Not used (models already trained)
        """
        self.models = models
        
        # Set equal weights if not provided
        if self.weights is None:
            self.weights = np.ones(len(models)) / len(models)
        else:
            # Normalize weights
            self.weights = np.array(self.weights)
            self.weights = self.weights / np.sum(self.weights)
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make weighted average predictions
        
        Args:
            X: Features
            
        Returns:
            Predictions
        """
        if not self.models:
            raise ValueError("No models in ensemble")
        
        # Collect predictions from all models
        all_predictions = []
        for model in self.models:
            pred = model.predict(X)
            all_predictions.append(pred)
        
        all_predictions = np.array(all_predictions)
        
        # Weighted average
        if self.method == 'mean':
            predictions = np.average(all_predictions, axis=0, weights=self.weights)
        elif self.method == 'median':
            predictions = np.median(all_predictions, axis=0)
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        return predictions
