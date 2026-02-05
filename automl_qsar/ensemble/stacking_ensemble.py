"""
Stacking ensemble methods
"""

import numpy as np
from typing import List, Any
from sklearn.linear_model import Ridge
from .base_ensemble import BaseEnsemble


class StackingEnsemble(BaseEnsemble):
    """
    Stacking ensemble that trains a meta-model on base model predictions
    
    Args:
        meta_model: Model to use for stacking (default: Ridge regression)
    """
    
    def __init__(self, meta_model: Any = None):
        super().__init__(name="StackingEnsemble")
        self.meta_model = meta_model if meta_model is not None else Ridge(alpha=1.0)
    
    def fit(self, models: List[Any], X: np.ndarray, y: np.ndarray):
        """
        Fit stacking ensemble
        
        Args:
            models: List of trained base models
            X: Features
            y: Target values
        """
        self.models = models
        
        # Generate predictions from base models
        base_predictions = []
        for model in self.models:
            pred = model.predict(X)
            base_predictions.append(pred)
        
        # Stack predictions horizontally
        X_meta = np.column_stack(base_predictions)
        
        # Train meta-model
        self.meta_model.fit(X_meta, y)
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make stacked predictions
        
        Args:
            X: Features
            
        Returns:
            Predictions
        """
        if not self.models:
            raise ValueError("No models in ensemble")
        
        # Generate predictions from base models
        base_predictions = []
        for model in self.models:
            pred = model.predict(X)
            base_predictions.append(pred)
        
        # Stack predictions
        X_meta = np.column_stack(base_predictions)
        
        # Meta-model prediction
        predictions = self.meta_model.predict(X_meta)
        
        return predictions
