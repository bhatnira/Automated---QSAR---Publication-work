"""
Top-K ensemble that selects and combines the best K models
"""

import numpy as np
from typing import List, Any, Tuple
from .base_ensemble import BaseEnsemble


class TopKEnsemble(BaseEnsemble):
    """
    Top-K ensemble that selects the best K models based on validation performance
    
    Args:
        k: Number of top models to include
        method: Aggregation method ('mean', 'weighted_mean', 'median')
    """
    
    def __init__(self, k: int = 5, method: str = 'weighted_mean'):
        super().__init__(name="TopKEnsemble")
        self.k = k
        self.method = method
        self.model_scores = []
    
    def fit(
        self,
        models: List[Any],
        X: np.ndarray,
        y: np.ndarray,
        validation_scores: List[float] = None
    ):
        """
        Fit ensemble by selecting top-K models
        
        Args:
            models: List of trained models
            X: Features (for validation if scores not provided)
            y: Target (for validation if scores not provided)
            validation_scores: Pre-computed validation scores (lower is better)
        """
        if validation_scores is None:
            # Compute validation scores (RMSE)
            validation_scores = []
            for model in models:
                pred = model.predict(X)
                rmse = np.sqrt(np.mean((y - pred) ** 2))
                validation_scores.append(rmse)
        
        # Select top-K models (lowest scores)
        model_score_pairs = list(zip(models, validation_scores))
        model_score_pairs.sort(key=lambda x: x[1])  # Sort by score (ascending)
        
        # Select top-K
        top_k = min(self.k, len(model_score_pairs))
        top_k_pairs = model_score_pairs[:top_k]
        
        self.models = [pair[0] for pair in top_k_pairs]
        self.model_scores = [pair[1] for pair in top_k_pairs]
        
        # Compute weights based on scores (inverse of error)
        if self.method == 'weighted_mean':
            # Convert scores to weights (higher weight for lower error)
            inv_scores = [1.0 / (score + 1e-10) for score in self.model_scores]
            total = sum(inv_scores)
            self.weights = [w / total for w in inv_scores]
        else:
            self.weights = np.ones(len(self.models)) / len(self.models)
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using top-K models
        
        Args:
            X: Features
            
        Returns:
            Predictions
        """
        if not self.models:
            raise ValueError("No models in ensemble")
        
        # Collect predictions
        all_predictions = []
        for model in self.models:
            pred = model.predict(X)
            all_predictions.append(pred)
        
        all_predictions = np.array(all_predictions)
        
        # Aggregate predictions
        if self.method == 'mean' or self.method == 'weighted_mean':
            predictions = np.average(all_predictions, axis=0, weights=self.weights)
        elif self.method == 'median':
            predictions = np.median(all_predictions, axis=0)
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        return predictions
    
    def get_model_info(self) -> List[Tuple[Any, float, float]]:
        """
        Get information about selected models
        
        Returns:
            List of (model, validation_score, weight) tuples
        """
        return list(zip(self.models, self.model_scores, self.weights))
