"""
LIME (Local Interpretable Model-agnostic Explanations) interpreter
"""

import numpy as np
from typing import List


class LIMEInterpreter:
    """
    LIME-based model interpretation
    """
    
    def __init__(self):
        self.name = "LIME"
        try:
            from lime import lime_tabular
            self.lime_tabular = lime_tabular
            self.lime_available = True
        except ImportError:
            self.lime_available = False
            print("Warning: LIME package not available")
    
    def explain_instance(
        self,
        model,
        X_train: np.ndarray,
        X_test: np.ndarray,
        instance_idx: int = 0,
        feature_names: List[str] = None,
        num_features: int = 10
    ):
        """
        Generate LIME explanation for a single instance
        
        Args:
            model: Trained model
            X_train: Training data (for fitting explainer)
            X_test: Test data
            instance_idx: Index of instance to explain
            feature_names: Feature names
            num_features: Number of top features to show
            
        Returns:
            LIME explanation
        """
        if not self.lime_available:
            raise RuntimeError("LIME package is required")
        
        # Create explainer
        explainer = self.lime_tabular.LimeTabularExplainer(
            X_train,
            feature_names=feature_names,
            mode='regression',
            verbose=False
        )
        
        # Explain instance
        explanation = explainer.explain_instance(
            X_test[instance_idx],
            model.predict,
            num_features=num_features
        )
        
        return explanation
    
    def plot_explanation(self, explanation):
        """Plot LIME explanation"""
        if not self.lime_available:
            raise RuntimeError("LIME package is required")
        
        explanation.as_pyplot_figure()
