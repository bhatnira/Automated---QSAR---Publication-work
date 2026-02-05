"""
SHAP (SHapley Additive exPlanations) interpreter
"""

import numpy as np
from typing import List


class SHAPInterpreter:
    """
    SHAP-based model interpretation
    """
    
    def __init__(self):
        self.name = "SHAP"
        try:
            import shap
            self.shap = shap
            self.shap_available = True
        except ImportError:
            self.shap_available = False
            print("Warning: SHAP package not available")
    
    def explain_model(self, model, X: np.ndarray, feature_names: List[str] = None):
        """
        Generate SHAP explanations for a model
        
        Args:
            model: Trained model
            X: Features
            feature_names: Feature names
            
        Returns:
            SHAP values and explainer
        """
        if not self.shap_available:
            raise RuntimeError("SHAP package is required")
        
        # Create explainer based on model type
        try:
            # Try TreeExplainer for tree-based models
            explainer = self.shap.TreeExplainer(model)
        except Exception:
            try:
                # Try LinearExplainer for linear models
                explainer = self.shap.LinearExplainer(model, X)
            except Exception:
                # Fall back to KernelExplainer (model-agnostic)
                explainer = self.shap.KernelExplainer(model.predict, X[:100])
        
        # Calculate SHAP values
        shap_values = explainer.shap_values(X)
        
        return {
            'shap_values': shap_values,
            'explainer': explainer,
            'feature_names': feature_names
        }
    
    def plot_summary(self, shap_values, X: np.ndarray, feature_names: List[str] = None):
        """Plot SHAP summary"""
        if not self.shap_available:
            raise RuntimeError("SHAP package is required")
        
        self.shap.summary_plot(shap_values, X, feature_names=feature_names, show=False)
    
    def plot_waterfall(self, shap_values, X: np.ndarray, idx: int = 0, feature_names: List[str] = None):
        """Plot SHAP waterfall for a single prediction"""
        if not self.shap_available:
            raise RuntimeError("SHAP package is required")
        
        self.shap.waterfall_plot(
            self.shap.Explanation(
                values=shap_values[idx],
                base_values=shap_values[idx].sum(),
                data=X[idx],
                feature_names=feature_names
            ),
            show=False
        )
