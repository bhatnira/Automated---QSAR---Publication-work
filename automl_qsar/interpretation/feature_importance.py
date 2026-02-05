"""
Feature importance calculation
"""

import numpy as np
from typing import List, Dict
import matplotlib.pyplot as plt


class FeatureImportance:
    """
    Calculate and visualize feature importance
    """
    
    @staticmethod
    def get_tree_importance(model, feature_names: List[str] = None) -> Dict[str, float]:
        """
        Get feature importance from tree-based models
        
        Args:
            model: Trained tree-based model
            feature_names: Names of features
            
        Returns:
            Dictionary mapping feature names to importance scores
        """
        try:
            importances = model.get_feature_importances()
        except AttributeError:
            try:
                importances = model.feature_importances_
            except AttributeError:
                raise ValueError("Model does not support feature importance")
        
        if feature_names is None:
            feature_names = [f"Feature_{i}" for i in range(len(importances))]
        
        importance_dict = dict(zip(feature_names, importances))
        
        # Sort by importance
        importance_dict = dict(sorted(
            importance_dict.items(),
            key=lambda x: x[1],
            reverse=True
        ))
        
        return importance_dict
    
    @staticmethod
    def get_linear_importance(model, feature_names: List[str] = None) -> Dict[str, float]:
        """
        Get feature importance from linear models (based on coefficients)
        
        Args:
            model: Trained linear model
            feature_names: Names of features
            
        Returns:
            Dictionary mapping feature names to importance scores
        """
        try:
            coefficients = model.model.coef_
        except AttributeError:
            raise ValueError("Model does not have coefficients")
        
        # Use absolute values of coefficients as importance
        importances = np.abs(coefficients)
        
        if feature_names is None:
            feature_names = [f"Feature_{i}" for i in range(len(importances))]
        
        importance_dict = dict(zip(feature_names, importances))
        
        # Sort by importance
        importance_dict = dict(sorted(
            importance_dict.items(),
            key=lambda x: x[1],
            reverse=True
        ))
        
        return importance_dict
    
    @staticmethod
    def permutation_importance(
        model,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str] = None,
        n_repeats: int = 10,
        random_state: int = 42
    ) -> Dict[str, float]:
        """
        Calculate permutation-based feature importance
        
        Args:
            model: Trained model
            X: Features
            y: Target
            feature_names: Names of features
            n_repeats: Number of permutation repeats
            random_state: Random state
            
        Returns:
            Dictionary mapping feature names to importance scores
        """
        from sklearn.inspection import permutation_importance as perm_imp
        
        result = perm_imp(
            model,
            X,
            y,
            n_repeats=n_repeats,
            random_state=random_state,
            n_jobs=-1
        )
        
        importances = result.importances_mean
        
        if feature_names is None:
            feature_names = [f"Feature_{i}" for i in range(len(importances))]
        
        importance_dict = dict(zip(feature_names, importances))
        
        # Sort by importance
        importance_dict = dict(sorted(
            importance_dict.items(),
            key=lambda x: x[1],
            reverse=True
        ))
        
        return importance_dict
    
    @staticmethod
    def plot_importance(
        importance_dict: Dict[str, float],
        top_n: int = 20,
        title: str = "Feature Importance",
        figsize: tuple = (10, 8)
    ):
        """
        Plot feature importance
        
        Args:
            importance_dict: Dictionary of feature importances
            top_n: Number of top features to plot
            title: Plot title
            figsize: Figure size
        """
        # Get top N features
        sorted_features = list(importance_dict.items())[:top_n]
        features = [f[0] for f in sorted_features]
        importances = [f[1] for f in sorted_features]
        
        # Create plot
        plt.figure(figsize=figsize)
        plt.barh(range(len(features)), importances)
        plt.yticks(range(len(features)), features)
        plt.xlabel('Importance')
        plt.title(title)
        plt.gca().invert_yaxis()
        plt.tight_layout()
        
        return plt.gcf()
