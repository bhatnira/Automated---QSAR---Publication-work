"""
Cross-validation strategies
"""

import numpy as np
from typing import List, Tuple, Callable
from sklearn.model_selection import KFold, LeaveOneOut


class KFoldCV:
    """
    K-Fold Cross-Validation
    
    Args:
        n_splits: Number of folds
        shuffle: Whether to shuffle data
        random_state: Random state for reproducibility
    """
    
    def __init__(self, n_splits: int = 5, shuffle: bool = True, random_state: int = 42):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state
        self.name = "KFoldCV"
    
    def split(self, X: np.ndarray, y: np.ndarray = None):
        """Generate train/test splits"""
        kfold = KFold(
            n_splits=self.n_splits,
            shuffle=self.shuffle,
            random_state=self.random_state
        )
        return kfold.split(X, y)
    
    def evaluate(
        self,
        model_func: Callable,
        X: np.ndarray,
        y: np.ndarray,
        metric_func: Callable
    ) -> dict:
        """
        Evaluate model using K-Fold CV
        
        Args:
            model_func: Function that returns fitted model
            X: Features
            y: Target
            metric_func: Function to compute metric
            
        Returns:
            Dictionary with mean and std of metrics
        """
        scores = []
        predictions = []
        
        for train_idx, test_idx in self.split(X, y):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Train model
            model = model_func(X_train, y_train)
            
            # Predict
            y_pred = model.predict(X_test)
            
            # Calculate metric
            score = metric_func(y_test, y_pred)
            scores.append(score)
            predictions.extend(list(zip(test_idx, y_test, y_pred)))
        
        return {
            'mean': np.mean(scores),
            'std': np.std(scores),
            'scores': scores,
            'predictions': predictions
        }


class LOSOCV:
    """
    Leave-One-Series-Out Cross-Validation
    
    For time series or grouped data where entire series should be held out
    
    Args:
        groups: Array of group labels
    """
    
    def __init__(self, groups: np.ndarray = None):
        self.groups = groups
        self.name = "LOSOCV"
    
    def split(self, X: np.ndarray, y: np.ndarray = None):
        """Generate train/test splits"""
        if self.groups is None:
            # Fall back to LeaveOneOut if no groups provided
            loo = LeaveOneOut()
            return loo.split(X)
        
        unique_groups = np.unique(self.groups)
        
        for group in unique_groups:
            test_idx = np.where(self.groups == group)[0]
            train_idx = np.where(self.groups != group)[0]
            yield train_idx, test_idx
    
    def evaluate(
        self,
        model_func: Callable,
        X: np.ndarray,
        y: np.ndarray,
        metric_func: Callable
    ) -> dict:
        """Evaluate model using LOSO CV"""
        scores = []
        predictions = []
        
        for train_idx, test_idx in self.split(X, y):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Train model
            model = model_func(X_train, y_train)
            
            # Predict
            y_pred = model.predict(X_test)
            
            # Calculate metric
            score = metric_func(y_test, y_pred)
            scores.append(score)
            predictions.extend(list(zip(test_idx, y_test, y_pred)))
        
        return {
            'mean': np.mean(scores),
            'std': np.std(scores),
            'scores': scores,
            'predictions': predictions
        }


class NestedCV:
    """
    Nested Cross-Validation for unbiased model evaluation
    
    Outer loop for evaluation, inner loop for hyperparameter tuning
    
    Args:
        outer_cv: Outer CV strategy
        inner_cv: Inner CV strategy
    """
    
    def __init__(self, outer_cv: KFoldCV = None, inner_cv: KFoldCV = None):
        self.outer_cv = outer_cv if outer_cv is not None else KFoldCV(n_splits=5)
        self.inner_cv = inner_cv if inner_cv is not None else KFoldCV(n_splits=3)
        self.name = "NestedCV"
    
    def evaluate(
        self,
        model_func: Callable,
        optimizer_func: Callable,
        X: np.ndarray,
        y: np.ndarray,
        metric_func: Callable
    ) -> dict:
        """
        Evaluate model using Nested CV
        
        Args:
            model_func: Function that returns model with given params
            optimizer_func: Function for hyperparameter optimization
            X: Features
            y: Target
            metric_func: Function to compute metric
            
        Returns:
            Dictionary with nested CV results
        """
        outer_scores = []
        best_params_list = []
        predictions = []
        
        for outer_train_idx, outer_test_idx in self.outer_cv.split(X, y):
            X_outer_train = X[outer_train_idx]
            y_outer_train = y[outer_train_idx]
            X_outer_test = X[outer_test_idx]
            y_outer_test = y[outer_test_idx]
            
            # Inner loop: hyperparameter optimization
            best_params = optimizer_func(X_outer_train, y_outer_train, self.inner_cv)
            best_params_list.append(best_params)
            
            # Train model with best params on outer training set
            model = model_func(best_params)
            model.fit(X_outer_train, y_outer_train)
            
            # Evaluate on outer test set
            y_pred = model.predict(X_outer_test)
            score = metric_func(y_outer_test, y_pred)
            outer_scores.append(score)
            
            predictions.extend(list(zip(outer_test_idx, y_outer_test, y_pred)))
        
        return {
            'mean': np.mean(outer_scores),
            'std': np.std(outer_scores),
            'scores': outer_scores,
            'best_params': best_params_list,
            'predictions': predictions
        }
