"""
Evaluation metrics for QSAR models
"""

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


class Metrics:
    """
    Collection of QSAR evaluation metrics
    """
    
    @staticmethod
    def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Root Mean Squared Error"""
        return np.sqrt(mean_squared_error(y_true, y_pred))
    
    @staticmethod
    def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Mean Absolute Error"""
        return mean_absolute_error(y_true, y_pred)
    
    @staticmethod
    def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """R² Score (Coefficient of Determination)"""
        return r2_score(y_true, y_pred)
    
    @staticmethod
    def q2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Q² (Cross-validated R²)
        
        Q² = 1 - PRESS/TSS
        where PRESS = sum of squared prediction errors
              TSS = total sum of squares
        """
        press = np.sum((y_true - y_pred) ** 2)
        tss = np.sum((y_true - np.mean(y_true)) ** 2)
        
        if tss == 0:
            return 0.0
        
        return 1 - (press / tss)
    
    @staticmethod
    def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Mean Squared Error"""
        return mean_squared_error(y_true, y_pred)
    
    @staticmethod
    def concordance_correlation_coefficient(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Concordance Correlation Coefficient (CCC)
        
        Measures agreement between predicted and observed values
        """
        mean_true = np.mean(y_true)
        mean_pred = np.mean(y_pred)
        var_true = np.var(y_true)
        var_pred = np.var(y_pred)
        covariance = np.mean((y_true - mean_true) * (y_pred - mean_pred))
        
        denominator = var_true + var_pred + (mean_true - mean_pred) ** 2
        
        if denominator == 0:
            return 0.0
        
        return (2 * covariance) / denominator
    
    @staticmethod
    def calculate_all(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        """Calculate all metrics"""
        return {
            'rmse': Metrics.rmse(y_true, y_pred),
            'mae': Metrics.mae(y_true, y_pred),
            'r2': Metrics.r2(y_true, y_pred),
            'q2': Metrics.q2(y_true, y_pred),
            'mse': Metrics.mse(y_true, y_pred),
            'ccc': Metrics.concordance_correlation_coefficient(y_true, y_pred)
        }
