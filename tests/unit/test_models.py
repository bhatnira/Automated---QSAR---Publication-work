"""
Unit Tests for Models Module
"""

import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class TestLinearModels:
    """Tests for linear regression models."""
    
    @pytest.fixture
    def sample_data(self):
        np.random.seed(42)
        X = np.random.randn(100, 10)
        y = X[:, 0] * 2 + X[:, 1] * 0.5 + np.random.randn(100) * 0.1
        return X, y
    
    def test_ridge_regression(self, sample_data):
        """Test Ridge regression model."""
        from automl_qsar.models import Ridge
        
        X, y = sample_data
        model = Ridge()
        model.fit(X, y)
        
        predictions = model.predict(X)
        assert len(predictions) == len(y)
    
    def test_lasso_regression(self, sample_data):
        """Test Lasso regression model."""
        from automl_qsar.models import Lasso
        
        X, y = sample_data
        model = Lasso()
        model.fit(X, y)
        
        predictions = model.predict(X)
        assert len(predictions) == len(y)
    
    def test_elasticnet_regression(self, sample_data):
        """Test ElasticNet regression model."""
        from automl_qsar.models import ElasticNet
        
        X, y = sample_data
        model = ElasticNet()
        model.fit(X, y)
        
        predictions = model.predict(X)
        assert len(predictions) == len(y)


class TestTreeModels:
    """Tests for tree-based models."""
    
    @pytest.fixture
    def sample_data(self):
        np.random.seed(42)
        X = np.random.randn(100, 10)
        y = X[:, 0] * 2 + X[:, 1] ** 2 + np.random.randn(100) * 0.1
        return X, y
    
    def test_random_forest(self, sample_data):
        """Test Random Forest regressor."""
        from automl_qsar.models import RandomForest
        
        X, y = sample_data
        model = RandomForest()
        model.fit(X, y)
        
        predictions = model.predict(X)
        assert len(predictions) == len(y)
    
    def test_gradient_boosting(self, sample_data):
        """Test Gradient Boosting regressor."""
        from automl_qsar.models import GradientBoosting
        
        X, y = sample_data
        model = GradientBoosting()
        model.fit(X, y)
        
        predictions = model.predict(X)
        assert len(predictions) == len(y)
    
    def test_xgboost(self, sample_data):
        """Test XGBoost regressor."""
        try:
            from automl_qsar.models import XGBoost
            
            X, y = sample_data
            model = XGBoost()
            model.fit(X, y)
            
            predictions = model.predict(X)
            assert len(predictions) == len(y)
        except ImportError:
            pytest.skip("XGBoost not installed")


class TestKernelModels:
    """Tests for kernel-based models."""
    
    @pytest.fixture
    def sample_data(self):
        np.random.seed(42)
        X = np.random.randn(50, 10)
        y = np.sin(X[:, 0]) + np.random.randn(50) * 0.1
        return X, y
    
    def test_svr(self, sample_data):
        """Test Support Vector Regression."""
        from automl_qsar.models import SVR
        
        X, y = sample_data
        model = SVR()
        model.fit(X, y)
        
        predictions = model.predict(X)
        assert len(predictions) == len(y)
    
    def test_kernel_ridge(self, sample_data):
        """Test Kernel Ridge Regression."""
        try:
            from automl_qsar.models import KernelRidge
            
            X, y = sample_data
            model = KernelRidge()
            model.fit(X, y)
            
            predictions = model.predict(X)
            assert len(predictions) == len(y)
        except (ImportError, AttributeError):
            pytest.skip("KernelRidge not available")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
