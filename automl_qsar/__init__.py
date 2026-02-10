"""
AutoML QSAR: Automated Machine Learning for QSAR Regression Modeling

An agentic AutoML framework for regression-based QSAR analysis with:
- Automated data validation and noise detection
- Outlier detection and management
- Diverse feature space exploration
- Multiple model family exploration
- Iterative hyperparameter optimization
- Automated report generation
"""

__version__ = "0.2.0"

from .pipeline.automl_regressor import AutoMLQSARRegressor, AutoMLQSAR
from .pipeline.automl_regressor import (
    DataValidator,
    FeatureSpaceExplorer,
    ModelExplorer,
    HyperparameterOptimizer,
    ReportGenerator
)

__all__ = [
    "AutoMLQSAR",
    "AutoMLQSARRegressor",
    "DataValidator",
    "FeatureSpaceExplorer",
    "ModelExplorer",
    "HyperparameterOptimizer",
    "ReportGenerator"
]
