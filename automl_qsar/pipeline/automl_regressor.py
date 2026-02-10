"""
Agentic AutoML QSAR Pipeline for Regression

A comprehensive automated machine learning framework for QSAR modeling
that handles:
- Data validation, noise detection, and outlier management
- Diverse feature space exploration (descriptors, fingerprints, preprocessing)
- Multiple model families (tree-based, neural networks, kernel methods)
- Iterative hyperparameter optimization
- Robust evaluation, benchmarking, and automated report generation

Inspired by TPOT and H2O AutoML, but specialized for molecular property prediction.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
import pickle
import json
from pathlib import Path
from datetime import datetime
import warnings

from ..featurization import (
    RDKitDescriptors, ECFPFingerprints, MACCSFingerprints,
    PharmacophoreFeatures, MolecularEmbeddings, GraphFeaturizer
)
from ..preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler,
    VarianceThreshold, BorutaSelector, CorrelationSelector,
    PCAReducer, UMAPReducer
)
from ..models import (
    LinearRegression, Ridge, Lasso, ElasticNet,
    RandomForest, GradientBoosting, XGBoost,
    SVR, KernelRidge,
    FeedForwardNN, DeepNN, GraphNeuralNetwork
)
from ..optimization import BayesianOptimizer, GeneticOptimizer, GridSearchOptimizer
from ..evaluation import Metrics, NestedCV, LOSOCV, KFoldCV
from ..ensemble import VotingEnsemble, StackingEnsemble, TopKEnsemble
from ..interpretation import FeatureImportance, SHAPInterpreter, LIMEInterpreter


class DataValidator:
    """
    Handles data validation, noise detection, and outlier management for QSAR datasets.
    """
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.validation_report = {}
    
    def validate_smiles(self, smiles: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Validate SMILES strings and identify invalid molecules.
        
        Returns:
            valid_mask: Boolean array indicating valid molecules
            validation_info: Dict with validation statistics
        """
        from rdkit import Chem
        
        valid_mask = np.zeros(len(smiles), dtype=bool)
        invalid_smiles = []
        
        for i, smi in enumerate(smiles):
            mol = Chem.MolFromSmiles(str(smi))
            if mol is not None:
                valid_mask[i] = True
            else:
                invalid_smiles.append((i, smi))
        
        self.validation_report['smiles'] = {
            'total': len(smiles),
            'valid': valid_mask.sum(),
            'invalid': len(invalid_smiles),
            'invalid_indices': [x[0] for x in invalid_smiles]
        }
        
        if self.verbose and invalid_smiles:
            print(f"⚠️ Found {len(invalid_smiles)} invalid SMILES strings")
        
        return valid_mask, self.validation_report['smiles']
    
    def detect_outliers(self, y: np.ndarray, method: str = 'iqr', 
                        threshold: float = 1.5) -> Tuple[np.ndarray, Dict]:
        """
        Detect outliers in target values using various methods.
        
        Args:
            y: Target values
            method: 'iqr', 'zscore', or 'isolation_forest'
            threshold: Threshold for outlier detection
            
        Returns:
            outlier_mask: Boolean array indicating outliers
            outlier_info: Dict with outlier statistics
        """
        outlier_mask = np.zeros(len(y), dtype=bool)
        
        if method == 'iqr':
            Q1, Q3 = np.percentile(y, [25, 75])
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            outlier_mask = (y < lower_bound) | (y > upper_bound)
            
        elif method == 'zscore':
            z_scores = np.abs((y - np.mean(y)) / np.std(y))
            outlier_mask = z_scores > threshold
            
        elif method == 'isolation_forest':
            from sklearn.ensemble import IsolationForest
            iso_forest = IsolationForest(contamination=0.1, random_state=42)
            predictions = iso_forest.fit_predict(y.reshape(-1, 1))
            outlier_mask = predictions == -1
        
        outlier_info = {
            'method': method,
            'threshold': threshold,
            'n_outliers': outlier_mask.sum(),
            'outlier_indices': np.where(outlier_mask)[0].tolist(),
            'outlier_values': y[outlier_mask].tolist() if outlier_mask.sum() > 0 else []
        }
        
        self.validation_report['outliers'] = outlier_info
        
        if self.verbose and outlier_mask.sum() > 0:
            print(f"⚠️ Detected {outlier_mask.sum()} outliers using {method} method")
        
        return outlier_mask, outlier_info
    
    def detect_noise(self, X: np.ndarray, y: np.ndarray, 
                     method: str = 'residual') -> Tuple[np.ndarray, Dict]:
        """
        Detect potentially noisy data points.
        
        Args:
            X: Feature matrix
            y: Target values
            method: 'residual' or 'knn'
            
        Returns:
            noise_mask: Boolean array indicating noisy points
            noise_info: Dict with noise statistics
        """
        noise_scores = np.zeros(len(y))
        
        if method == 'residual':
            # Use a simple model to detect high-residual points
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.model_selection import cross_val_predict
            
            rf = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
            y_pred = cross_val_predict(rf, X, y, cv=5)
            residuals = np.abs(y - y_pred)
            noise_scores = residuals / np.std(residuals)
            
        elif method == 'knn':
            # Use k-NN consistency to detect noise
            from sklearn.neighbors import NearestNeighbors
            
            k = min(5, len(y) - 1)
            nn = NearestNeighbors(n_neighbors=k + 1)
            nn.fit(X)
            distances, indices = nn.kneighbors(X)
            
            for i in range(len(y)):
                neighbor_indices = indices[i, 1:]  # Exclude self
                neighbor_values = y[neighbor_indices]
                noise_scores[i] = np.abs(y[i] - np.mean(neighbor_values)) / np.std(neighbor_values + 1e-10)
        
        # Flag points with noise score > 2.5 standard deviations
        noise_threshold = 2.5
        noise_mask = noise_scores > noise_threshold
        
        noise_info = {
            'method': method,
            'threshold': noise_threshold,
            'n_noisy': noise_mask.sum(),
            'noisy_indices': np.where(noise_mask)[0].tolist(),
            'noise_scores': noise_scores[noise_mask].tolist() if noise_mask.sum() > 0 else []
        }
        
        self.validation_report['noise'] = noise_info
        
        if self.verbose and noise_mask.sum() > 0:
            print(f"⚠️ Detected {noise_mask.sum()} potentially noisy data points")
        
        return noise_mask, noise_info
    
    def check_duplicates(self, smiles: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Check for duplicate molecules.
        """
        from rdkit import Chem
        
        canonical_smiles = []
        for smi in smiles:
            mol = Chem.MolFromSmiles(str(smi))
            if mol:
                canonical_smiles.append(Chem.MolToSmiles(mol, canonical=True))
            else:
                canonical_smiles.append(None)
        
        seen = {}
        duplicate_mask = np.zeros(len(smiles), dtype=bool)
        
        for i, csmi in enumerate(canonical_smiles):
            if csmi is None:
                continue
            if csmi in seen:
                duplicate_mask[i] = True
            else:
                seen[csmi] = i
        
        duplicate_info = {
            'n_duplicates': duplicate_mask.sum(),
            'duplicate_indices': np.where(duplicate_mask)[0].tolist()
        }
        
        self.validation_report['duplicates'] = duplicate_info
        
        if self.verbose and duplicate_mask.sum() > 0:
            print(f"⚠️ Found {duplicate_mask.sum()} duplicate molecules")
        
        return duplicate_mask, duplicate_info
    
    def get_validation_summary(self) -> Dict:
        """Return complete validation report."""
        return self.validation_report


class FeatureSpaceExplorer:
    """
    Searches diverse feature space including descriptor generation,
    feature selection, and preprocessing methods.
    """
    
    def __init__(self, random_state: int = 42, verbose: bool = True):
        self.random_state = random_state
        self.verbose = verbose
        self.feature_configurations = []
        self.best_configuration = None
        self.exploration_results = []
    
    def get_featurizer(self, name: str):
        """Get featurizer by name."""
        featurizers = {
            'rdkit': RDKitDescriptors(),
            'ecfp': ECFPFingerprints(radius=2, n_bits=2048),
            'ecfp4': ECFPFingerprints(radius=2, n_bits=2048),
            'ecfp6': ECFPFingerprints(radius=3, n_bits=2048),
            'maccs': MACCSFingerprints(),
            'embeddings': MolecularEmbeddings(),
        }
        
        # Handle optional featurizers
        if name == 'pharmacophore':
            if PharmacophoreFeatures is not None:
                return PharmacophoreFeatures()
            return None
        if name == 'graph':
            try:
                return GraphFeaturizer()
            except ImportError:
                return None
        
        return featurizers.get(name.lower())
    
    def explore_feature_space(
        self,
        smiles: np.ndarray,
        y: np.ndarray,
        featurizers: List[str] = None,
        n_configurations: int = 10
    ) -> Dict[str, Any]:
        """
        Explore different feature configurations and find the best one.
        
        Args:
            smiles: Array of SMILES strings
            y: Target values
            featurizers: List of featurizers to try
            n_configurations: Number of configurations to evaluate
            
        Returns:
            Best feature configuration and results
        """
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.model_selection import cross_val_score
        
        featurizers = featurizers or ['rdkit', 'ecfp', 'maccs']
        results = []
        
        if self.verbose:
            print("\n🔍 Exploring Feature Space...")
        
        for feat_name in featurizers:
            featurizer = self.get_featurizer(feat_name)
            if featurizer is None:
                continue
            
            try:
                # Generate features
                X = featurizer.featurize(smiles)
                
                # Handle NaN/Inf
                X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
                
                # Try different preprocessing combinations
                preprocessing_options = [
                    ('standard', StandardScaler()),
                    ('minmax', MinMaxScaler()),
                    ('robust', RobustScaler()),
                ]
                
                for prep_name, scaler in preprocessing_options:
                    X_scaled = scaler.fit_transform(X)
                    
                    # Quick evaluation with Random Forest
                    rf = RandomForestRegressor(n_estimators=50, random_state=self.random_state, n_jobs=-1)
                    
                    try:
                        scores = cross_val_score(rf, X_scaled, y, cv=3, scoring='neg_root_mean_squared_error')
                        rmse = -scores.mean()
                        
                        config = {
                            'featurizer': feat_name,
                            'n_features': X.shape[1],
                            'preprocessing': prep_name,
                            'rmse': rmse,
                            'rmse_std': scores.std()
                        }
                        results.append(config)
                        
                        if self.verbose:
                            print(f"  {feat_name} + {prep_name}: RMSE = {rmse:.4f} ± {scores.std():.4f}")
                    
                    except Exception as e:
                        if self.verbose:
                            print(f"  ⚠️ {feat_name} + {prep_name}: Failed - {str(e)[:50]}")
                
            except Exception as e:
                if self.verbose:
                    print(f"  ⚠️ {feat_name}: Failed to featurize - {str(e)[:50]}")
        
        # Sort by RMSE
        results = sorted(results, key=lambda x: x['rmse'])
        self.exploration_results = results
        
        if results:
            self.best_configuration = results[0]
            if self.verbose:
                print(f"\n✅ Best configuration: {results[0]['featurizer']} + {results[0]['preprocessing']}")
                print(f"   RMSE = {results[0]['rmse']:.4f}")
        
        return {
            'best_configuration': self.best_configuration,
            'all_results': results
        }


class ModelExplorer:
    """
    Explores multiple model families for regression tasks.
    """
    
    # Define model families
    MODEL_FAMILIES = {
        'linear': ['ridge', 'lasso', 'elasticnet', 'linearregression'],
        'tree': ['randomforest', 'gradientboosting', 'xgboost'],
        'kernel': ['svr', 'kernelridge'],
        'neural': ['feedforward', 'deepnn'],
        'graph': ['gnn']
    }
    
    def __init__(self, random_state: int = 42, verbose: bool = True):
        self.random_state = random_state
        self.verbose = verbose
        self.trained_models = {}
        self.model_scores = {}
    
    def get_model(self, name: str, params: Dict = None):
        """Get model instance by name."""
        params = params or {}
        name = name.lower()
        
        model_classes = {
            'linearregression': LinearRegression,
            'ridge': Ridge,
            'lasso': Lasso,
            'elasticnet': ElasticNet,
            'randomforest': RandomForest,
            'gradientboosting': GradientBoosting,
            'xgboost': XGBoost,
            'svr': SVR,
            'kernelridge': KernelRidge,
            'feedforward': FeedForwardNN,
            'deepnn': DeepNN,
            'nn': FeedForwardNN,
            'gnn': GraphNeuralNetwork
        }
        
        if name not in model_classes:
            raise ValueError(f"Unknown model: {name}")
        
        return model_classes[name](**params)
    
    def get_hyperparameter_space(self, model_name: str) -> Dict:
        """Get hyperparameter search space for a model."""
        spaces = {
            'ridge': {
                'alpha': {'type': 'float', 'low': 0.001, 'high': 100.0, 'log': True}
            },
            'lasso': {
                'alpha': {'type': 'float', 'low': 0.001, 'high': 10.0, 'log': True}
            },
            'elasticnet': {
                'alpha': {'type': 'float', 'low': 0.001, 'high': 10.0, 'log': True},
                'l1_ratio': {'type': 'float', 'low': 0.1, 'high': 0.9}
            },
            'randomforest': {
                'n_estimators': {'type': 'int', 'low': 50, 'high': 300},
                'max_depth': {'type': 'int', 'low': 3, 'high': 20},
                'min_samples_split': {'type': 'int', 'low': 2, 'high': 20},
                'min_samples_leaf': {'type': 'int', 'low': 1, 'high': 10}
            },
            'gradientboosting': {
                'n_estimators': {'type': 'int', 'low': 50, 'high': 300},
                'max_depth': {'type': 'int', 'low': 3, 'high': 10},
                'learning_rate': {'type': 'float', 'low': 0.01, 'high': 0.3, 'log': True}
            },
            'xgboost': {
                'n_estimators': {'type': 'int', 'low': 50, 'high': 300},
                'max_depth': {'type': 'int', 'low': 3, 'high': 10},
                'learning_rate': {'type': 'float', 'low': 0.01, 'high': 0.3, 'log': True},
                'subsample': {'type': 'float', 'low': 0.6, 'high': 1.0}
            },
            'svr': {
                'C': {'type': 'float', 'low': 0.1, 'high': 100.0, 'log': True},
                'epsilon': {'type': 'float', 'low': 0.01, 'high': 1.0},
                'kernel': {'type': 'categorical', 'choices': ['rbf', 'linear', 'poly']}
            },
            'feedforward': {
                'hidden_dim': {'type': 'int', 'low': 32, 'high': 256},
                'n_layers': {'type': 'int', 'low': 1, 'high': 4},
                'dropout': {'type': 'float', 'low': 0.1, 'high': 0.5}
            },
            'deepnn': {
                'hidden_dims': {'type': 'categorical', 'choices': [[128, 64], [256, 128, 64], [512, 256, 128]]}
            }
        }
        
        return spaces.get(model_name.lower(), {})
    
    def explore_models(
        self,
        X: np.ndarray,
        y: np.ndarray,
        models: List[str] = None,
        cv_folds: int = 5
    ) -> Dict[str, Any]:
        """
        Explore different model families.
        
        Args:
            X: Feature matrix
            y: Target values
            models: List of models to try
            cv_folds: Number of CV folds
            
        Returns:
            Model exploration results
        """
        from sklearn.model_selection import cross_val_score
        
        models = models or ['ridge', 'randomforest', 'xgboost', 'svr']
        results = []
        
        if self.verbose:
            print("\n🤖 Exploring Model Families...")
        
        for model_name in models:
            try:
                model = self.get_model(model_name)
                
                scores = cross_val_score(
                    model.model if hasattr(model, 'model') else model,
                    X, y,
                    cv=cv_folds,
                    scoring='neg_root_mean_squared_error',
                    n_jobs=-1
                )
                
                rmse = -scores.mean()
                rmse_std = scores.std()
                
                # Get model family
                family = 'other'
                for fam, members in self.MODEL_FAMILIES.items():
                    if model_name.lower() in members:
                        family = fam
                        break
                
                result = {
                    'model': model_name,
                    'family': family,
                    'rmse': rmse,
                    'rmse_std': rmse_std,
                    'r2': 1 - (rmse ** 2) / np.var(y)  # Approximate R²
                }
                results.append(result)
                
                if self.verbose:
                    print(f"  {model_name} ({family}): RMSE = {rmse:.4f} ± {rmse_std:.4f}")
                
            except Exception as e:
                if self.verbose:
                    print(f"  ⚠️ {model_name}: Failed - {str(e)[:50]}")
        
        # Sort by RMSE
        results = sorted(results, key=lambda x: x['rmse'])
        self.model_scores = {r['model']: r for r in results}
        
        return {
            'best_model': results[0] if results else None,
            'all_results': results
        }


class HyperparameterOptimizer:
    """
    Performs iterative hyperparameter optimization.
    """
    
    def __init__(
        self,
        method: str = 'bayesian',
        n_trials: int = 50,
        random_state: int = 42,
        verbose: bool = True
    ):
        self.method = method
        self.n_trials = n_trials
        self.random_state = random_state
        self.verbose = verbose
        self.optimization_history = []
    
    def optimize(
        self,
        model_name: str,
        X: np.ndarray,
        y: np.ndarray,
        param_space: Dict,
        cv_folds: int = 5
    ) -> Dict[str, Any]:
        """
        Optimize hyperparameters for a model.
        
        Args:
            model_name: Name of the model
            X: Feature matrix
            y: Target values
            param_space: Hyperparameter search space
            cv_folds: Number of CV folds
            
        Returns:
            Best parameters and optimization results
        """
        if not param_space:
            return {'best_params': {}, 'best_score': None}
        
        if self.verbose:
            print(f"\n⚙️ Optimizing {model_name} hyperparameters ({self.method})...")
        
        if self.method == 'bayesian':
            return self._bayesian_optimization(model_name, X, y, param_space, cv_folds)
        elif self.method == 'genetic':
            return self._genetic_optimization(model_name, X, y, param_space, cv_folds)
        else:  # grid
            return self._grid_search(model_name, X, y, param_space, cv_folds)
    
    def _bayesian_optimization(self, model_name, X, y, param_space, cv_folds):
        """Bayesian optimization using Optuna."""
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        
        from sklearn.model_selection import cross_val_score
        
        model_explorer = ModelExplorer(random_state=self.random_state, verbose=False)
        
        def objective(trial):
            params = {}
            for param_name, config in param_space.items():
                if config['type'] == 'int':
                    params[param_name] = trial.suggest_int(
                        param_name, config['low'], config['high']
                    )
                elif config['type'] == 'float':
                    if config.get('log', False):
                        params[param_name] = trial.suggest_float(
                            param_name, config['low'], config['high'], log=True
                        )
                    else:
                        params[param_name] = trial.suggest_float(
                            param_name, config['low'], config['high']
                        )
                elif config['type'] == 'categorical':
                    params[param_name] = trial.suggest_categorical(
                        param_name, config['choices']
                    )
            
            try:
                model = model_explorer.get_model(model_name, params)
                model_obj = model.model if hasattr(model, 'model') else model
                
                scores = cross_val_score(
                    model_obj, X, y, cv=cv_folds,
                    scoring='neg_root_mean_squared_error',
                    n_jobs=-1
                )
                return -scores.mean()
            except Exception:
                return float('inf')
        
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=self.verbose)
        
        self.optimization_history = [
            {'trial': i, 'value': t.value, 'params': t.params}
            for i, t in enumerate(study.trials)
        ]
        
        if self.verbose:
            print(f"  Best RMSE: {study.best_value:.4f}")
            print(f"  Best params: {study.best_params}")
        
        return {
            'best_params': study.best_params,
            'best_score': study.best_value,
            'n_trials': len(study.trials),
            'history': self.optimization_history
        }
    
    def _genetic_optimization(self, model_name, X, y, param_space, cv_folds):
        """Genetic algorithm optimization."""
        from sklearn.model_selection import cross_val_score
        
        model_explorer = ModelExplorer(random_state=self.random_state, verbose=False)
        
        def objective(params):
            try:
                model = model_explorer.get_model(model_name, params)
                model_obj = model.model if hasattr(model, 'model') else model
                
                scores = cross_val_score(
                    model_obj, X, y, cv=cv_folds,
                    scoring='neg_root_mean_squared_error',
                    n_jobs=-1
                )
                return -scores.mean()
            except Exception:
                return float('inf')
        
        optimizer = GeneticOptimizer(
            population_size=20,
            n_generations=self.n_trials // 20 + 1,
            random_state=self.random_state
        )
        
        best_params = optimizer.optimize(objective, param_space, direction='minimize')
        
        return {
            'best_params': best_params,
            'best_score': optimizer.best_value,
            'history': []
        }
    
    def _grid_search(self, model_name, X, y, param_space, cv_folds):
        """Grid search optimization."""
        from sklearn.model_selection import GridSearchCV
        
        model_explorer = ModelExplorer(random_state=self.random_state, verbose=False)
        model = model_explorer.get_model(model_name)
        model_obj = model.model if hasattr(model, 'model') else model
        
        # Convert param space to grid
        param_grid = {}
        for param_name, config in param_space.items():
            if config['type'] == 'int':
                param_grid[param_name] = list(range(config['low'], config['high'] + 1, 
                                                    max(1, (config['high'] - config['low']) // 5)))
            elif config['type'] == 'float':
                param_grid[param_name] = np.linspace(config['low'], config['high'], 5).tolist()
            elif config['type'] == 'categorical':
                param_grid[param_name] = config['choices']
        
        grid_search = GridSearchCV(
            model_obj, param_grid, cv=cv_folds,
            scoring='neg_root_mean_squared_error',
            n_jobs=-1
        )
        grid_search.fit(X, y)
        
        return {
            'best_params': grid_search.best_params_,
            'best_score': -grid_search.best_score_,
            'history': []
        }


class ReportGenerator:
    """
    Generates automated reports for QSAR model evaluation and benchmarking.
    """
    
    def __init__(self, output_dir: str = './qsar_report'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.report_data = {}
    
    def add_section(self, section_name: str, content: Dict):
        """Add a section to the report."""
        self.report_data[section_name] = content
    
    def generate_markdown_report(self) -> str:
        """Generate a Markdown report."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        report = f"""# AutoML QSAR Regression Report

**Generated:** {timestamp}

---

## Executive Summary

"""
        if 'summary' in self.report_data:
            summary = self.report_data['summary']
            report += f"""
- **Dataset Size:** {summary.get('n_samples', 'N/A')} molecules
- **Best Model:** {summary.get('best_model', 'N/A')}
- **Best RMSE:** {summary.get('best_rmse', 'N/A'):.4f}
- **Best R²:** {summary.get('best_r2', 'N/A'):.4f}

"""
        
        # Data Validation Section
        if 'validation' in self.report_data:
            val = self.report_data['validation']
            report += """## Data Validation

| Check | Status | Details |
|-------|--------|---------|
"""
            if 'smiles' in val:
                status = '✅' if val['smiles']['invalid'] == 0 else '⚠️'
                report += f"| SMILES Validation | {status} | {val['smiles']['valid']}/{val['smiles']['total']} valid |\n"
            if 'outliers' in val:
                status = '✅' if val['outliers']['n_outliers'] == 0 else '⚠️'
                report += f"| Outlier Detection | {status} | {val['outliers']['n_outliers']} outliers found |\n"
            if 'duplicates' in val:
                status = '✅' if val['duplicates']['n_duplicates'] == 0 else '⚠️'
                report += f"| Duplicate Check | {status} | {val['duplicates']['n_duplicates']} duplicates found |\n"
            report += "\n"
        
        # Feature Exploration Section
        if 'features' in self.report_data:
            feat = self.report_data['features']
            report += """## Feature Space Exploration

| Featurizer | Preprocessing | RMSE | Features |
|------------|---------------|------|----------|
"""
            for config in feat.get('all_results', [])[:5]:
                report += f"| {config['featurizer']} | {config['preprocessing']} | {config['rmse']:.4f} | {config['n_features']} |\n"
            report += "\n"
        
        # Model Exploration Section
        if 'models' in self.report_data:
            models = self.report_data['models']
            report += """## Model Benchmarking

| Model | Family | RMSE | RMSE Std |
|-------|--------|------|----------|
"""
            for result in models.get('all_results', []):
                report += f"| {result['model']} | {result['family']} | {result['rmse']:.4f} | {result['rmse_std']:.4f} |\n"
            report += "\n"
        
        # Hyperparameter Optimization Section
        if 'hpo' in self.report_data:
            hpo = self.report_data['hpo']
            report += f"""## Hyperparameter Optimization

- **Method:** {hpo.get('method', 'N/A')}
- **Trials:** {hpo.get('n_trials', 'N/A')}
- **Best Score:** {hpo.get('best_score', 'N/A'):.4f}

**Best Parameters:**
```json
{json.dumps(hpo.get('best_params', {}), indent=2)}
```

"""
        
        # Final Model Section
        if 'final_model' in self.report_data:
            final = self.report_data['final_model']
            report += f"""## Final Model Performance

| Metric | Value |
|--------|-------|
| RMSE | {final.get('rmse', 'N/A'):.4f} |
| MAE | {final.get('mae', 'N/A'):.4f} |
| R² | {final.get('r2', 'N/A'):.4f} |
| Q² (CV) | {final.get('q2', 'N/A'):.4f} |

"""
        
        # Feature Importance Section
        if 'importance' in self.report_data:
            imp = self.report_data['importance']
            report += """## Feature Importance

Top 10 Most Important Features:

| Rank | Feature | Importance |
|------|---------|------------|
"""
            for i, (feat, score) in enumerate(list(imp.items())[:10], 1):
                report += f"| {i} | {feat} | {score:.4f} |\n"
            report += "\n"
        
        report += """---

*Report generated by AutoML QSAR Framework*
"""
        
        return report
    
    def save_report(self, filename: str = 'qsar_report.md') -> str:
        """Save report to file."""
        report = self.generate_markdown_report()
        filepath = self.output_dir / filename
        
        with open(filepath, 'w') as f:
            f.write(report)
        
        # Also save JSON data
        json_path = self.output_dir / 'report_data.json'
        with open(json_path, 'w') as f:
            json.dump(self.report_data, f, indent=2, default=str)
        
        return str(filepath)


class AutoMLQSARRegressor:
    """
    Agentic AutoML QSAR Framework for Regression Tasks
    
    A comprehensive automated machine learning pipeline that:
    - Handles data validation, noise detection, and outlier management
    - Searches diverse feature space (descriptors, fingerprints, preprocessing)
    - Explores multiple model families (tree-based, neural networks, kernel methods)
    - Performs iterative hyperparameter optimization
    - Conducts robust evaluation, benchmarking, and automated report generation
    
    Inspired by TPOT and H2O AutoML, specialized for molecular property prediction.
    
    Args:
        featurizers: List of featurizers to explore
        models: List of models to try
        optimization_method: HPO method ('bayesian', 'genetic', 'grid')
        n_trials: Number of HPO trials
        cv_folds: Number of cross-validation folds
        handle_outliers: Whether to detect and handle outliers
        detect_noise: Whether to detect noisy data points
        random_state: Random state for reproducibility
        verbose: Print progress messages
    """
    
    def __init__(
        self,
        featurizers: List[str] = None,
        models: List[str] = None,
        optimization_method: str = 'bayesian',
        n_trials: int = 50,
        cv_folds: int = 5,
        handle_outliers: bool = True,
        detect_noise: bool = True,
        ensemble_method: str = 'topk',
        random_state: int = 42,
        verbose: bool = True
    ):
        self.featurizers = featurizers or ['rdkit', 'ecfp', 'maccs']
        self.models = models or ['ridge', 'randomforest', 'xgboost', 'svr']
        self.optimization_method = optimization_method
        self.n_trials = n_trials
        self.cv_folds = cv_folds
        self.handle_outliers = handle_outliers
        self.detect_noise = detect_noise
        self.ensemble_method = ensemble_method
        self.random_state = random_state
        self.verbose = verbose
        
        # Components
        self.data_validator = DataValidator(verbose=verbose)
        self.feature_explorer = FeatureSpaceExplorer(random_state=random_state, verbose=verbose)
        self.model_explorer = ModelExplorer(random_state=random_state, verbose=verbose)
        self.hpo = HyperparameterOptimizer(
            method=optimization_method,
            n_trials=n_trials,
            random_state=random_state,
            verbose=verbose
        )
        self.report_generator = ReportGenerator()
        
        # Data
        self.smiles = None
        self.y = None
        self.X = None
        self.valid_mask = None
        
        # Fitted components
        self.featurizer = None
        self.scaler = None
        self.best_model = None
        self.ensemble = None
        self.best_params = None
        
        # Results
        self.results = {}
    
    def load_data(
        self,
        data: pd.DataFrame = None,
        filepath: str = None,
        smiles_col: str = 'SMILES',
        target_col: str = 'Activity'
    ) -> 'AutoMLQSARRegressor':
        """
        Load and validate molecular data.
        
        Args:
            data: DataFrame with SMILES and target values
            filepath: Path to CSV file
            smiles_col: Name of SMILES column
            target_col: Name of target column (continuous values for regression)
        """
        if filepath is not None:
            data = pd.read_csv(filepath)
        
        if data is None:
            raise ValueError("Must provide either data or filepath")
        
        self.smiles = data[smiles_col].values
        self.y = data[target_col].values.astype(float)
        
        if self.verbose:
            print(f"\n📊 Loaded {len(self.smiles)} molecules")
            print(f"   Target range: {self.y.min():.4f} - {self.y.max():.4f}")
            print(f"   Target mean: {self.y.mean():.4f} ± {self.y.std():.4f}")
        
        return self
    
    def validate_data(self) -> Dict:
        """
        Perform comprehensive data validation.
        """
        if self.verbose:
            print("\n🔍 Validating Data...")
        
        # Validate SMILES
        valid_mask, smiles_info = self.data_validator.validate_smiles(self.smiles)
        self.valid_mask = valid_mask
        
        # Check duplicates
        dup_mask, dup_info = self.data_validator.check_duplicates(self.smiles[valid_mask])
        
        # Detect outliers
        if self.handle_outliers:
            outlier_mask, outlier_info = self.data_validator.detect_outliers(
                self.y[valid_mask], method='iqr'
            )
        
        validation_summary = self.data_validator.get_validation_summary()
        self.report_generator.add_section('validation', validation_summary)
        
        # Filter to valid molecules
        self.smiles = self.smiles[valid_mask]
        self.y = self.y[valid_mask]
        
        if self.verbose:
            print(f"   ✅ {len(self.smiles)} valid molecules after filtering")
        
        return validation_summary
    
    def fit(self) -> Dict[str, Any]:
        """
        Run the complete AutoML pipeline.
        
        Returns:
            Dictionary with all results and trained models
        """
        if self.smiles is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        print("\n" + "="*60)
        print("🚀 AutoML QSAR Regression Pipeline")
        print("="*60)
        
        # Step 1: Data Validation
        validation_results = self.validate_data()
        
        # Step 2: Feature Space Exploration
        feature_results = self.feature_explorer.explore_feature_space(
            self.smiles, self.y, self.featurizers
        )
        self.report_generator.add_section('features', feature_results)
        
        # Use best featurizer
        best_feat_config = feature_results['best_configuration']
        self.featurizer = self.feature_explorer.get_featurizer(best_feat_config['featurizer'])
        self.X = self.featurizer.featurize(self.smiles)
        self.X = np.nan_to_num(self.X, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Apply preprocessing
        if best_feat_config['preprocessing'] == 'standard':
            self.scaler = StandardScaler()
        elif best_feat_config['preprocessing'] == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            self.scaler = RobustScaler()
        
        self.X = self.scaler.fit_transform(self.X)
        
        # Step 3: Detect noise (after featurization)
        if self.detect_noise:
            noise_mask, noise_info = self.data_validator.detect_noise(self.X, self.y)
        
        # Step 4: Model Exploration
        model_results = self.model_explorer.explore_models(
            self.X, self.y, self.models, self.cv_folds
        )
        self.report_generator.add_section('models', model_results)
        
        # Step 5: Hyperparameter Optimization for top models
        best_model_name = model_results['best_model']['model']
        param_space = self.model_explorer.get_hyperparameter_space(best_model_name)
        
        hpo_results = self.hpo.optimize(
            best_model_name, self.X, self.y, param_space, self.cv_folds
        )
        self.best_params = hpo_results['best_params']
        self.report_generator.add_section('hpo', {
            'method': self.optimization_method,
            'n_trials': self.n_trials,
            **hpo_results
        })
        
        # Step 6: Train final model with best parameters
        if self.verbose:
            print("\n🎯 Training Final Model...")
        
        self.best_model = self.model_explorer.get_model(best_model_name, self.best_params)
        model_obj = self.best_model.model if hasattr(self.best_model, 'model') else self.best_model
        model_obj.fit(self.X, self.y)
        
        # Step 7: Final Evaluation
        from sklearn.model_selection import cross_val_predict
        y_pred_cv = cross_val_predict(model_obj, self.X, self.y, cv=self.cv_folds)
        
        final_metrics = {
            'rmse': Metrics.rmse(self.y, y_pred_cv),
            'mae': Metrics.mae(self.y, y_pred_cv),
            'r2': Metrics.r2(self.y, y_pred_cv),
            'q2': 1 - np.sum((self.y - y_pred_cv)**2) / np.sum((self.y - np.mean(self.y))**2)
        }
        self.report_generator.add_section('final_model', final_metrics)
        
        if self.verbose:
            print(f"\n📈 Final Model Performance:")
            print(f"   RMSE: {final_metrics['rmse']:.4f}")
            print(f"   MAE:  {final_metrics['mae']:.4f}")
            print(f"   R²:   {final_metrics['r2']:.4f}")
            print(f"   Q²:   {final_metrics['q2']:.4f}")
        
        # Step 8: Build Ensemble
        if self.verbose:
            print("\n🔗 Building Ensemble...")
        
        trained_models = []
        for model_name in self.models[:3]:  # Top 3 models
            try:
                model = self.model_explorer.get_model(model_name)
                model_obj = model.model if hasattr(model, 'model') else model
                model_obj.fit(self.X, self.y)
                trained_models.append(model)
            except Exception:
                pass
        
        if trained_models:
            self.ensemble = TopKEnsemble(k=min(3, len(trained_models)))
            self.ensemble.fit(trained_models, self.X, self.y)
        
        # Step 9: Feature Importance
        if self.verbose:
            print("\n📊 Computing Feature Importance...")
        
        try:
            feature_names = self.featurizer.get_feature_names() if hasattr(self.featurizer, 'get_feature_names') else [f'feature_{i}' for i in range(self.X.shape[1])]
            importance_dict = FeatureImportance.get_importance(model_obj, feature_names)
            self.report_generator.add_section('importance', importance_dict)
        except Exception:
            pass
        
        # Step 10: Generate Summary
        summary = {
            'n_samples': len(self.smiles),
            'n_features': self.X.shape[1],
            'best_model': best_model_name,
            'best_rmse': final_metrics['rmse'],
            'best_r2': final_metrics['r2']
        }
        self.report_generator.add_section('summary', summary)
        
        # Generate Report
        if self.verbose:
            print("\n📝 Generating Report...")
        
        report_path = self.report_generator.save_report()
        
        print("\n" + "="*60)
        print("✅ AutoML QSAR Pipeline Complete!")
        print("="*60)
        print(f"\n📄 Report saved to: {report_path}")
        
        self.results = {
            'validation': validation_results,
            'features': feature_results,
            'models': model_results,
            'hpo': hpo_results,
            'final_metrics': final_metrics,
            'summary': summary
        }
        
        return self.results
    
    def predict(self, smiles: List[str]) -> np.ndarray:
        """
        Make predictions on new molecules.
        
        Args:
            smiles: List of SMILES strings
            
        Returns:
            Predicted values (regression)
        """
        if self.best_model is None:
            raise ValueError("Model not trained. Call fit() first.")
        
        # Featurize
        X = self.featurizer.featurize(smiles)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        X = self.scaler.transform(X)
        
        # Predict
        if self.ensemble is not None:
            return self.ensemble.predict(X)
        else:
            model_obj = self.best_model.model if hasattr(self.best_model, 'model') else self.best_model
            return model_obj.predict(X)
    
    def predict_with_uncertainty(self, smiles: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions with uncertainty estimates.
        
        Args:
            smiles: List of SMILES strings
            
        Returns:
            Tuple of (predictions, uncertainties)
        """
        if self.ensemble is None:
            # Return predictions with zero uncertainty
            preds = self.predict(smiles)
            return preds, np.zeros_like(preds)
        
        # Featurize
        X = self.featurizer.featurize(smiles)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        X = self.scaler.transform(X)
        
        return self.ensemble.predict_with_uncertainty(X)
    
    def save(self, filepath: str) -> None:
        """Save the trained pipeline."""
        filepath = Path(filepath)
        filepath.mkdir(parents=True, exist_ok=True)
        
        # Save configuration
        config = {
            'featurizers': self.featurizers,
            'models': self.models,
            'optimization_method': self.optimization_method,
            'n_trials': self.n_trials,
            'cv_folds': self.cv_folds,
            'best_params': self.best_params,
            'results': self.results
        }
        
        with open(filepath / 'config.json', 'w') as f:
            json.dump(config, f, indent=2, default=str)
        
        # Save scaler
        with open(filepath / 'scaler.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)
        
        # Save model
        if self.best_model is not None:
            model_obj = self.best_model.model if hasattr(self.best_model, 'model') else self.best_model
            with open(filepath / 'model.pkl', 'wb') as f:
                pickle.dump(model_obj, f)
        
        if self.verbose:
            print(f"✅ Pipeline saved to {filepath}")
    
    def load(self, filepath: str) -> None:
        """Load a trained pipeline."""
        filepath = Path(filepath)
        
        # Load configuration
        with open(filepath / 'config.json', 'r') as f:
            config = json.load(f)
        
        self.featurizers = config['featurizers']
        self.models = config['models']
        self.optimization_method = config['optimization_method']
        self.best_params = config.get('best_params', {})
        self.results = config.get('results', {})
        
        # Load scaler
        with open(filepath / 'scaler.pkl', 'rb') as f:
            self.scaler = pickle.load(f)
        
        # Load model
        with open(filepath / 'model.pkl', 'rb') as f:
            model = pickle.load(f)
        
        # Wrap in a class if needed
        class ModelWrapper:
            def __init__(self, model):
                self.model = model
        
        self.best_model = ModelWrapper(model)
        
        # Re-initialize featurizer based on config
        if self.results.get('features', {}).get('best_configuration'):
            feat_name = self.results['features']['best_configuration']['featurizer']
            self.featurizer = self.feature_explorer.get_featurizer(feat_name)
        else:
            self.featurizer = self.feature_explorer.get_featurizer('rdkit')
        
        if self.verbose:
            print(f"✅ Pipeline loaded from {filepath}")


# Backwards compatibility alias
AutoMLQSAR = AutoMLQSARRegressor
