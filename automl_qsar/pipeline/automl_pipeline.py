"""
Main AutoML QSAR Pipeline

Orchestrates the entire automated QSAR modeling workflow.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional
import pickle
import json
from pathlib import Path

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


class AutoMLQSAR:
    """
    Automated Machine Learning for QSAR Modeling
    
    Complete pipeline for automated molecular property prediction.
    
    Args:
        featurizers: List of featurizers to use (default: all)
        models: List of models to try (default: all)
        optimization_method: HPO method ('bayesian', 'genetic', 'grid')
        cv_strategy: Cross-validation strategy ('nested', 'kfold', 'loso')
        ensemble_method: Ensemble method ('voting', 'stacking', 'topk')
        n_trials: Number of HPO trials
        random_state: Random state for reproducibility
    """
    
    def __init__(
        self,
        featurizers: List[str] = None,
        models: List[str] = None,
        optimization_method: str = 'bayesian',
        cv_strategy: str = 'kfold',
        ensemble_method: str = 'topk',
        n_trials: int = 50,
        random_state: int = 42
    ):
        self.featurizers = featurizers or ['rdkit', 'ecfp', 'maccs']
        self.models = models or ['ridge', 'randomforest', 'xgboost', 'nn']
        self.optimization_method = optimization_method
        self.cv_strategy = cv_strategy
        self.ensemble_method = ensemble_method
        self.n_trials = n_trials
        self.random_state = random_state
        
        # Data
        self.smiles = None
        self.target = None
        self.X = None
        self.y = None
        
        # Fitted components
        self.featurizer = None
        self.scaler = None
        self.feature_selector = None
        self.best_model = None
        self.ensemble = None
        self.results = {}
    
    def load_data(
        self,
        data: pd.DataFrame = None,
        filepath: str = None,
        smiles_col: str = 'SMILES',
        target_col: str = 'Activity'
    ):
        """
        Load molecular data
        
        Args:
            data: DataFrame with SMILES and target
            filepath: Path to CSV file
            smiles_col: Name of SMILES column
            target_col: Name of target column
        """
        if filepath is not None:
            data = pd.read_csv(filepath)
        
        if data is None:
            raise ValueError("Must provide either data or filepath")
        
        self.smiles = data[smiles_col].values
        self.target = data[target_col].values
        
        print(f"Loaded {len(self.smiles)} molecules")
        
        return self
    
    def _select_featurizer(self, featurizer_name: str):
        """Select featurizer by name"""
        featurizer_name = featurizer_name.lower()
        
        # Initialize featurizers lazily to avoid import errors
        if featurizer_name == 'rdkit':
            return RDKitDescriptors()
        elif featurizer_name in ['ecfp', 'ecfp4']:
            return ECFPFingerprints(radius=2, n_bits=2048)
        elif featurizer_name == 'ecfp6':
            return ECFPFingerprints(radius=3, n_bits=2048)
        elif featurizer_name == 'maccs':
            return MACCSFingerprints()
        elif featurizer_name == 'embeddings':
            return MolecularEmbeddings()
        elif featurizer_name == 'graph':
            return GraphFeaturizer()
        elif featurizer_name == 'pharmacophore':
            if PharmacophoreFeatures is not None:
                return PharmacophoreFeatures()
            else:
                raise ValueError("Pharmacophore features not available")
        else:
            return None
    
    def _select_model(self, model_name: str):
        """Select model by name"""
        models_map = {
            'linear': LinearRegression(),
            'ridge': Ridge(),
            'lasso': Lasso(),
            'elasticnet': ElasticNet(),
            'randomforest': RandomForest(),
            'rf': RandomForest(),
            'gradientboosting': GradientBoosting(),
            'gb': GradientBoosting(),
            'xgboost': XGBoost(),
            'xgb': XGBoost(),
            'svr': SVR(),
            'kernelridge': KernelRidge(),
            'nn': FeedForwardNN(),
            'deepnn': DeepNN(),
            'gnn': GraphNeuralNetwork()
        }
        
        return models_map.get(model_name.lower())
    
    def fit(self, X: np.ndarray = None, y: np.ndarray = None):
        """
        Run the automated QSAR pipeline
        
        Args:
            X: Pre-computed features (optional)
            y: Target values (optional)
            
        Returns:
            Results dictionary
        """
        print("="*50)
        print("AutoML QSAR Pipeline")
        print("="*50)
        
        # Use provided X, y or featurize from SMILES
        if X is None:
            if self.smiles is None:
                raise ValueError("Must load data first or provide X")
            
            print("\n[1/6] Featurization...")
            X = self._featurize()
        
        if y is None:
            if self.target is None:
                raise ValueError("Must load data first or provide y")
            y = self.target
        
        self.X = X
        self.y = y
        
        # Preprocessing
        print("\n[2/6] Preprocessing...")
        X = self._preprocess(X)
        
        # Model selection and training
        print("\n[3/6] Model Selection...")
        trained_models = self._train_models(X, y)
        
        # Hyperparameter optimization
        print("\n[4/6] Hyperparameter Optimization...")
        optimized_models = self._optimize_models(trained_models, X, y)
        
        # Evaluation
        print("\n[5/6] Model Evaluation...")
        self._evaluate_models(optimized_models, X, y)
        
        # Ensemble
        print("\n[6/6] Building Ensemble...")
        self._build_ensemble(optimized_models, X, y)
        
        print("\n" + "="*50)
        print("Pipeline Complete!")
        print("="*50)
        
        return self.results
    
    def _featurize(self) -> np.ndarray:
        """Featurize molecules"""
        # For now, use first featurizer
        # TODO: Try multiple featurizers and select best
        featurizer_name = self.featurizers[0]
        self.featurizer = self._select_featurizer(featurizer_name)
        
        print(f"Using featurizer: {self.featurizer.name}")
        
        # Convert SMILES to molecules and track valid indices
        from rdkit import Chem
        valid_indices = []
        valid_smiles = []
        
        for idx, smi in enumerate(self.smiles):
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                valid_indices.append(idx)
                valid_smiles.append(smi)
        
        if len(valid_indices) < len(self.smiles):
            print(f"Warning: {len(self.smiles) - len(valid_indices)} molecules failed to parse")
            print(f"Continuing with {len(valid_indices)} valid molecules")
            
            # Filter SMILES and target to only valid molecules
            self.smiles = np.array(valid_smiles)
            self.target = self.target[valid_indices]
        
        X = self.featurizer.featurize(self.smiles)
        print(f"Generated {X.shape[1]} features")
        
        return X
    
    def _preprocess(self, X: np.ndarray) -> np.ndarray:
        """Preprocess features"""
        # Scaling
        self.scaler = StandardScaler()
        X = self.scaler.fit_transform(X)
        print(f"Scaled features")
        
        # Feature selection
        if X.shape[1] > 100:
            self.feature_selector = VarianceThreshold(threshold=0.01)
            X = self.feature_selector.fit_transform(X)
            print(f"Selected {X.shape[1]} features (removed low variance)")
        
        return X
    
    def _train_models(self, X: np.ndarray, y: np.ndarray) -> List:
        """Train initial models"""
        trained_models = []
        
        for model_name in self.models:
            try:
                model = self._select_model(model_name)
                if model is None:
                    continue
                
                print(f"Training {model.name}...")
                model.fit(X, y)
                trained_models.append(model)
            except Exception as e:
                print(f"Error training {model_name}: {e}")
        
        return trained_models
    
    def _optimize_models(self, models: List, X: np.ndarray, y: np.ndarray) -> List:
        """Optimize model hyperparameters"""
        # For now, return models as-is
        # TODO: Implement HPO for each model
        print(f"Using {len(models)} models (HPO to be implemented)")
        return models
    
    def _evaluate_models(self, models: List, X: np.ndarray, y: np.ndarray):
        """Evaluate models using cross-validation"""
        cv = KFoldCV(n_splits=5, random_state=self.random_state)
        
        model_results = []
        for model in models:
            try:
                # Simple evaluation
                scores = []
                for train_idx, test_idx in cv.split(X, y):
                    X_train, X_test = X[train_idx], X[test_idx]
                    y_train, y_test = y[train_idx], y[test_idx]
                    
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    
                    rmse = Metrics.rmse(y_test, y_pred)
                    scores.append(rmse)
                
                mean_rmse = np.mean(scores)
                std_rmse = np.std(scores)
                
                model_results.append({
                    'model': model.name,
                    'rmse_mean': mean_rmse,
                    'rmse_std': std_rmse
                })
                
                print(f"{model.name}: RMSE = {mean_rmse:.4f} ± {std_rmse:.4f}")
                
            except Exception as e:
                print(f"Error evaluating {model.name}: {e}")
        
        self.results['model_performance'] = model_results
    
    def _build_ensemble(self, models: List, X: np.ndarray, y: np.ndarray):
        """Build ensemble of best models"""
        if self.ensemble_method == 'topk':
            self.ensemble = TopKEnsemble(k=min(5, len(models)))
        elif self.ensemble_method == 'voting':
            self.ensemble = VotingEnsemble()
        elif self.ensemble_method == 'stacking':
            self.ensemble = StackingEnsemble()
        else:
            self.ensemble = TopKEnsemble(k=min(5, len(models)))
        
        self.ensemble.fit(models, X, y)
        print(f"Built {self.ensemble.name} with {len(self.ensemble.models)} models")
    
    def predict(self, smiles: List[str] = None, X: np.ndarray = None) -> np.ndarray:
        """
        Make predictions
        
        Args:
            smiles: List of SMILES strings
            X: Pre-computed features
            
        Returns:
            Predictions
        """
        if X is None:
            if smiles is None:
                raise ValueError("Must provide either smiles or X")
            X = self.featurizer.featurize(smiles)
        
        # Preprocess
        X = self.scaler.transform(X)
        if self.feature_selector is not None:
            X = self.feature_selector.transform(X)
        
        # Predict
        predictions = self.ensemble.predict(X)
        
        return predictions
    
    def predict_with_uncertainty(self, smiles: List[str] = None, X: np.ndarray = None):
        """Make predictions with uncertainty estimates"""
        if X is None:
            if smiles is None:
                raise ValueError("Must provide either smiles or X")
            X = self.featurizer.featurize(smiles)
        
        # Preprocess
        X = self.scaler.transform(X)
        if self.feature_selector is not None:
            X = self.feature_selector.transform(X)
        
        # Predict with uncertainty
        predictions, uncertainties = self.ensemble.predict_with_uncertainty(X)
        
        return predictions, uncertainties
    
    def interpret(self):
        """Generate model interpretations"""
        print("\nGenerating model interpretations...")
        
        interpretations = {}
        
        # Feature importance
        try:
            if hasattr(self.ensemble.models[0], 'get_feature_importances'):
                importance = FeatureImportance.get_tree_importance(
                    self.ensemble.models[0],
                    self.featurizer.get_feature_names()
                )
                interpretations['feature_importance'] = importance
                print("Generated feature importance")
        except Exception as e:
            print(f"Could not generate feature importance: {e}")
        
        return interpretations
    
    def save(self, filepath: str):
        """Save trained pipeline"""
        save_dir = Path(filepath)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save components
        with open(save_dir / 'featurizer.pkl', 'wb') as f:
            pickle.dump(self.featurizer, f)
        
        with open(save_dir / 'scaler.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)
        
        with open(save_dir / 'ensemble.pkl', 'wb') as f:
            pickle.dump(self.ensemble, f)
        
        # Save results
        with open(save_dir / 'results.json', 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"Saved pipeline to {filepath}")
    
    def load(self, filepath: str):
        """Load trained pipeline"""
        load_dir = Path(filepath)
        
        with open(load_dir / 'featurizer.pkl', 'rb') as f:
            self.featurizer = pickle.load(f)
        
        with open(load_dir / 'scaler.pkl', 'rb') as f:
            self.scaler = pickle.load(f)
        
        with open(load_dir / 'ensemble.pkl', 'rb') as f:
            self.ensemble = pickle.load(f)
        
        with open(load_dir / 'results.json', 'r') as f:
            self.results = json.load(f)
        
        print(f"Loaded pipeline from {filepath}")
