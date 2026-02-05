"""
Bayesian Hyperparameter Optimization using Optuna
"""

import numpy as np
from typing import Dict, Any, Callable
import optuna
from optuna.samplers import TPESampler


class BayesianOptimizer:
    """
    Bayesian Optimization for hyperparameter tuning using Optuna
    
    Args:
        n_trials: Number of optimization trials
        timeout: Timeout in seconds
        n_jobs: Number of parallel jobs
    """
    
    def __init__(
        self,
        n_trials: int = 100,
        timeout: int = None,
        n_jobs: int = 1,
        random_state: int = 42
    ):
        self.n_trials = n_trials
        self.timeout = timeout
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.name = "BayesianOptimizer"
        self.study = None
        self.best_params = None
        self.best_value = None
    
    def optimize(
        self,
        objective_func: Callable,
        param_space: Dict[str, Any],
        direction: str = 'minimize'
    ) -> Dict[str, Any]:
        """
        Run Bayesian optimization
        
        Args:
            objective_func: Function to optimize (returns metric value)
            param_space: Parameter search space
            direction: 'minimize' or 'maximize'
            
        Returns:
            Best parameters found
        """
        # Create study
        sampler = TPESampler(seed=self.random_state)
        self.study = optuna.create_study(
            direction=direction,
            sampler=sampler
        )
        
        # Optimize
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        self.study.optimize(
            objective_func,
            n_trials=self.n_trials,
            timeout=self.timeout,
            n_jobs=self.n_jobs,
            show_progress_bar=True
        )
        
        self.best_params = self.study.best_params
        self.best_value = self.study.best_value
        
        return self.best_params
    
    def get_optimization_history(self):
        """Get optimization history"""
        if self.study is None:
            return None
        
        return {
            'values': [trial.value for trial in self.study.trials],
            'params': [trial.params for trial in self.study.trials]
        }
