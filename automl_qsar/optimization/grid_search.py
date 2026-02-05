"""
Grid Search Hyperparameter Optimization
"""

import numpy as np
from typing import Dict, Any, Callable, List
from itertools import product


class GridSearchOptimizer:
    """
    Grid Search for hyperparameter optimization
    
    Args:
        n_jobs: Number of parallel jobs
    """
    
    def __init__(self, n_jobs: int = 1):
        self.n_jobs = n_jobs
        self.name = "GridSearchOptimizer"
        self.best_params = None
        self.best_value = None
        self.results = []
    
    def _generate_grid(self, param_space: Dict[str, List]) -> List[Dict]:
        """Generate parameter grid"""
        param_names = list(param_space.keys())
        param_values = list(param_space.values())
        
        grid = []
        for values in product(*param_values):
            params = dict(zip(param_names, values))
            grid.append(params)
        
        return grid
    
    def optimize(
        self,
        objective_func: Callable,
        param_space: Dict[str, List],
        direction: str = 'minimize'
    ) -> Dict[str, Any]:
        """
        Run grid search optimization
        
        Args:
            objective_func: Function to optimize
            param_space: Parameter search space (dict of lists)
            direction: 'minimize' or 'maximize'
            
        Returns:
            Best parameters found
        """
        # Generate grid
        grid = self._generate_grid(param_space)
        
        print(f"Grid Search: {len(grid)} parameter combinations to evaluate")
        
        # Evaluate all combinations
        self.results = []
        for i, params in enumerate(grid):
            try:
                score = objective_func(params)
                self.results.append({
                    'params': params,
                    'score': score
                })
                
                # Track best
                if self.best_value is None or \
                   (direction == 'minimize' and score < self.best_value) or \
                   (direction == 'maximize' and score > self.best_value):
                    self.best_value = score
                    self.best_params = params.copy()
                
                if (i + 1) % 10 == 0:
                    print(f"Evaluated {i + 1}/{len(grid)} combinations. Best so far: {self.best_value:.4f}")
                    
            except Exception as e:
                print(f"Error evaluating params {params}: {e}")
                continue
        
        return self.best_params
    
    def get_all_results(self) -> List[Dict]:
        """Get all evaluation results"""
        return self.results
