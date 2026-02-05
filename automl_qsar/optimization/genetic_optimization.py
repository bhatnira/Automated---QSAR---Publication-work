"""
Genetic Algorithm for Hyperparameter Optimization
"""

import numpy as np
from typing import Dict, Any, Callable, List
import random


class GeneticOptimizer:
    """
    Genetic Algorithm for hyperparameter optimization
    
    Args:
        population_size: Size of population
        n_generations: Number of generations
        mutation_rate: Probability of mutation
        crossover_rate: Probability of crossover
    """
    
    def __init__(
        self,
        population_size: int = 50,
        n_generations: int = 20,
        mutation_rate: float = 0.1,
        crossover_rate: float = 0.7,
        random_state: int = 42
    ):
        self.population_size = population_size
        self.n_generations = n_generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.random_state = random_state
        self.name = "GeneticOptimizer"
        self.best_params = None
        self.best_value = None
        
        random.seed(random_state)
        np.random.seed(random_state)
    
    def _initialize_population(self, param_space: Dict[str, Any]) -> List[Dict]:
        """Initialize random population"""
        population = []
        
        for _ in range(self.population_size):
            individual = {}
            for param_name, param_config in param_space.items():
                if param_config['type'] == 'int':
                    individual[param_name] = random.randint(
                        param_config['low'],
                        param_config['high']
                    )
                elif param_config['type'] == 'float':
                    individual[param_name] = random.uniform(
                        param_config['low'],
                        param_config['high']
                    )
                elif param_config['type'] == 'categorical':
                    individual[param_name] = random.choice(param_config['choices'])
            
            population.append(individual)
        
        return population
    
    def _mutate(self, individual: Dict, param_space: Dict[str, Any]) -> Dict:
        """Mutate an individual"""
        mutated = individual.copy()
        
        for param_name, param_config in param_space.items():
            if random.random() < self.mutation_rate:
                if param_config['type'] == 'int':
                    mutated[param_name] = random.randint(
                        param_config['low'],
                        param_config['high']
                    )
                elif param_config['type'] == 'float':
                    mutated[param_name] = random.uniform(
                        param_config['low'],
                        param_config['high']
                    )
                elif param_config['type'] == 'categorical':
                    mutated[param_name] = random.choice(param_config['choices'])
        
        return mutated
    
    def _crossover(self, parent1: Dict, parent2: Dict) -> tuple:
        """Crossover two parents"""
        if random.random() > self.crossover_rate:
            return parent1.copy(), parent2.copy()
        
        child1, child2 = parent1.copy(), parent2.copy()
        
        for param_name in parent1.keys():
            if random.random() < 0.5:
                child1[param_name] = parent2[param_name]
                child2[param_name] = parent1[param_name]
        
        return child1, child2
    
    def optimize(
        self,
        objective_func: Callable,
        param_space: Dict[str, Any],
        direction: str = 'minimize'
    ) -> Dict[str, Any]:
        """
        Run genetic algorithm optimization
        
        Args:
            objective_func: Function to optimize
            param_space: Parameter search space
            direction: 'minimize' or 'maximize'
            
        Returns:
            Best parameters found
        """
        # Initialize population
        population = self._initialize_population(param_space)
        
        # Evolution
        for generation in range(self.n_generations):
            # Evaluate fitness
            fitness_scores = []
            for individual in population:
                try:
                    score = objective_func(individual)
                    fitness_scores.append(score)
                except Exception as e:
                    fitness_scores.append(float('inf') if direction == 'minimize' else float('-inf'))
            
            # Track best
            if direction == 'minimize':
                best_idx = np.argmin(fitness_scores)
            else:
                best_idx = np.argmax(fitness_scores)
            
            if self.best_value is None or \
               (direction == 'minimize' and fitness_scores[best_idx] < self.best_value) or \
               (direction == 'maximize' and fitness_scores[best_idx] > self.best_value):
                self.best_value = fitness_scores[best_idx]
                self.best_params = population[best_idx].copy()
            
            # Selection (tournament)
            selected = []
            for _ in range(self.population_size):
                tournament_size = 3
                tournament_indices = random.sample(range(self.population_size), tournament_size)
                if direction == 'minimize':
                    winner_idx = min(tournament_indices, key=lambda i: fitness_scores[i])
                else:
                    winner_idx = max(tournament_indices, key=lambda i: fitness_scores[i])
                selected.append(population[winner_idx].copy())
            
            # Crossover and mutation
            new_population = []
            for i in range(0, self.population_size, 2):
                if i + 1 < self.population_size:
                    child1, child2 = self._crossover(selected[i], selected[i + 1])
                    child1 = self._mutate(child1, param_space)
                    child2 = self._mutate(child2, param_space)
                    new_population.extend([child1, child2])
                else:
                    child = self._mutate(selected[i], param_space)
                    new_population.append(child)
            
            population = new_population[:self.population_size]
            
            print(f"Generation {generation + 1}/{self.n_generations}: Best = {self.best_value:.4f}")
        
        return self.best_params
