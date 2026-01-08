#!/usr/bin/env python3
"""
Bayesian Optimization for Automated Hyperparameter Tuning.
Implements efficient hyperparameter optimization with Bayesian methods.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Callable, Any
import logging
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
import time
from dataclasses import dataclass
from scipy.optimize import minimize
from scipy.stats import norm
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

@dataclass
class BayesianOptConfig:
    """Configuration for Bayesian optimization."""
    n_initial_points: int = 10
    n_iterations: int = 50
    acquisition_function: str = 'ei'  # 'ei', 'pi', 'ucb'
    xi: float = 0.01  # Exploration parameter
    kappa: float = 2.576  # UCB parameter
    random_state: int = 42
    cv_folds: int = 3
    scoring: str = 'f1'
    n_jobs: int = -1

class AcquisitionFunction:
    """Acquisition functions for Bayesian optimization."""
    
    @staticmethod
    def expected_improvement(X: np.ndarray, 
                           gp: GaussianProcessRegressor, 
                           y_best: float, 
                           xi: float = 0.01) -> np.ndarray:
        """
        Expected Improvement acquisition function.
        
        Args:
            X: Points to evaluate
            gp: Fitted Gaussian Process
            y_best: Best observed value so far
            xi: Exploration parameter
            
        Returns:
            Expected improvement values
        """
        mu, sigma = gp.predict(X, return_std=True)
        sigma = sigma.reshape(-1, 1)
        
        with np.errstate(divide='warn'):
            imp = mu - y_best - xi
            Z = imp / sigma
            ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
            ei[sigma == 0.0] = 0.0
        
        return ei.flatten()
    
    @staticmethod
    def probability_of_improvement(X: np.ndarray, 
                                 gp: GaussianProcessRegressor, 
                                 y_best: float, 
                                 xi: float = 0.01) -> np.ndarray:
        """
        Probability of Improvement acquisition function.
        
        Args:
            X: Points to evaluate
            gp: Fitted Gaussian Process
            y_best: Best observed value so far
            xi: Exploration parameter
            
        Returns:
            Probability of improvement values
        """
        mu, sigma = gp.predict(X, return_std=True)
        
        with np.errstate(divide='warn'):
            Z = (mu - y_best - xi) / sigma
            pi = norm.cdf(Z)
        
        return pi
    
    @staticmethod
    def upper_confidence_bound(X: np.ndarray, 
                             gp: GaussianProcessRegressor, 
                             kappa: float = 2.576) -> np.ndarray:
        """
        Upper Confidence Bound acquisition function.
        
        Args:
            X: Points to evaluate
            gp: Fitted Gaussian Process
            kappa: Exploration parameter
            
        Returns:
            UCB values
        """
        mu, sigma = gp.predict(X, return_std=True)
        return mu + kappa * sigma

class ParameterSpace:
    """Defines the hyperparameter search space."""
    
    def __init__(self):
        self.parameters = {}
        self.bounds = []
        self.param_names = []
    
    def add_parameter(self, name: str, param_type: str, bounds: Tuple, **kwargs):
        """
        Add a parameter to the search space.
        
        Args:
            name: Parameter name
            param_type: 'continuous', 'integer', 'categorical'
            bounds: (min, max) for continuous/integer, list of values for categorical
            **kwargs: Additional parameter options
        """
        self.parameters[name] = {
            'type': param_type,
            'bounds': bounds,
            **kwargs
        }
        
        if param_type in ['continuous', 'integer']:
            self.bounds.append(bounds)
            self.param_names.append(name)
        else:
            # For categorical, we'll handle separately
            pass
    
    def sample_random(self, n_samples: int) -> np.ndarray:
        """Sample random points from the parameter space."""
        samples = []
        
        for _ in range(n_samples):
            sample = []
            for name in self.param_names:
                param = self.parameters[name]
                if param['type'] == 'continuous':
                    value = np.random.uniform(param['bounds'][0], param['bounds'][1])
                elif param['type'] == 'integer':
                    value = np.random.randint(param['bounds'][0], param['bounds'][1] + 1)
                sample.append(value)
            samples.append(sample)
        
        return np.array(samples)
    
    def convert_to_params(self, X: np.ndarray) -> Dict:
        """Convert optimization vector to parameter dictionary."""
        params = {}
        
        for i, name in enumerate(self.param_names):
            param = self.parameters[name]
            value = X[i]
            
            if param['type'] == 'integer':
                value = int(round(value))
            
            params[name] = value
        
        return params

class BayesianOptimizer:
    """
    Bayesian optimization for hyperparameter tuning.
    """
    
    def __init__(self, config: BayesianOptConfig = None):
        """
        Initialize the Bayesian optimizer.
        
        Args:
            config: Optimization configuration
        """
        self.config = config or BayesianOptConfig()
        self.parameter_space = None
        self.objective_function = None
        self.gp = None
        
        # Optimization history
        self.X_observed = []
        self.y_observed = []
        self.best_params = None
        self.best_score = -np.inf
        self.optimization_history = []
        
        # Initialize Gaussian Process
        kernel = Matern(length_scale=1.0, nu=2.5)
        self.gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-6,
            normalize_y=True,
            n_restarts_optimizer=5,
            random_state=self.config.random_state
        )
        
        logger.info("Initialized BayesianOptimizer")
    
    def set_parameter_space(self, parameter_space: ParameterSpace):
        """Set the parameter search space."""
        self.parameter_space = parameter_space
    
    def set_objective_function(self, objective_func: Callable):
        """Set the objective function to optimize."""
        self.objective_function = objective_func
    
    def optimize(self, 
                X_train: np.ndarray, 
                y_train: np.ndarray,
                X_val: Optional[np.ndarray] = None,
                y_val: Optional[np.ndarray] = None) -> Dict:
        """
        Run Bayesian optimization.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            
        Returns:
            Optimization results
        """
        logger.info(f"🔧 Starting Bayesian optimization with {self.config.n_iterations} iterations...")
        
        if self.parameter_space is None:
            raise ValueError("Parameter space must be set before optimization")
        
        start_time = time.time()
        
        # Create objective function wrapper
        def objective_wrapper(X_params):
            params = self.parameter_space.convert_to_params(X_params)
            return self._evaluate_objective(params, X_train, y_train, X_val, y_val)
        
        # Initial random sampling
        logger.info(f"🎲 Initial random sampling: {self.config.n_initial_points} points")
        X_init = self.parameter_space.sample_random(self.config.n_initial_points)
        
        for X_params in X_init:
            score = objective_wrapper(X_params)
            self.X_observed.append(X_params)
            self.y_observed.append(score)
            
            if score > self.best_score:
                self.best_score = score
                self.best_params = self.parameter_space.convert_to_params(X_params)
        
        # Bayesian optimization iterations
        for iteration in range(self.config.n_iterations):
            logger.info(f"Iteration {iteration + 1}/{self.config.n_iterations}")
            
            # Fit Gaussian Process
            X_observed_array = np.array(self.X_observed)
            y_observed_array = np.array(self.y_observed)
            
            self.gp.fit(X_observed_array, y_observed_array)
            
            # Find next point to evaluate
            next_X = self._find_next_point()
            
            # Evaluate objective
            score = objective_wrapper(next_X)
            
            # Update observations
            self.X_observed.append(next_X)
            self.y_observed.append(score)
            
            # Update best
            if score > self.best_score:
                self.best_score = score
                self.best_params = self.parameter_space.convert_to_params(next_X)
                logger.info(f"🎯 New best score: {self.best_score:.4f}")
            
            # Record history
            self.optimization_history.append({
                'iteration': iteration + 1,
                'score': score,
                'best_score': self.best_score,
                'params': self.parameter_space.convert_to_params(next_X)
            })
        
        optimization_time = time.time() - start_time
        
        results = {
            'best_params': self.best_params,
            'best_score': self.best_score,
            'optimization_history': self.optimization_history,
            'total_evaluations': len(self.X_observed),
            'optimization_time': optimization_time,
            'convergence_iteration': self._find_convergence_iteration()
        }
        
        logger.info(f"✅ Optimization complete in {optimization_time:.1f}s")
        logger.info(f"🏆 Best score: {self.best_score:.4f}")
        logger.info(f"🎯 Best params: {self.best_params}")
        
        return results
    
    def _find_next_point(self) -> np.ndarray:
        """Find the next point to evaluate using acquisition function."""
        # Define acquisition function
        if self.config.acquisition_function == 'ei':
            acq_func = lambda X: -AcquisitionFunction.expected_improvement(
                X.reshape(1, -1), self.gp, self.best_score, self.config.xi
            )[0]
        elif self.config.acquisition_function == 'pi':
            acq_func = lambda X: -AcquisitionFunction.probability_of_improvement(
                X.reshape(1, -1), self.gp, self.best_score, self.config.xi
            )[0]
        elif self.config.acquisition_function == 'ucb':
            acq_func = lambda X: -AcquisitionFunction.upper_confidence_bound(
                X.reshape(1, -1), self.gp, self.config.kappa
            )[0]
        else:
            raise ValueError(f"Unknown acquisition function: {self.config.acquisition_function}")
        
        # Optimize acquisition function
        best_acq = np.inf
        best_x = None
        
        # Multiple random starts for optimization
        for _ in range(10):
            x0 = self.parameter_space.sample_random(1)[0]
            
            try:
                result = minimize(
                    acq_func,
                    x0,
                    bounds=self.parameter_space.bounds,
                    method='L-BFGS-B'
                )
                
                if result.fun < best_acq:
                    best_acq = result.fun
                    best_x = result.x
            except:
                continue
        
        if best_x is None:
            # Fallback to random sampling
            best_x = self.parameter_space.sample_random(1)[0]
        
        return best_x
    
    def _evaluate_objective(self, 
                          params: Dict, 
                          X_train: np.ndarray, 
                          y_train: np.ndarray,
                          X_val: Optional[np.ndarray] = None,
                          y_val: Optional[np.ndarray] = None) -> float:
        """Evaluate the objective function with given parameters."""
        try:
            # Create model with parameters
            model = RandomForestClassifier(
                random_state=self.config.random_state,
                n_jobs=1,  # Avoid nested parallelism
                **params
            )
            
            # Evaluate using validation set or cross-validation
            if X_val is not None and y_val is not None:
                model.fit(X_train, y_train)
                predictions = model.predict(X_val)
                
                if self.config.scoring == 'accuracy':
                    score = accuracy_score(y_val, predictions)
                elif self.config.scoring == 'f1':
                    score = f1_score(y_val, predictions, average='binary')
                else:
                    score = accuracy_score(y_val, predictions)
            else:
                # Use cross-validation
                cv_scores = cross_val_score(
                    model, X_train, y_train,
                    cv=StratifiedKFold(n_splits=self.config.cv_folds, shuffle=True, random_state=42),
                    scoring=self.config.scoring,
                    n_jobs=1
                )
                score = cv_scores.mean()
            
            return score
            
        except Exception as e:
            logger.warning(f"Evaluation failed with params {params}: {e}")
            return -1.0  # Return very low score for failed evaluations
    
    def _find_convergence_iteration(self) -> Optional[int]:
        """Find the iteration where optimization converged."""
        if len(self.optimization_history) < 5:
            return None
        
        # Look for convergence (no improvement in last 5 iterations)
        best_scores = [h['best_score'] for h in self.optimization_history]
        
        for i in range(5, len(best_scores)):
            if best_scores[i] == best_scores[i-5]:
                return i - 4  # Return the iteration where convergence started
        
        return None
    
    def plot_optimization_history(self, save_path: Optional[str] = None):
        """Plot the optimization history."""
        import matplotlib.pyplot as plt
        
        if not self.optimization_history:
            logger.warning("No optimization history to plot")
            return
        
        iterations = [h['iteration'] for h in self.optimization_history]
        scores = [h['score'] for h in self.optimization_history]
        best_scores = [h['best_score'] for h in self.optimization_history]
        
        plt.figure(figsize=(12, 5))
        
        # Plot scores
        plt.subplot(1, 2, 1)
        plt.plot(iterations, scores, 'b-o', alpha=0.6, label='Evaluation Score')
        plt.plot(iterations, best_scores, 'r-', linewidth=2, label='Best Score')
        plt.xlabel('Iteration')
        plt.ylabel('Score')
        plt.title('Bayesian Optimization Progress')
        plt.legend()
        plt.grid(True)
        
        # Plot acquisition function values (if available)
        plt.subplot(1, 2, 2)
        improvements = [scores[i] - scores[i-1] if i > 0 else 0 for i in range(len(scores))]
        plt.bar(iterations, improvements, alpha=0.6, color='green')
        plt.xlabel('Iteration')
        plt.ylabel('Score Improvement')
        plt.title('Score Improvement per Iteration')
        plt.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Optimization history plot saved to {save_path}")
        
        plt.show()

def create_random_forest_parameter_space() -> ParameterSpace:
    """Create parameter space for Random Forest optimization."""
    space = ParameterSpace()
    
    space.add_parameter('n_estimators', 'integer', (50, 500))
    space.add_parameter('max_depth', 'integer', (5, 30))
    space.add_parameter('min_samples_split', 'integer', (2, 20))
    space.add_parameter('min_samples_leaf', 'integer', (1, 10))
    space.add_parameter('max_features', 'continuous', (0.1, 1.0))
    
    return space

def main():
    """Example usage of BayesianOptimizer."""
    # Create sample data
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        n_clusters_per_class=1,
        weights=[0.7, 0.3],
        random_state=42
    )
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, stratify=y_train, random_state=42
    )
    
    # Configure optimization
    config = BayesianOptConfig(
        n_initial_points=5,
        n_iterations=20,
        acquisition_function='ei',
        scoring='f1',
        cv_folds=3
    )
    
    # Initialize optimizer
    optimizer = BayesianOptimizer(config)
    
    # Set parameter space
    parameter_space = create_random_forest_parameter_space()
    optimizer.set_parameter_space(parameter_space)
    
    # Run optimization
    results = optimizer.optimize(X_train, y_train, X_val, y_val)
    
    print("🔧 Bayesian Optimization Results:")
    print(f"Best Score: {results['best_score']:.4f}")
    print(f"Best Parameters: {results['best_params']}")
    print(f"Total Evaluations: {results['total_evaluations']}")
    print(f"Optimization Time: {results['optimization_time']:.1f}s")
    
    # Test optimized model
    optimized_model = RandomForestClassifier(**results['best_params'], random_state=42)
    optimized_model.fit(X_train, y_train)
    
    test_predictions = optimized_model.predict(X_test)
    test_accuracy = accuracy_score(y_test, test_predictions)
    test_f1 = f1_score(y_test, test_predictions, average='binary')
    
    print(f"\n📊 Test Performance:")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test F1-Score: {test_f1:.4f}")
    
    # Plot optimization history
    optimizer.plot_optimization_history("bayesian_optimization_history.png")
    
    print("\n✅ Bayesian optimization complete!")


if __name__ == "__main__":
    main()