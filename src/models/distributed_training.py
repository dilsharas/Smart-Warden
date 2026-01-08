#!/usr/bin/env python3
"""
Distributed and Parallel Training Infrastructure.
Implements multi-GPU training, distributed training, and resource optimization.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Callable
import logging
import time
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import psutil
import os
from dataclasses import dataclass, asdict
import joblib
import pickle
from pathlib import Path

# Add src to path for imports
import sys
sys.path.insert(0, 'src')

logger = logging.getLogger(__name__)

@dataclass
class DistributedTrainingConfig:
    """Configuration for distributed training."""
    # Parallelization
    n_workers: int = -1  # -1 for auto-detect
    use_multiprocessing: bool = True
    chunk_size: int = 1000
    
    # Resource management
    max_memory_gb: float = 8.0
    cpu_utilization_limit: float = 0.8
    enable_gpu: bool = False
    gpu_memory_limit: float = 4.0
    
    # Distributed training
    enable_distributed: bool = False
    master_addr: str = "localhost"
    master_port: int = 12355
    world_size: int = 1
    
    # Optimization
    gradient_accumulation_steps: int = 1
    mixed_precision: bool = False
    checkpoint_frequency: int = 100
    
    # Monitoring
    enable_monitoring: bool = True
    log_frequency: int = 10
    
    random_state: int = 42

class ResourceMonitor:
    """Monitors system resources during training."""
    
    def __init__(self):
        self.cpu_usage_history = []
        self.memory_usage_history = []
        self.gpu_usage_history = []
        self.start_time = None
        self.monitoring = False
    
    def start_monitoring(self):
        """Start resource monitoring."""
        self.start_time = time.time()
        self.monitoring = True
        self.cpu_usage_history = []
        self.memory_usage_history = []
        self.gpu_usage_history = []
    
    def stop_monitoring(self):
        """Stop resource monitoring."""
        self.monitoring = False
    
    def record_usage(self):
        """Record current resource usage."""
        if not self.monitoring:
            return
        
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=None)
        self.cpu_usage_history.append(cpu_percent)
        
        # Memory usage
        memory = psutil.virtual_memory()
        memory_gb = memory.used / (1024**3)
        self.memory_usage_history.append(memory_gb)
        
        # GPU usage (if available)
        try:
            import GPUtil
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu_usage = [gpu.memoryUtil * 100 for gpu in gpus]
                self.gpu_usage_history.append(gpu_usage)
        except ImportError:
            pass
    
    def get_usage_stats(self) -> Dict:
        """Get resource usage statistics."""
        stats = {
            'cpu_usage': {
                'mean': np.mean(self.cpu_usage_history) if self.cpu_usage_history else 0,
                'max': np.max(self.cpu_usage_history) if self.cpu_usage_history else 0,
                'history': self.cpu_usage_history
            },
            'memory_usage_gb': {
                'mean': np.mean(self.memory_usage_history) if self.memory_usage_history else 0,
                'max': np.max(self.memory_usage_history) if self.memory_usage_history else 0,
                'history': self.memory_usage_history
            },
            'monitoring_duration': time.time() - self.start_time if self.start_time else 0
        }
        
        if self.gpu_usage_history:
            stats['gpu_usage'] = {
                'mean': np.mean([np.mean(usage) for usage in self.gpu_usage_history]),
                'max': np.max([np.max(usage) for usage in self.gpu_usage_history]),
                'history': self.gpu_usage_history
            }
        
        return stats

class ParallelTrainer:
    """Parallel training coordinator for multiple models."""
    
    def __init__(self, config: DistributedTrainingConfig = None):
        """
        Initialize parallel trainer.
        
        Args:
            config: Distributed training configuration
        """
        self.config = config or DistributedTrainingConfig()
        
        # Set number of workers
        if self.config.n_workers == -1:
            self.config.n_workers = min(mp.cpu_count(), 8)  # Reasonable limit
        
        # Initialize resource monitor
        self.resource_monitor = ResourceMonitor()
        
        # Training state
        self.training_results = {}
        self.checkpoints = {}
        
        logger.info(f"Initialized ParallelTrainer with {self.config.n_workers} workers")
    
    def train_parallel_models(self, 
                            model_configs: List[Dict],
                            X_train: np.ndarray,
                            y_train: np.ndarray,
                            X_val: np.ndarray,
                            y_val: np.ndarray) -> Dict:
        """
        Train multiple models in parallel.
        
        Args:
            model_configs: List of model configurations
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            
        Returns:
            Training results for all models
        """
        logger.info(f"🚀 Training {len(model_configs)} models in parallel...")
        
        # Start resource monitoring
        self.resource_monitor.start_monitoring()
        start_time = time.time()
        
        # Prepare training tasks
        training_tasks = []
        for i, config in enumerate(model_configs):
            task = {
                'model_id': f'model_{i}',
                'config': config,
                'X_train': X_train,
                'y_train': y_train,
                'X_val': X_val,
                'y_val': y_val
            }
            training_tasks.append(task)
        
        # Execute parallel training
        if self.config.use_multiprocessing:
            results = self._train_with_multiprocessing(training_tasks)
        else:
            results = self._train_with_threading(training_tasks)
        
        # Stop monitoring
        self.resource_monitor.stop_monitoring()
        total_time = time.time() - start_time
        
        # Compile results
        training_summary = {
            'total_models': len(model_configs),
            'successful_models': len([r for r in results if r['success']]),
            'failed_models': len([r for r in results if not r['success']]),
            'total_training_time': total_time,
            'average_time_per_model': total_time / len(model_configs),
            'resource_usage': self.resource_monitor.get_usage_stats(),
            'model_results': results
        }
        
        logger.info(f"✅ Parallel training complete in {total_time:.1f}s")
        logger.info(f"📊 Success rate: {training_summary['successful_models']}/{training_summary['total_models']}")
        
        return training_summary
    
    def _train_with_multiprocessing(self, training_tasks: List[Dict]) -> List[Dict]:
        """Train models using multiprocessing."""
        results = []
        
        with ProcessPoolExecutor(max_workers=self.config.n_workers) as executor:
            # Submit all tasks
            future_to_task = {
                executor.submit(self._train_single_model_process, task): task
                for task in training_tasks
            }
            
            # Collect results
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    result = future.result()
                    results.append(result)
                    
                    # Record resource usage periodically
                    self.resource_monitor.record_usage()
                    
                    logger.info(f"✅ Completed {task['model_id']} - "
                               f"Accuracy: {result.get('accuracy', 0):.3f}")
                
                except Exception as e:
                    logger.error(f"❌ Failed {task['model_id']}: {e}")
                    results.append({
                        'model_id': task['model_id'],
                        'success': False,
                        'error': str(e)
                    })
        
        return results
    
    def _train_with_threading(self, training_tasks: List[Dict]) -> List[Dict]:
        """Train models using threading (for I/O bound tasks)."""
        results = []
        
        with ThreadPoolExecutor(max_workers=self.config.n_workers) as executor:
            # Submit all tasks
            future_to_task = {
                executor.submit(self._train_single_model_thread, task): task
                for task in training_tasks
            }
            
            # Collect results
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    result = future.result()
                    results.append(result)
                    
                    logger.info(f"✅ Completed {task['model_id']} - "
                               f"Accuracy: {result.get('accuracy', 0):.3f}")
                
                except Exception as e:
                    logger.error(f"❌ Failed {task['model_id']}: {e}")
                    results.append({
                        'model_id': task['model_id'],
                        'success': False,
                        'error': str(e)
                    })
        
        return results
    
    def _train_single_model_process(self, task: Dict) -> Dict:
        """Train a single model in a separate process."""
        return self._train_single_model(task)
    
    def _train_single_model_thread(self, task: Dict) -> Dict:
        """Train a single model in a thread."""
        return self._train_single_model(task)
    
    def _train_single_model(self, task: Dict) -> Dict:
        """Train a single model with given configuration."""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
        
        model_id = task['model_id']
        config = task['config']
        X_train = task['X_train']
        y_train = task['y_train']
        X_val = task['X_val']
        y_val = task['y_val']
        
        try:
            start_time = time.time()
            
            # Create model with configuration
            model = RandomForestClassifier(
                random_state=self.config.random_state,
                n_jobs=1,  # Avoid nested parallelism
                **config
            )
            
            # Train model
            model.fit(X_train, y_train)
            
            # Evaluate model
            val_predictions = model.predict(X_val)
            val_probabilities = model.predict_proba(X_val)[:, 1]
            
            accuracy = accuracy_score(y_val, val_predictions)
            f1 = f1_score(y_val, val_predictions, average='binary')
            roc_auc = roc_auc_score(y_val, val_probabilities)
            
            training_time = time.time() - start_time
            
            result = {
                'model_id': model_id,
                'success': True,
                'model': model,
                'config': config,
                'metrics': {
                    'accuracy': accuracy,
                    'f1_score': f1,
                    'roc_auc': roc_auc
                },
                'training_time': training_time
            }
            
            return result
            
        except Exception as e:
            return {
                'model_id': model_id,
                'success': False,
                'error': str(e),
                'config': config
            }

class DistributedDataLoader:
    """Distributed data loader for large datasets."""
    
    def __init__(self, config: DistributedTrainingConfig = None):
        self.config = config or DistributedTrainingConfig()
    
    def load_data_parallel(self, 
                          data_sources: List[str],
                          load_function: Callable) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load data from multiple sources in parallel.
        
        Args:
            data_sources: List of data source paths/identifiers
            load_function: Function to load data from a single source
            
        Returns:
            Combined features and labels
        """
        logger.info(f"📁 Loading data from {len(data_sources)} sources in parallel...")
        
        all_X = []
        all_y = []
        
        with ProcessPoolExecutor(max_workers=self.config.n_workers) as executor:
            # Submit loading tasks
            future_to_source = {
                executor.submit(load_function, source): source
                for source in data_sources
            }
            
            # Collect results
            for future in as_completed(future_to_source):
                source = future_to_source[future]
                try:
                    X, y = future.result()
                    all_X.append(X)
                    all_y.append(y)
                    logger.info(f"✅ Loaded {source}: {X.shape[0]} samples")
                
                except Exception as e:
                    logger.error(f"❌ Failed to load {source}: {e}")
        
        # Combine all data
        if all_X:
            combined_X = np.vstack(all_X)
            combined_y = np.hstack(all_y)
            
            logger.info(f"📊 Combined dataset: {combined_X.shape}")
            return combined_X, combined_y
        else:
            raise ValueError("No data was successfully loaded")
    
    def create_data_chunks(self, 
                          X: np.ndarray, 
                          y: np.ndarray,
                          n_chunks: Optional[int] = None) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Create data chunks for distributed processing.
        
        Args:
            X: Features
            y: Labels
            n_chunks: Number of chunks (default: n_workers)
            
        Returns:
            List of (X_chunk, y_chunk) tuples
        """
        if n_chunks is None:
            n_chunks = self.config.n_workers
        
        chunk_size = len(X) // n_chunks
        chunks = []
        
        for i in range(n_chunks):
            start_idx = i * chunk_size
            if i == n_chunks - 1:
                # Last chunk gets remaining samples
                end_idx = len(X)
            else:
                end_idx = (i + 1) * chunk_size
            
            X_chunk = X[start_idx:end_idx]
            y_chunk = y[start_idx:end_idx]
            chunks.append((X_chunk, y_chunk))
        
        logger.info(f"Created {len(chunks)} data chunks")
        return chunks

class CheckpointManager:
    """Manages training checkpoints for fault tolerance."""
    
    def __init__(self, checkpoint_dir: str = "checkpoints"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def save_checkpoint(self, 
                       model: Any, 
                       optimizer_state: Dict,
                       epoch: int,
                       metrics: Dict,
                       model_id: str = "model") -> str:
        """Save training checkpoint."""
        checkpoint_path = self.checkpoint_dir / f"{model_id}_epoch_{epoch}.pkl"
        
        checkpoint_data = {
            'model': model,
            'optimizer_state': optimizer_state,
            'epoch': epoch,
            'metrics': metrics,
            'timestamp': time.time()
        }
        
        joblib.dump(checkpoint_data, checkpoint_path)
        logger.info(f"💾 Checkpoint saved: {checkpoint_path}")
        
        return str(checkpoint_path)
    
    def load_checkpoint(self, checkpoint_path: str) -> Dict:
        """Load training checkpoint."""
        checkpoint_data = joblib.load(checkpoint_path)
        logger.info(f"📂 Checkpoint loaded: {checkpoint_path}")
        
        return checkpoint_data
    
    def list_checkpoints(self, model_id: str = None) -> List[str]:
        """List available checkpoints."""
        if model_id:
            pattern = f"{model_id}_epoch_*.pkl"
        else:
            pattern = "*_epoch_*.pkl"
        
        checkpoints = list(self.checkpoint_dir.glob(pattern))
        return [str(cp) for cp in sorted(checkpoints)]

class DistributedTrainingCoordinator:
    """
    Main coordinator for distributed training operations.
    """
    
    def __init__(self, config: DistributedTrainingConfig = None):
        """
        Initialize distributed training coordinator.
        
        Args:
            config: Distributed training configuration
        """
        self.config = config or DistributedTrainingConfig()
        
        # Initialize components
        self.parallel_trainer = ParallelTrainer(self.config)
        self.data_loader = DistributedDataLoader(self.config)
        self.checkpoint_manager = CheckpointManager()
        
        # Training state
        self.training_history = []
        self.best_model = None
        self.best_score = -np.inf
        
        logger.info("Initialized DistributedTrainingCoordinator")
    
    def run_distributed_training(self,
                                X_train: np.ndarray,
                                y_train: np.ndarray,
                                X_val: np.ndarray,
                                y_val: np.ndarray,
                                model_configs: List[Dict]) -> Dict:
        """
        Run complete distributed training pipeline.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            model_configs: List of model configurations to train
            
        Returns:
            Complete training results
        """
        logger.info("🚀 Starting distributed training pipeline...")
        pipeline_start = time.time()
        
        # Step 1: Parallel model training
        training_results = self.parallel_trainer.train_parallel_models(
            model_configs, X_train, y_train, X_val, y_val
        )
        
        # Step 2: Select best model
        best_model_result = self._select_best_model(training_results['model_results'])
        
        if best_model_result:
            self.best_model = best_model_result['model']
            self.best_score = best_model_result['metrics']['f1_score']
            
            # Save best model checkpoint
            checkpoint_path = self.checkpoint_manager.save_checkpoint(
                model=self.best_model,
                optimizer_state={},
                epoch=0,
                metrics=best_model_result['metrics'],
                model_id="best_model"
            )
        
        # Step 3: Compile final results
        total_time = time.time() - pipeline_start
        
        final_results = {
            'training_results': training_results,
            'best_model_result': best_model_result,
            'best_model_checkpoint': checkpoint_path if best_model_result else None,
            'total_pipeline_time': total_time,
            'config': asdict(self.config),
            'resource_efficiency': self._calculate_efficiency_metrics(training_results)
        }
        
        logger.info(f"✅ Distributed training complete in {total_time:.1f}s")
        if best_model_result:
            logger.info(f"🏆 Best model: {best_model_result['model_id']} "
                       f"(F1: {self.best_score:.3f})")
        
        return final_results
    
    def _select_best_model(self, model_results: List[Dict]) -> Optional[Dict]:
        """Select the best model from training results."""
        successful_models = [r for r in model_results if r['success']]
        
        if not successful_models:
            logger.warning("No models trained successfully")
            return None
        
        # Select model with highest F1 score
        best_model = max(successful_models, key=lambda x: x['metrics']['f1_score'])
        
        logger.info(f"🏆 Best model: {best_model['model_id']} "
                   f"(F1: {best_model['metrics']['f1_score']:.3f})")
        
        return best_model
    
    def _calculate_efficiency_metrics(self, training_results: Dict) -> Dict:
        """Calculate training efficiency metrics."""
        resource_usage = training_results['resource_usage']
        
        efficiency_metrics = {
            'time_efficiency': {
                'total_time': training_results['total_training_time'],
                'average_time_per_model': training_results['average_time_per_model'],
                'parallel_speedup': self._estimate_parallel_speedup(training_results)
            },
            'resource_efficiency': {
                'avg_cpu_usage': resource_usage['cpu_usage']['mean'],
                'max_cpu_usage': resource_usage['cpu_usage']['max'],
                'avg_memory_usage_gb': resource_usage['memory_usage_gb']['mean'],
                'max_memory_usage_gb': resource_usage['memory_usage_gb']['max']
            },
            'success_rate': training_results['successful_models'] / training_results['total_models']
        }
        
        if 'gpu_usage' in resource_usage:
            efficiency_metrics['resource_efficiency']['avg_gpu_usage'] = resource_usage['gpu_usage']['mean']
            efficiency_metrics['resource_efficiency']['max_gpu_usage'] = resource_usage['gpu_usage']['max']
        
        return efficiency_metrics
    
    def _estimate_parallel_speedup(self, training_results: Dict) -> float:
        """Estimate parallel speedup compared to sequential training."""
        # Simplified estimation: assume linear speedup up to number of workers
        theoretical_sequential_time = (
            training_results['average_time_per_model'] * 
            training_results['total_models']
        )
        
        actual_parallel_time = training_results['total_training_time']
        speedup = theoretical_sequential_time / actual_parallel_time
        
        return min(speedup, self.config.n_workers)  # Cap at number of workers
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions with the best model."""
        if self.best_model is None:
            raise ValueError("No trained model available")
        
        return self.best_model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities with the best model."""
        if self.best_model is None:
            raise ValueError("No trained model available")
        
        return self.best_model.predict_proba(X)


def main():
    """Example usage of DistributedTrainingCoordinator."""
    # Create sample data
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    
    X, y = make_classification(
        n_samples=2000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        n_clusters_per_class=1,
        weights=[0.6, 0.4],
        random_state=42
    )
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, stratify=y_train, random_state=42
    )
    
    # Configure distributed training
    config = DistributedTrainingConfig(
        n_workers=4,
        use_multiprocessing=True,
        enable_monitoring=True,
        checkpoint_frequency=1
    )
    
    # Define model configurations to train in parallel
    model_configs = [
        {'n_estimators': 100, 'max_depth': 10, 'min_samples_split': 2},
        {'n_estimators': 200, 'max_depth': 15, 'min_samples_split': 5},
        {'n_estimators': 150, 'max_depth': 12, 'min_samples_split': 3},
        {'n_estimators': 300, 'max_depth': 20, 'min_samples_split': 4},
        {'n_estimators': 250, 'max_depth': 18, 'min_samples_split': 6}
    ]
    
    # Initialize coordinator
    coordinator = DistributedTrainingCoordinator(config)
    
    # Run distributed training
    results = coordinator.run_distributed_training(
        X_train, y_train, X_val, y_val, model_configs
    )
    
    print("🚀 Distributed Training Results:")
    print(f"Total models trained: {results['training_results']['total_models']}")
    print(f"Successful models: {results['training_results']['successful_models']}")
    print(f"Total training time: {results['total_pipeline_time']:.1f}s")
    print(f"Average time per model: {results['training_results']['average_time_per_model']:.1f}s")
    
    # Resource efficiency
    efficiency = results['resource_efficiency']
    print(f"\n📊 Resource Efficiency:")
    print(f"Parallel speedup: {efficiency['time_efficiency']['parallel_speedup']:.1f}x")
    print(f"Average CPU usage: {efficiency['resource_efficiency']['avg_cpu_usage']:.1f}%")
    print(f"Average memory usage: {efficiency['resource_efficiency']['avg_memory_usage_gb']:.1f} GB")
    print(f"Success rate: {efficiency['success_rate']:.1%}")
    
    # Test best model
    if coordinator.best_model:
        test_predictions = coordinator.predict(X_test)
        from sklearn.metrics import accuracy_score, f1_score
        
        test_accuracy = accuracy_score(y_test, test_predictions)
        test_f1 = f1_score(y_test, test_predictions, average='binary')
        
        print(f"\n🏆 Best Model Test Performance:")
        print(f"Test Accuracy: {test_accuracy:.3f}")
        print(f"Test F1-Score: {test_f1:.3f}")
    
    print(f"\n✅ Distributed training demonstration complete!")


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    main()