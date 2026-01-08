#!/usr/bin/env python3
"""
Progressive Training and Continuous Learning System.
Implements curriculum learning, incremental learning, and online adaptation.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Callable, Any
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import joblib
import time
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
import copy

logger = logging.getLogger(__name__)

@dataclass
class ProgressiveTrainingConfig:
    """Configuration for progressive training."""
    # Curriculum learning
    curriculum_strategy: str = 'difficulty'  # 'difficulty', 'diversity', 'confidence'
    curriculum_stages: int = 5
    stage_overlap: float = 0.1
    
    # Incremental learning
    batch_size: int = 100
    memory_size: int = 1000
    rehearsal_strategy: str = 'random'  # 'random', 'uncertainty', 'diversity'
    
    # Online learning
    learning_rate_decay: float = 0.95
    adaptation_threshold: float = 0.1
    performance_window: int = 100
    
    # Model versioning
    enable_versioning: bool = True
    max_versions: int = 5
    version_threshold: float = 0.02
    
    random_state: int = 42

class CurriculumDesigner(ABC):
    """Abstract base class for curriculum design strategies."""
    
    @abstractmethod
    def design_curriculum(self, X: np.ndarray, y: np.ndarray) -> List[np.ndarray]:
        """Design curriculum stages."""
        pass

class DifficultyBasedCurriculum(CurriculumDesigner):
    """Curriculum based on sample difficulty."""
    
    def __init__(self, n_stages: int = 5):
        self.n_stages = n_stages
    
    def design_curriculum(self, X: np.ndarray, y: np.ndarray) -> List[np.ndarray]:
        """Design curriculum based on sample difficulty."""
        # Train a simple model to estimate difficulty
        simple_model = RandomForestClassifier(n_estimators=50, random_state=42)
        simple_model.fit(X, y)
        
        # Get prediction probabilities as difficulty proxy
        probabilities = simple_model.predict_proba(X)
        confidence_scores = np.max(probabilities, axis=1)
        
        # Lower confidence = higher difficulty
        difficulty_scores = 1 - confidence_scores
        
        # Sort by difficulty (easy to hard)
        sorted_indices = np.argsort(difficulty_scores)
        
        # Create curriculum stages
        stages = []
        stage_size = len(X) // self.n_stages
        
        for i in range(self.n_stages):
            start_idx = i * stage_size
            if i == self.n_stages - 1:
                # Last stage includes all remaining samples
                end_idx = len(X)
            else:
                end_idx = (i + 1) * stage_size
            
            stage_indices = sorted_indices[start_idx:end_idx]
            stages.append(stage_indices)
        
        return stages

class DiversityBasedCurriculum(CurriculumDesigner):
    """Curriculum based on sample diversity."""
    
    def __init__(self, n_stages: int = 5):
        self.n_stages = n_stages
    
    def design_curriculum(self, X: np.ndarray, y: np.ndarray) -> List[np.ndarray]:
        """Design curriculum based on sample diversity."""
        from sklearn.cluster import KMeans
        
        # Cluster samples to find diverse groups
        kmeans = KMeans(n_clusters=self.n_stages * 2, random_state=42)
        cluster_labels = kmeans.fit_predict(X)
        
        # Create stages by selecting diverse samples
        stages = []
        remaining_indices = set(range(len(X)))
        
        for stage in range(self.n_stages):
            stage_indices = []
            target_size = len(X) // self.n_stages
            
            # Select samples from different clusters
            for cluster_id in range(self.n_stages * 2):
                cluster_indices = [i for i in remaining_indices if cluster_labels[i] == cluster_id]
                
                if cluster_indices:
                    # Take a portion from this cluster
                    n_from_cluster = min(len(cluster_indices), target_size // (self.n_stages * 2) + 1)
                    selected = np.random.choice(cluster_indices, n_from_cluster, replace=False)
                    stage_indices.extend(selected)
                    remaining_indices -= set(selected)
                
                if len(stage_indices) >= target_size:
                    break
            
            if stage_indices:
                stages.append(np.array(stage_indices))
        
        # Add any remaining samples to the last stage
        if remaining_indices:
            if stages:
                stages[-1] = np.concatenate([stages[-1], list(remaining_indices)])
            else:
                stages.append(np.array(list(remaining_indices)))
        
        return stages

class MemoryBuffer:
    """Memory buffer for rehearsal in incremental learning."""
    
    def __init__(self, max_size: int, strategy: str = 'random'):
        self.max_size = max_size
        self.strategy = strategy
        self.X_memory = []
        self.y_memory = []
        self.importance_scores = []
    
    def add_samples(self, X: np.ndarray, y: np.ndarray, importance: Optional[np.ndarray] = None):
        """Add samples to memory buffer."""
        if importance is None:
            importance = np.ones(len(X))
        
        # Add new samples
        for i in range(len(X)):
            self.X_memory.append(X[i])
            self.y_memory.append(y[i])
            self.importance_scores.append(importance[i])
        
        # Maintain buffer size
        if len(self.X_memory) > self.max_size:
            self._evict_samples()
    
    def _evict_samples(self):
        """Evict samples based on strategy."""
        n_to_remove = len(self.X_memory) - self.max_size
        
        if self.strategy == 'random':
            # Random eviction
            indices_to_remove = np.random.choice(
                len(self.X_memory), n_to_remove, replace=False
            )
        elif self.strategy == 'importance':
            # Keep most important samples
            importance_array = np.array(self.importance_scores)
            indices_to_remove = np.argsort(importance_array)[:n_to_remove]
        else:
            # Default to random
            indices_to_remove = np.random.choice(
                len(self.X_memory), n_to_remove, replace=False
            )
        
        # Remove samples
        indices_to_remove = sorted(indices_to_remove, reverse=True)
        for idx in indices_to_remove:
            del self.X_memory[idx]
            del self.y_memory[idx]
            del self.importance_scores[idx]
    
    def get_samples(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get all samples from memory."""
        if not self.X_memory:
            return np.array([]), np.array([])
        
        return np.array(self.X_memory), np.array(self.y_memory)
    
    def size(self) -> int:
        """Get current buffer size."""
        return len(self.X_memory)

class ModelVersionManager:
    """Manages multiple model versions for A/B testing."""
    
    def __init__(self, max_versions: int = 5):
        self.max_versions = max_versions
        self.versions = {}
        self.performance_history = {}
        self.current_version = None
    
    def add_version(self, version_id: str, model: Any, performance: float):
        """Add a new model version."""
        self.versions[version_id] = {
            'model': copy.deepcopy(model),
            'performance': performance,
            'created_at': time.time()
        }
        
        self.performance_history[version_id] = [performance]
        
        # Maintain version limit
        if len(self.versions) > self.max_versions:
            self._evict_oldest_version()
        
        # Update current version if this is better
        if (self.current_version is None or 
            performance > self.versions[self.current_version]['performance']):
            self.current_version = version_id
    
    def _evict_oldest_version(self):
        """Remove the oldest version."""
        oldest_version = min(
            self.versions.keys(),
            key=lambda v: self.versions[v]['created_at']
        )
        
        if oldest_version != self.current_version:
            del self.versions[oldest_version]
            del self.performance_history[oldest_version]
    
    def get_current_model(self):
        """Get the current best model."""
        if self.current_version is None:
            return None
        return self.versions[self.current_version]['model']
    
    def get_version_info(self) -> Dict:
        """Get information about all versions."""
        return {
            version_id: {
                'performance': info['performance'],
                'created_at': info['created_at']
            }
            for version_id, info in self.versions.items()
        }

class ProgressiveLearner:
    """
    Progressive learning system with curriculum learning and incremental updates.
    """
    
    def __init__(self, config: ProgressiveTrainingConfig = None):
        """
        Initialize the progressive learner.
        
        Args:
            config: Progressive training configuration
        """
        self.config = config or ProgressiveTrainingConfig()
        
        # Initialize components
        self.curriculum_designer = self._create_curriculum_designer()
        self.memory_buffer = MemoryBuffer(
            max_size=self.config.memory_size,
            strategy=self.config.rehearsal_strategy
        )
        self.version_manager = ModelVersionManager(
            max_versions=self.config.max_versions
        ) if self.config.enable_versioning else None
        
        # Training state
        self.current_model = None
        self.training_history = []
        self.performance_history = []
        self.is_trained = False
        
        logger.info("Initialized ProgressiveLearner")
    
    def _create_curriculum_designer(self) -> CurriculumDesigner:
        """Create curriculum designer based on strategy."""
        if self.config.curriculum_strategy == 'difficulty':
            return DifficultyBasedCurriculum(self.config.curriculum_stages)
        elif self.config.curriculum_strategy == 'diversity':
            return DiversityBasedCurriculum(self.config.curriculum_stages)
        else:
            return DifficultyBasedCurriculum(self.config.curriculum_stages)
    
    def train_progressive(self, 
                         X_train: np.ndarray, 
                         y_train: np.ndarray,
                         X_val: np.ndarray,
                         y_val: np.ndarray) -> Dict:
        """
        Train model using progressive curriculum learning.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            
        Returns:
            Training results
        """
        logger.info("🎓 Starting progressive curriculum training...")
        start_time = time.time()
        
        # Design curriculum
        curriculum_stages = self.curriculum_designer.design_curriculum(X_train, y_train)
        
        logger.info(f"Created curriculum with {len(curriculum_stages)} stages")
        
        # Initialize model
        self.current_model = RandomForestClassifier(
            n_estimators=100,
            class_weight='balanced',
            random_state=self.config.random_state,
            n_jobs=-1
        )
        
        # Progressive training through curriculum stages
        cumulative_indices = []
        
        for stage_idx, stage_indices in enumerate(curriculum_stages):
            logger.info(f"Training stage {stage_idx + 1}/{len(curriculum_stages)} "
                       f"with {len(stage_indices)} samples")
            
            # Add current stage samples
            cumulative_indices.extend(stage_indices)
            
            # Add overlap from previous stage if configured
            if stage_idx > 0 and self.config.stage_overlap > 0:
                prev_stage = curriculum_stages[stage_idx - 1]
                overlap_size = int(len(prev_stage) * self.config.stage_overlap)
                overlap_indices = np.random.choice(prev_stage, overlap_size, replace=False)
                cumulative_indices.extend(overlap_indices)
            
            # Get training data for this stage
            stage_X = X_train[cumulative_indices]
            stage_y = y_train[cumulative_indices]
            
            # Train model
            self.current_model.fit(stage_X, stage_y)
            
            # Evaluate on validation set
            val_predictions = self.current_model.predict(X_val)
            val_accuracy = accuracy_score(y_val, val_predictions)
            val_f1 = f1_score(y_val, val_predictions, average='binary')
            
            # Record stage performance
            stage_performance = {
                'stage': stage_idx + 1,
                'samples_used': len(cumulative_indices),
                'val_accuracy': val_accuracy,
                'val_f1': val_f1
            }
            
            self.training_history.append(stage_performance)
            
            logger.info(f"Stage {stage_idx + 1} - Accuracy: {val_accuracy:.3f}, "
                       f"F1: {val_f1:.3f}")
            
            # Add samples to memory buffer
            self.memory_buffer.add_samples(
                X_train[stage_indices], 
                y_train[stage_indices]
            )
        
        training_time = time.time() - start_time
        self.is_trained = True
        
        # Create version if versioning is enabled
        if self.version_manager:
            final_performance = self.training_history[-1]['val_f1']
            version_id = f"progressive_v{int(time.time())}"
            self.version_manager.add_version(version_id, self.current_model, final_performance)
        
        results = {
            'training_time': training_time,
            'curriculum_stages': len(curriculum_stages),
            'final_performance': self.training_history[-1],
            'training_history': self.training_history,
            'memory_buffer_size': self.memory_buffer.size()
        }
        
        logger.info(f"✅ Progressive training complete in {training_time:.1f}s")
        
        return results
    
    def incremental_update(self, 
                          X_new: np.ndarray, 
                          y_new: np.ndarray,
                          X_val: np.ndarray,
                          y_val: np.ndarray) -> Dict:
        """
        Perform incremental learning with new data.
        
        Args:
            X_new: New training features
            y_new: New training labels
            X_val: Validation features
            y_val: Validation labels
            
        Returns:
            Update results
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before incremental updates")
        
        logger.info(f"🔄 Performing incremental update with {len(X_new)} new samples")
        
        # Get rehearsal samples from memory
        X_memory, y_memory = self.memory_buffer.get_samples()
        
        # Combine new data with rehearsal data
        if len(X_memory) > 0:
            X_combined = np.vstack([X_new, X_memory])
            y_combined = np.hstack([y_new, y_memory])
        else:
            X_combined = X_new
            y_combined = y_new
        
        # Store current performance for comparison
        current_predictions = self.current_model.predict(X_val)
        current_performance = f1_score(y_val, current_predictions, average='binary')
        
        # Create new model (incremental learning)
        updated_model = RandomForestClassifier(
            n_estimators=self.current_model.n_estimators,
            class_weight='balanced',
            random_state=self.config.random_state,
            n_jobs=-1
        )
        
        # Train on combined data
        updated_model.fit(X_combined, y_combined)
        
        # Evaluate updated model
        updated_predictions = updated_model.predict(X_val)
        updated_performance = f1_score(y_val, updated_predictions, average='binary')
        
        # Decide whether to keep the update
        performance_improvement = updated_performance - current_performance
        
        if performance_improvement > -self.config.adaptation_threshold:
            # Accept the update
            self.current_model = updated_model
            
            # Add new samples to memory buffer
            self.memory_buffer.add_samples(X_new, y_new)
            
            # Create new version if significant improvement
            if (self.version_manager and 
                performance_improvement > self.config.version_threshold):
                version_id = f"incremental_v{int(time.time())}"
                self.version_manager.add_version(version_id, updated_model, updated_performance)
            
            update_accepted = True
            logger.info(f"✅ Update accepted - Performance change: {performance_improvement:+.3f}")
        else:
            # Reject the update
            update_accepted = False
            logger.info(f"❌ Update rejected - Performance change: {performance_improvement:+.3f}")
        
        results = {
            'update_accepted': update_accepted,
            'performance_improvement': performance_improvement,
            'current_performance': current_performance,
            'updated_performance': updated_performance,
            'memory_buffer_size': self.memory_buffer.size(),
            'samples_processed': len(X_new)
        }
        
        return results
    
    def online_adaptation(self, 
                         data_stream: List[Tuple[np.ndarray, np.ndarray]],
                         X_val: np.ndarray,
                         y_val: np.ndarray) -> Dict:
        """
        Perform online adaptation with streaming data.
        
        Args:
            data_stream: List of (X_batch, y_batch) tuples
            X_val: Validation features
            y_val: Validation labels
            
        Returns:
            Adaptation results
        """
        logger.info(f"🌊 Starting online adaptation with {len(data_stream)} batches")
        
        adaptation_history = []
        performance_window = []
        
        for batch_idx, (X_batch, y_batch) in enumerate(data_stream):
            # Perform incremental update
            update_results = self.incremental_update(X_batch, y_batch, X_val, y_val)
            
            # Track performance
            current_performance = update_results['current_performance']
            performance_window.append(current_performance)
            
            # Maintain performance window
            if len(performance_window) > self.config.performance_window:
                performance_window.pop(0)
            
            # Record adaptation step
            adaptation_step = {
                'batch': batch_idx + 1,
                'update_accepted': update_results['update_accepted'],
                'performance': current_performance,
                'performance_improvement': update_results['performance_improvement'],
                'avg_performance_window': np.mean(performance_window)
            }
            
            adaptation_history.append(adaptation_step)
            
            if batch_idx % 10 == 0:
                logger.info(f"Batch {batch_idx + 1}: Performance {current_performance:.3f}, "
                           f"Window avg: {np.mean(performance_window):.3f}")
        
        results = {
            'total_batches': len(data_stream),
            'updates_accepted': sum(1 for h in adaptation_history if h['update_accepted']),
            'final_performance': adaptation_history[-1]['performance'],
            'adaptation_history': adaptation_history,
            'performance_trend': np.mean(performance_window)
        }
        
        logger.info(f"✅ Online adaptation complete - "
                   f"Accepted {results['updates_accepted']}/{results['total_batches']} updates")
        
        return results
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions with the current model."""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        return self.current_model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities with the current model."""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        return self.current_model.predict_proba(X)
    
    def get_training_summary(self) -> Dict:
        """Get comprehensive training summary."""
        summary = {
            'is_trained': self.is_trained,
            'curriculum_stages': len(self.training_history) if self.training_history else 0,
            'memory_buffer_size': self.memory_buffer.size(),
            'training_history': self.training_history,
            'config': asdict(self.config)
        }
        
        if self.version_manager:
            summary['version_info'] = self.version_manager.get_version_info()
            summary['current_version'] = self.version_manager.current_version
        
        return summary
    
    def save_learner(self, filepath: str):
        """Save the progressive learner."""
        learner_data = {
            'config': self.config,
            'current_model': self.current_model,
            'memory_buffer': self.memory_buffer,
            'version_manager': self.version_manager,
            'training_history': self.training_history,
            'is_trained': self.is_trained
        }
        
        joblib.dump(learner_data, filepath)
        logger.info(f"Progressive learner saved to {filepath}")
    
    def load_learner(self, filepath: str):
        """Load a saved progressive learner."""
        learner_data = joblib.load(filepath)
        
        self.config = learner_data['config']
        self.current_model = learner_data['current_model']
        self.memory_buffer = learner_data['memory_buffer']
        self.version_manager = learner_data['version_manager']
        self.training_history = learner_data['training_history']
        self.is_trained = learner_data['is_trained']
        
        logger.info(f"Progressive learner loaded from {filepath}")


def main():
    """Example usage of ProgressiveLearner."""
    # Create sample data with varying difficulty
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    
    # Create base dataset
    X, y = make_classification(
        n_samples=2000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        n_clusters_per_class=2,
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
    
    # Configure progressive learning
    config = ProgressiveTrainingConfig(
        curriculum_strategy='difficulty',
        curriculum_stages=4,
        stage_overlap=0.1,
        memory_size=500,
        enable_versioning=True,
        max_versions=3
    )
    
    # Initialize learner
    learner = ProgressiveLearner(config)
    
    # Progressive training
    training_results = learner.train_progressive(X_train, y_train, X_val, y_val)
    
    print("🎓 Progressive Training Results:")
    print(f"Training time: {training_results['training_time']:.1f}s")
    print(f"Curriculum stages: {training_results['curriculum_stages']}")
    print(f"Final performance: {training_results['final_performance']['val_f1']:.3f}")
    
    # Test on held-out data
    test_predictions = learner.predict(X_test)
    test_accuracy = accuracy_score(y_test, test_predictions)
    test_f1 = f1_score(y_test, test_predictions, average='binary')
    
    print(f"\n📊 Test Performance:")
    print(f"Test Accuracy: {test_accuracy:.3f}")
    print(f"Test F1-Score: {test_f1:.3f}")
    
    # Simulate incremental learning with new data
    print(f"\n🔄 Testing incremental learning...")
    
    # Create new data batches
    X_new, y_new = make_classification(
        n_samples=200,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        random_state=123
    )
    
    # Split into batches for online learning
    batch_size = 50
    data_stream = []
    for i in range(0, len(X_new), batch_size):
        X_batch = X_new[i:i+batch_size]
        y_batch = y_new[i:i+batch_size]
        data_stream.append((X_batch, y_batch))
    
    # Online adaptation
    adaptation_results = learner.online_adaptation(data_stream, X_val, y_val)
    
    print(f"Online adaptation results:")
    print(f"Updates accepted: {adaptation_results['updates_accepted']}/{adaptation_results['total_batches']}")
    print(f"Final performance: {adaptation_results['final_performance']:.3f}")
    
    # Get training summary
    summary = learner.get_training_summary()
    print(f"\n📋 Training Summary:")
    print(f"Memory buffer size: {summary['memory_buffer_size']}")
    if 'version_info' in summary:
        print(f"Model versions: {len(summary['version_info'])}")
    
    # Save learner
    learner.save_learner("models/progressive_learner.pkl")
    print(f"\n✅ Progressive learner saved!")


if __name__ == "__main__":
    main()