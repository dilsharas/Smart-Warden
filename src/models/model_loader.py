"""
Model loader for trained AI models in Smart Contract AI Analyzer.
"""

import joblib
import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

class ModelLoader:
    """Loads and manages trained AI models."""
    
    def __init__(self, models_dir: str = "models"):
        self.models_dir = Path(models_dir)
        self.models = {}
        self.metadata = None
        self.load_metadata()
    
    def load_metadata(self):
        """Load model metadata."""
        metadata_path = self.models_dir / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
            logger.info("Model metadata loaded")
        else:
            logger.warning("No model metadata found")
    
    def load_binary_model(self):
        """Load binary classification model."""
        model_path = self.models_dir / "binary_classifier.joblib"
        if model_path.exists():
            try:
                model = joblib.load(model_path)
                self.models['binary'] = model
                logger.info("Binary classifier loaded successfully")
                return model
            except Exception as e:
                logger.error(f"Error loading binary model: {e}")
                return None
        else:
            logger.error("Binary classifier not found")
            return None
    
    def load_multiclass_model(self):
        """Load multi-class classification model."""
        model_path = self.models_dir / "multiclass_classifier.joblib"
        if model_path.exists():
            try:
                model_data = joblib.load(model_path)
                self.models['multiclass'] = model_data
                logger.info("Multi-class classifier loaded successfully")
                return model_data
            except Exception as e:
                logger.error(f"Error loading multi-class model: {e}")
                return None
        else:
            logger.error("Multi-class classifier not found")
            return None
    
    def load_all_models(self):
        """Load all available models."""
        logger.info("Loading all available models...")
        
        binary_model = self.load_binary_model()
        multiclass_model = self.load_multiclass_model()
        
        loaded_count = 0
        if binary_model:
            loaded_count += 1
        if multiclass_model:
            loaded_count += 1
            
        logger.info(f"Loaded {loaded_count} models successfully")
        return {
            'binary': binary_model,
            'multiclass': multiclass_model,
            'count': loaded_count
        }
    
    def predict_vulnerability(self, features: Dict[str, float]):
        """Make predictions using loaded models."""
        if not self.models:
            self.load_all_models()
        
        results = {
            'available': False,
            'binary_prediction': None,
            'multiclass_prediction': None
        }
        
        # Convert features to DataFrame
        features_df = pd.DataFrame([features])
        
        # Binary prediction
        if 'binary' in self.models and self.models['binary']:
            try:
                binary_model = self.models['binary']
                predictions, probabilities = binary_model.predict(features_df)
                
                results['binary_prediction'] = {
                    'is_vulnerable': bool(predictions[0]),
                    'confidence': float(max(probabilities[0])),
                    'vulnerability_probability': float(probabilities[0][1]) if len(probabilities[0]) > 1 else 0.5
                }
                results['available'] = True
            except Exception as e:
                logger.error(f"Binary prediction error: {e}")
        
        # Multi-class prediction
        if 'multiclass' in self.models and self.models['multiclass']:
            try:
                multiclass_data = self.models['multiclass']
                model = multiclass_data['model']
                label_encoder = multiclass_data['label_encoder']
                
                predictions = model.predict(features_df)
                probabilities = model.predict_proba(features_df)
                
                predicted_class = label_encoder.inverse_transform(predictions)[0]
                confidence = float(max(probabilities[0]))
                
                # Get all class probabilities
                class_probabilities = {}
                for i, class_name in enumerate(label_encoder.classes_):
                    class_probabilities[class_name] = float(probabilities[0][i])
                
                results['multiclass_prediction'] = {
                    'vulnerability_type': predicted_class,
                    'confidence': confidence,
                    'class_probabilities': class_probabilities
                }
                results['available'] = True
            except Exception as e:
                logger.error(f"Multi-class prediction error: {e}")
        
        return results
    
    def get_model_info(self):
        """Get information about loaded models."""
        info = {
            'models_loaded': len(self.models),
            'available_models': list(self.models.keys()),
            'metadata': self.metadata
        }
        
        if 'binary' in self.models:
            info['binary_model'] = {
                'type': 'RandomForestVulnerabilityDetector',
                'status': 'loaded'
            }
        
        if 'multiclass' in self.models:
            multiclass_data = self.models['multiclass']
            info['multiclass_model'] = {
                'type': 'RandomForestClassifier',
                'status': 'loaded',
                'classes': list(multiclass_data['label_encoder'].classes_) if 'label_encoder' in multiclass_data else []
            }
        
        return info


def predict_vulnerability(features: Dict[str, float]):
    """Standalone function for vulnerability prediction."""
    loader = ModelLoader()
    return loader.predict_vulnerability(features)