"""
API Integration Module

Integrates fraud detection with the Smart Contract AI Analyzer API.
"""

import logging
from typing import Dict, Optional

from flask import Blueprint, request, jsonify
from werkzeug.exceptions import BadRequest

from .data_loader import TransactionDataLoader
from .preprocessor import DataPreprocessor
from .feature_extractor import TransactionFeatureExtractor
from .fraud_detector import FraudDetector
from .evaluator import ModelEvaluator
from .visualization import VisualizationEngine
from .report_generator import ReportGenerator

logger = logging.getLogger(__name__)


class FraudDetectionAPI:
    """
    API endpoints for fraud detection module.
    
    Integrates with Flask application for REST API access.
    """

    def __init__(self):
        """Initialize the API."""
        self.blueprint = Blueprint('fraud_detection', __name__, url_prefix='/api/fraud-detection')
        self.detector = None
        self.preprocessor = None
        self.extractor = None
        self.evaluator = None
        self.visualizer = None
        self.reporter = None
        self._register_routes()

    def _register_routes(self):
        """Register API routes."""
        self.blueprint.route('/analyze', methods=['POST'])(self.analyze_transactions)
        self.blueprint.route('/models/status', methods=['GET'])(self.get_model_status)
        self.blueprint.route('/health', methods=['GET'])(self.health_check)

    def initialize_models(self):
        """Initialize fraud detection models."""
        logger.info("Initializing fraud detection models...")
        
        self.detector = FraudDetector()
        self.preprocessor = DataPreprocessor()
        self.extractor = TransactionFeatureExtractor()
        self.evaluator = ModelEvaluator()
        self.visualizer = VisualizationEngine()
        self.reporter = ReportGenerator()
        
        # Try to load pre-trained model
        import os
        model_path = "results/fraud_detection_example/fraud_detector_model.pkl"
        if os.path.exists(model_path):
            try:
                self.detector.load_model(model_path)
                logger.info(f"Pre-trained model loaded from {model_path}")
            except Exception as e:
                logger.warning(f"Could not load pre-trained model: {e}")
        else:
            logger.info(f"No pre-trained model found at {model_path}")
        
        logger.info("Fraud detection models initialized successfully")

    def analyze_transactions(self):
        """
        Analyze transactions for fraud.
        
        Expected JSON payload:
        {
            "transactions": [
                {
                    "sender": "0x...",
                    "receiver": "0x...",
                    "value": 1.5,
                    "gas_used": 21000,
                    "timestamp": 1000000
                },
                ...
            ]
        }
        
        Returns:
            JSON response with fraud predictions
        """
        try:
            # Check if model is trained
            if not self.detector.trained:
                return jsonify({
                    "error": "Model not trained",
                    "message": "Fraud detection model must be trained first. Run: python examples/fraud_detection_example.py"
                }), 400
            
            # Validate request
            if not request.is_json:
                return jsonify({"error": "Request must be JSON"}), 400
            
            data = request.get_json()
            
            if 'transactions' not in data:
                return jsonify({"error": "Missing 'transactions' field"}), 400
            
            transactions = data['transactions']
            
            if not isinstance(transactions, list) or len(transactions) == 0:
                return jsonify({"error": "Transactions must be a non-empty list"}), 400
            
            # Convert to DataFrame
            import pandas as pd
            df = pd.DataFrame(transactions)
            
            # Validate required fields
            required_fields = ['sender', 'receiver', 'value', 'gas_used', 'timestamp']
            missing_fields = [f for f in required_fields if f not in df.columns]
            
            if missing_fields:
                return jsonify({
                    "error": f"Missing required fields: {missing_fields}"
                }), 400
            
            # Extract features
            features = self.extractor.extract_features(df)
            
            # Make predictions
            predictions = self.detector.predict(features.values)
            probabilities = self.detector.predict_proba(features.values)
            
            # Prepare response
            results = []
            for i, (pred, proba) in enumerate(zip(predictions, probabilities)):
                results.append({
                    "transaction_index": i,
                    "prediction": int(pred),
                    "fraud_probability": float(proba[1]),
                    "legitimate_probability": float(proba[0]),
                    "risk_score": float(proba[1]),
                })
            
            return jsonify({
                "status": "success",
                "n_transactions": len(transactions),
                "results": results,
            }), 200
        
        except Exception as e:
            logger.error(f"Error analyzing transactions: {str(e)}")
            return jsonify({"error": str(e)}), 500

    def get_model_status(self):
        """
        Get fraud detection model status.
        
        Returns:
            JSON response with model information
        """
        try:
            if self.detector is None or not self.detector.trained:
                return jsonify({
                    "status": "not_ready",
                    "message": "Fraud detection model not trained",
                }), 200
            
            model_info = self.detector.get_model_info()
            
            return jsonify({
                "status": "ready",
                "model_info": model_info,
                "feature_count": self.extractor.get_feature_count() if self.extractor else None,
            }), 200
        
        except Exception as e:
            logger.error(f"Error getting model status: {str(e)}")
            return jsonify({"error": str(e)}), 500

    def health_check(self):
        """
        Health check endpoint.
        
        Returns:
            JSON response with health status
        """
        return jsonify({
            "status": "healthy",
            "module": "fraud_detection",
            "models_initialized": self.detector is not None,
        }), 200

    def get_blueprint(self):
        """
        Get the Flask blueprint for registration.
        
        Returns:
            Flask blueprint
        """
        return self.blueprint


def create_fraud_detection_api() -> FraudDetectionAPI:
    """
    Factory function to create fraud detection API.
    
    Returns:
        FraudDetectionAPI instance
    """
    api = FraudDetectionAPI()
    api.initialize_models()
    return api
