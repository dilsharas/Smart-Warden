"""
API routes for the Smart Contract Security Analyzer.
"""

from flask import Flask, request, jsonify, Blueprint
from typing import Dict, Any, Optional
import logging
from datetime import datetime

from .middleware import require_json, validate_json_schema, handle_file_upload
from .utils import (
    validate_contract_code, format_error_response, format_success_response,
    sanitize_filename, get_file_hash, create_analysis_id, estimate_analysis_time,
    get_model_info, check_system_health
)

logger = logging.getLogger(__name__)

# Create blueprints for different API sections
api_bp = Blueprint('api', __name__, url_prefix='/api')
health_bp = Blueprint('health', __name__)


def register_routes(app: Flask):
    """
    Register all routes with the Flask application.
    
    Args:
        app: Flask application instance
    """
    # Register blueprints
    app.register_blueprint(api_bp)
    app.register_blueprint(health_bp)
    
    # Register route handlers
    register_analysis_routes()
    register_model_routes()
    register_system_routes()


def register_analysis_routes():
    """Register contract analysis routes."""
    
    @api_bp.route('/analyze', methods=['POST'])
    @require_json
    @validate_json_schema({
        'contract_code': str,
        'options': dict
    })
    def analyze_contract():
        """
        Analyze a smart contract for vulnerabilities.
        
        Expected JSON payload:
        {
            "contract_code": "pragma solidity ^0.8.0; contract Example { ... }",
            "options": {
                "enable_ai_analysis": true,
                "enable_slither": true,
                "enable_mythril": true,
                "include_feature_importance": true,
                "analysis_timeout": 300
            }
        }
        """
        try:
            data = request.get_json()
            contract_code = data['contract_code']
            options = data.get('options', {})
            
            # Validate contract code
            is_valid, error_msg = validate_contract_code(contract_code)
            if not is_valid:
                return jsonify(format_error_response(
                    'INVALID_CONTRACT_CODE',
                    'Invalid contract code',
                    error_msg
                )), 400
            
            # Create analysis ID
            analysis_id = create_analysis_id()
            
            # Estimate analysis time
            estimated_time = estimate_analysis_time(
                len(contract_code),
                options.get('enable_slither', True) or options.get('enable_mythril', True)
            )
            
            # TODO: Implement actual analysis logic
            # This is a placeholder response
            analysis_result = {
                'analysis_id': analysis_id,
                'contract_hash': get_file_hash(contract_code),
                'status': 'completed',
                'estimated_time': estimated_time,
                'actual_time': 2.5,
                'overall_risk_score': 75,
                'is_vulnerable': True,
                'confidence_level': 0.87,
                'vulnerabilities': [
                    {
                        'type': 'reentrancy',
                        'severity': 'High',
                        'confidence': 0.92,
                        'line_number': 15,
                        'function_name': 'withdraw',
                        'description': 'Potential reentrancy vulnerability detected',
                        'recommendation': 'Use checks-effects-interactions pattern'
                    }
                ],
                'tool_results': {
                    'ai_binary': {
                        'prediction': 'vulnerable',
                        'confidence': 0.87,
                        'execution_time': 0.5
                    },
                    'ai_multiclass': {
                        'prediction': 'reentrancy',
                        'confidence': 0.92,
                        'execution_time': 0.6
                    },
                    'slither': {
                        'findings_count': 2,
                        'execution_time': 1.2,
                        'success': True
                    },
                    'mythril': {
                        'findings_count': 1,
                        'execution_time': 0.8,
                        'success': True
                    }
                },
                'feature_importance': [
                    {'feature': 'external_call_count', 'importance': 0.15},
                    {'feature': 'state_change_after_call', 'importance': 0.12},
                    {'feature': 'has_reentrancy_guard', 'importance': 0.10}
                ] if options.get('include_feature_importance', False) else None
            }
            
            return jsonify(format_success_response(
                analysis_result,
                'Contract analysis completed successfully'
            ))
            
        except Exception as e:
            logger.error(f"Error in contract analysis: {e}")
            return jsonify(format_error_response(
                'ANALYSIS_ERROR',
                'Analysis failed',
                str(e)
            )), 500
    
    @api_bp.route('/analyze/file', methods=['POST'])
    @handle_file_upload(max_size=1024*1024, allowed_extensions={'.sol'})
    def analyze_file():
        """
        Analyze a smart contract file upload.
        
        Expected form data:
        - file: Solidity contract file (.sol)
        - options: JSON string with analysis options (optional)
        """
        try:
            file = request.files['file']
            
            # Read file content
            contract_code = file.read().decode('utf-8')
            
            # Get options from form data
            options_str = request.form.get('options', '{}')
            try:
                import json
                options = json.loads(options_str)
            except json.JSONDecodeError:
                options = {}
            
            # Validate contract code
            is_valid, error_msg = validate_contract_code(contract_code)
            if not is_valid:
                return jsonify(format_error_response(
                    'INVALID_CONTRACT_CODE',
                    'Invalid contract code',
                    error_msg
                )), 400
            
            # Create analysis request
            analysis_data = {
                'contract_code': contract_code,
                'options': options,
                'filename': sanitize_filename(file.filename)
            }
            
            # Reuse the analyze_contract logic
            # TODO: Implement file-specific analysis
            analysis_result = {
                'analysis_id': create_analysis_id(),
                'filename': analysis_data['filename'],
                'contract_hash': get_file_hash(contract_code),
                'status': 'completed',
                'message': 'File analysis completed (placeholder)'
            }
            
            return jsonify(format_success_response(
                analysis_result,
                'File analysis completed successfully'
            ))
            
        except Exception as e:
            logger.error(f"Error in file analysis: {e}")
            return jsonify(format_error_response(
                'FILE_ANALYSIS_ERROR',
                'File analysis failed',
                str(e)
            )), 500
    
    @api_bp.route('/analysis/<analysis_id>', methods=['GET'])
    def get_analysis_result(analysis_id: str):
        """
        Get analysis result by ID.
        
        Args:
            analysis_id: Analysis identifier
        """
        try:
            # TODO: Implement result retrieval from storage
            # This is a placeholder response
            
            if not analysis_id:
                return jsonify(format_error_response(
                    'INVALID_ANALYSIS_ID',
                    'Invalid analysis ID',
                    'Analysis ID cannot be empty'
                )), 400
            
            # Placeholder result
            result = {
                'analysis_id': analysis_id,
                'status': 'completed',
                'created_at': datetime.utcnow().isoformat(),
                'message': 'Analysis result retrieved (placeholder)'
            }
            
            return jsonify(format_success_response(result))
            
        except Exception as e:
            logger.error(f"Error retrieving analysis {analysis_id}: {e}")
            return jsonify(format_error_response(
                'RETRIEVAL_ERROR',
                'Failed to retrieve analysis result',
                str(e)
            )), 500


def register_model_routes():
    """Register model information routes."""
    
    @api_bp.route('/models/info', methods=['GET'])
    def get_models_info():
        """Get information about available models."""
        try:
            # TODO: Load actual model information
            models_info = {
                'binary_classifier': {
                    'available': True,
                    'model_type': 'RandomForestClassifier',
                    'accuracy': 0.87,
                    'last_trained': '2024-01-15T10:30:00Z'
                },
                'multiclass_classifier': {
                    'available': True,
                    'model_type': 'RandomForestClassifier',
                    'accuracy': 0.83,
                    'classes': ['safe', 'reentrancy', 'access_control', 'arithmetic'],
                    'last_trained': '2024-01-15T10:30:00Z'
                },
                'feature_extractor': {
                    'available': True,
                    'feature_count': 35,
                    'version': '1.0.0'
                }
            }
            
            return jsonify(format_success_response(models_info))
            
        except Exception as e:
            logger.error(f"Error getting models info: {e}")
            return jsonify(format_error_response(
                'MODELS_INFO_ERROR',
                'Failed to get models information',
                str(e)
            )), 500
    
    @api_bp.route('/models/performance', methods=['GET'])
    def get_model_performance():
        """Get model performance metrics."""
        try:
            # TODO: Load actual performance metrics
            performance_metrics = {
                'binary_classifier': {
                    'accuracy': 0.87,
                    'precision': 0.85,
                    'recall': 0.89,
                    'f1_score': 0.87,
                    'roc_auc': 0.92,
                    'test_samples': 150
                },
                'multiclass_classifier': {
                    'accuracy': 0.83,
                    'macro_f1': 0.81,
                    'micro_f1': 0.83,
                    'per_class_metrics': {
                        'safe': {'precision': 0.90, 'recall': 0.88, 'f1': 0.89},
                        'reentrancy': {'precision': 0.82, 'recall': 0.85, 'f1': 0.83},
                        'access_control': {'precision': 0.78, 'recall': 0.80, 'f1': 0.79},
                        'arithmetic': {'precision': 0.75, 'recall': 0.72, 'f1': 0.73}
                    },
                    'test_samples': 150
                }
            }
            
            return jsonify(format_success_response(performance_metrics))
            
        except Exception as e:
            logger.error(f"Error getting model performance: {e}")
            return jsonify(format_error_response(
                'PERFORMANCE_ERROR',
                'Failed to get model performance',
                str(e)
            )), 500


def register_system_routes():
    """Register system and utility routes."""
    
    @health_bp.route('/health', methods=['GET'])
    def health_check():
        """System health check."""
        try:
            health_info = check_system_health()
            return jsonify(health_info)
            
        except Exception as e:
            logger.error(f"Health check error: {e}")
            return jsonify({
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }), 500
    
    @api_bp.route('/status', methods=['GET'])
    def get_system_status():
        """Get detailed system status."""
        try:
            # TODO: Implement comprehensive status check
            status = {
                'api_version': '1.0.0',
                'status': 'operational',
                'services': {
                    'ai_models': 'available',
                    'slither': 'available',
                    'mythril': 'available',
                    'feature_extractor': 'available'
                },
                'uptime': '2 hours 15 minutes',
                'total_analyses': 42,
                'cache_status': 'enabled',
                'last_updated': datetime.utcnow().isoformat()
            }
            
            return jsonify(format_success_response(status))
            
        except Exception as e:
            logger.error(f"Error getting system status: {e}")
            return jsonify(format_error_response(
                'STATUS_ERROR',
                'Failed to get system status',
                str(e)
            )), 500
    
    @api_bp.route('/tools/status', methods=['GET'])
    def get_tools_status():
        """Get status of external analysis tools."""
        try:
            # TODO: Check actual tool availability
            tools_status = {
                'slither': {
                    'available': True,
                    'version': '0.9.6',
                    'last_check': datetime.utcnow().isoformat()
                },
                'mythril': {
                    'available': True,
                    'version': '0.23.25',
                    'last_check': datetime.utcnow().isoformat()
                }
            }
            
            return jsonify(format_success_response(tools_status))
            
        except Exception as e:
            logger.error(f"Error getting tools status: {e}")
            return jsonify(format_error_response(
                'TOOLS_STATUS_ERROR',
                'Failed to get tools status',
                str(e)
            )), 500
    
    @api_bp.route('/compare', methods=['POST'])
    @require_json
    @validate_json_schema({
        'contract_code': str,
        'tools': list
    })
    def compare_tools():
        """
        Compare multiple analysis tools on a contract.
        
        Expected JSON payload:
        {
            "contract_code": "pragma solidity ^0.8.0; contract Example { ... }",
            "tools": ["ai_binary", "ai_multiclass", "slither", "mythril"],
            "ground_truth": ["reentrancy"] // optional
        }
        """
        try:
            data = request.get_json()
            contract_code = data['contract_code']
            tools = data['tools']
            ground_truth = data.get('ground_truth', None)
            
            # Validate contract code
            is_valid, error_msg = validate_contract_code(contract_code)
            if not is_valid:
                return jsonify(format_error_response(
                    'INVALID_CONTRACT_CODE',
                    'Invalid contract code',
                    error_msg
                )), 400
            
            # Validate tools list
            available_tools = ['ai_binary', 'ai_multiclass', 'slither', 'mythril']
            invalid_tools = [tool for tool in tools if tool not in available_tools]
            if invalid_tools:
                return jsonify(format_error_response(
                    'INVALID_TOOLS',
                    'Invalid tools specified',
                    f'Unknown tools: {invalid_tools}. Available: {available_tools}'
                )), 400
            
            # TODO: Implement actual tool comparison
            comparison_result = {
                'comparison_id': create_analysis_id(),
                'contract_hash': get_file_hash(contract_code),
                'tools_compared': tools,
                'ground_truth': ground_truth,
                'results': {
                    'ai_binary': {
                        'prediction': 'vulnerable',
                        'confidence': 0.87,
                        'execution_time': 0.5,
                        'success': True
                    },
                    'slither': {
                        'vulnerabilities_found': ['reentrancy'],
                        'findings_count': 1,
                        'execution_time': 1.2,
                        'success': True
                    }
                },
                'consensus_findings': ['reentrancy'],
                'agreement_score': 0.85,
                'execution_time': 2.1
            }
            
            return jsonify(format_success_response(
                comparison_result,
                'Tool comparison completed successfully'
            ))
            
        except Exception as e:
            logger.error(f"Error in tool comparison: {e}")
            return jsonify(format_error_response(
                'COMPARISON_ERROR',
                'Tool comparison failed',
                str(e)
            )), 500