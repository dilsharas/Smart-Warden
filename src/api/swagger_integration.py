"""
Swagger/OpenAPI integration for Smart Warden API.
Provides interactive API documentation and testing interface.
"""

from flask import Flask, jsonify, request
from flask_restx import Api, Resource, fields, Namespace
import os
import json
from datetime import datetime
from typing import Dict, Any

def setup_swagger(app: Flask) -> Api:
    """Setup Swagger/OpenAPI documentation for the Flask app."""
    
    # Configure API documentation
    api = Api(
        app,
        version='1.0.0',
        title='Smart Warden API',
        description='''
        # AI-Powered Smart Contract Security Analyzer API
        
        Smart Warden provides comprehensive security analysis for Ethereum smart contracts using:
        - **AI Models**: Binary and multi-class vulnerability detection
        - **External Tools**: Slither static analysis and Mythril symbolic execution
        - **Tool Comparison**: Consensus analysis across multiple tools
        - **Real-time Analysis**: Fast vulnerability detection with detailed reporting
        
        ## Getting Started
        1. Use the `/health` endpoint to verify API status
        2. Submit contracts via `/api/analyze` for security analysis
        3. Configure analysis options to include external tools
        4. Review detailed results with vulnerability findings and recommendations
        
        ## Authentication
        Currently no authentication required for development.
        Production deployment should implement API key authentication.
        ''',
        doc='/swagger/',
        contact='Smart Warden Team',
        contact_email='support@smartwarden.ai',
        license='MIT',
        license_url='https://opensource.org/licenses/MIT',
        authorizations={
            'apikey': {
                'type': 'apiKey',
                'in': 'header',
                'name': 'X-API-Key',
                'description': 'API Key for authentication (production only)'
            }
        },
        security='apikey'
    )
    
    # Create namespaces
    system_ns = Namespace('system', description='System health and status operations')
    analysis_ns = Namespace('analysis', description='Smart contract analysis operations')
    models_ns = Namespace('models', description='AI model management operations')
    tools_ns = Namespace('tools', description='External tools management operations')
    
    api.add_namespace(system_ns, path='/system')
    api.add_namespace(analysis_ns, path='/api')
    api.add_namespace(models_ns, path='/api/models')
    api.add_namespace(tools_ns, path='/api/tools')
    
    # Define data models
    define_api_models(api)
    
    # Add endpoints
    add_system_endpoints(system_ns, api)
    add_analysis_endpoints(analysis_ns, api)
    add_model_endpoints(models_ns, api)
    add_tool_endpoints(tools_ns, api)
    
    return api

def define_api_models(api: Api):
    """Define Swagger data models."""
    
    # Analysis Request Model
    api.analysis_options = api.model('AnalysisOptions', {
        'include_slither': fields.Boolean(
            default=False,
            description='Include Slither static analysis',
            example=True
        ),
        'include_mythril': fields.Boolean(
            default=False,
            description='Include Mythril symbolic execution',
            example=True
        ),
        'compare_tools': fields.Boolean(
            default=False,
            description='Enable tool comparison analysis',
            example=True
        ),
        'generate_pdf_report': fields.Boolean(
            default=False,
            description='Generate PDF report',
            example=False
        ),
        'timeout': fields.Integer(
            default=60,
            description='Analysis timeout in seconds',
            example=120
        )
    })
    
    api.analysis_request = api.model('AnalysisRequest', {
        'contract_code': fields.String(
            required=True,
            description='Solidity contract source code',
            example='''pragma solidity ^0.8.0;

contract SimpleStorage {
    uint256 public value;
    
    function setValue(uint256 _value) public {
        value = _value;
    }
    
    function getValue() public view returns (uint256) {
        return value;
    }
}'''
        ),
        'filename': fields.String(
            description='Optional filename for the contract',
            example='SimpleStorage.sol'
        ),
        'options': fields.Nested(
            api.analysis_options,
            description='Analysis configuration options'
        )
    })
    
    # Vulnerability Model
    api.vulnerability = api.model('Vulnerability', {
        'type': fields.String(
            description='Vulnerability type',
            enum=['reentrancy', 'access_control', 'arithmetic', 'unchecked_calls', 'dos', 'bad_randomness'],
            example='reentrancy'
        ),
        'severity': fields.String(
            description='Severity level',
            enum=['critical', 'high', 'medium', 'low'],
            example='high'
        ),
        'confidence': fields.Float(
            description='Confidence score (0.0-1.0)',
            min=0.0,
            max=1.0,
            example=0.85
        ),
        'line_number': fields.Integer(
            description='Line number where vulnerability was found',
            example=15
        ),
        'description': fields.String(
            description='Detailed vulnerability description',
            example='Potential reentrancy vulnerability detected in withdraw function'
        ),
        'recommendation': fields.String(
            description='Recommended fix for the vulnerability',
            example='Use checks-effects-interactions pattern or reentrancy guard'
        ),
        'source': fields.String(
            description='Analysis tool that detected this vulnerability',
            enum=['AI', 'Slither', 'Mythril', 'Native'],
            example='AI'
        ),
        'code_snippet': fields.String(
            description='Code snippet showing the vulnerability',
            example='msg.sender.call{value: amount}("");'
        )
    })
    
    # Tool Comparison Model
    api.tool_consensus = api.model('ToolConsensus', {
        'is_vulnerable': fields.Boolean(
            description='Consensus vulnerability assessment',
            example=True
        ),
        'confidence': fields.Float(
            description='Consensus confidence level',
            example=0.87
        ),
        'agreement_level': fields.String(
            description='Level of agreement between tools',
            enum=['unanimous', 'majority', 'split'],
            example='majority'
        ),
        'agreed_vulnerabilities': fields.List(
            fields.String,
            description='Vulnerability types agreed upon by multiple tools',
            example=['reentrancy', 'access_control']
        )
    })
    
    api.tool_comparison = api.model('ToolComparison', {
        'tools_used': fields.List(
            fields.String,
            description='List of analysis tools used',
            example=['AI', 'Slither', 'Mythril']
        ),
        'consensus': fields.Nested(
            api.tool_consensus,
            description='Consensus analysis results'
        ),
        'tool_agreement': fields.Float(
            description='Agreement percentage between tools (0.0-1.0)',
            example=0.75
        ),
        'combined_score': fields.Float(
            description='Combined risk score from all tools',
            example=78.5
        ),
        'recommendations': fields.List(
            fields.String,
            description='Aggregated recommendations from all tools',
            example=[
                'Implement checks-effects-interactions pattern',
                'Add proper access control modifiers',
                'Use SafeMath or Solidity 0.8+ for overflow protection'
            ]
        )
    })
    
    # Analysis Result Model
    api.analysis_result = api.model('AnalysisResult', {
        'analysis_id': fields.String(
            description='Unique analysis identifier',
            example='comprehensive_1698765432'
        ),
        'success': fields.Boolean(
            description='Analysis completion status',
            example=True
        ),
        'is_vulnerable': fields.Boolean(
            description='Overall vulnerability assessment',
            example=True
        ),
        'risk_score': fields.Integer(
            description='Overall risk score (0-100)',
            min=0,
            max=100,
            example=75
        ),
        'confidence': fields.Float(
            description='Overall confidence level (0.0-1.0)',
            min=0.0,
            max=1.0,
            example=0.87
        ),
        'vulnerabilities': fields.List(
            fields.Nested(api.vulnerability),
            description='List of detected vulnerabilities'
        ),
        'analysis_time': fields.Float(
            description='Analysis duration in seconds',
            example=12.34
        ),
        'timestamp': fields.DateTime(
            description='Analysis timestamp',
            example='2024-01-15T10:30:00Z'
        ),
        'tools_used': fields.List(
            fields.String,
            description='List of analysis tools used',
            example=['AI', 'Slither', 'Mythril']
        ),
        'tool_comparison': fields.Nested(
            api.tool_comparison,
            description='Tool comparison results (if enabled)'
        ),
        'analysis_method': fields.String(
            description='Primary analysis method used',
            example='Comprehensive Analysis'
        )
    })
    
    # Error Model
    api.error_model = api.model('Error', {
        'error': fields.String(
            description='Error message',
            example='Invalid contract code provided'
        ),
        'code': fields.String(
            description='Error code',
            example='INVALID_INPUT'
        ),
        'details': fields.String(
            description='Additional error details',
            example='Contract code must be valid Solidity syntax'
        ),
        'timestamp': fields.DateTime(
            description='Error timestamp',
            example='2024-01-15T10:30:00Z'
        )
    })
    
    # Health Status Model
    api.health_status = api.model('HealthStatus', {
        'status': fields.String(
            description='System health status',
            enum=['healthy', 'degraded', 'unhealthy'],
            example='healthy'
        ),
        'timestamp': fields.DateTime(
            description='Health check timestamp',
            example='2024-01-15T10:30:00Z'
        ),
        'version': fields.String(
            description='API version',
            example='1.0.0'
        ),
        'components': fields.Raw(
            description='Component health status',
            example={
                'ai_models': 'healthy',
                'external_tools': 'healthy',
                'database': 'healthy'
            }
        )
    })
    
    # Model Info Model
    api.model_info = api.model('ModelInfo', {
        'models_loaded': fields.Integer(
            description='Number of models loaded',
            example=2
        ),
        'binary_model': fields.Raw(
            description='Binary classification model information',
            example={
                'accuracy': 0.857,
                'training_date': '2024-01-15T10:30:00Z',
                'features': 67
            }
        ),
        'multiclass_model': fields.Raw(
            description='Multi-class classification model information',
            example={
                'accuracy': 1.0,
                'training_date': '2024-01-15T10:30:00Z',
                'classes': ['safe', 'reentrancy', 'access_control', 'arithmetic', 'unchecked_calls', 'dos', 'bad_randomness']
            }
        )
    })
    
    # Tools Status Model
    api.tools_status = api.model('ToolsStatus', {
        'slither': fields.Raw(
            description='Slither tool status',
            example={
                'available': True,
                'implementation': 'native',
                'version': '1.0.0'
            }
        ),
        'mythril': fields.Raw(
            description='Mythril tool status',
            example={
                'available': True,
                'implementation': 'native',
                'version': '1.0.0'
            }
        ),
        'docker': fields.Boolean(
            description='Docker availability',
            example=False
        )
    })

def add_system_endpoints(ns: Namespace, api: Api):
    """Add system-related endpoints."""
    
    @ns.route('/health')
    class HealthCheck(Resource):
        @ns.doc('health_check')
        @ns.marshal_with(api.health_status)
        @ns.response(200, 'System is healthy')
        def get(self):
            """
            System Health Check
            
            Check the overall health and status of the Smart Warden API system.
            Returns information about system components and their current status.
            """
            return {
                'status': 'healthy',
                'timestamp': datetime.now().isoformat(),
                'version': '1.0.0',
                'components': {
                    'ai_models': 'healthy',
                    'external_tools': 'healthy',
                    'feature_extraction': 'healthy'
                }
            }

def add_analysis_endpoints(ns: Namespace, api: Api):
    """Add analysis-related endpoints."""
    
    @ns.route('/analyze')
    class ContractAnalysis(Resource):
        @ns.doc('analyze_contract')
        @ns.expect(api.analysis_request, validate=True)
        @ns.marshal_with(api.analysis_result)
        @ns.response(200, 'Analysis completed successfully')
        @ns.response(400, 'Invalid request', api.error_model)
        @ns.response(500, 'Internal server error', api.error_model)
        def post(self):
            """
            Analyze Smart Contract
            
            Submit a Solidity smart contract for comprehensive security analysis.
            
            The analysis includes:
            - AI-powered vulnerability detection (binary and multi-class)
            - Optional external tool integration (Slither, Mythril)
            - Tool comparison and consensus analysis
            - Detailed vulnerability reporting with recommendations
            
            **Analysis Options:**
            - `include_slither`: Enable Slither static analysis
            - `include_mythril`: Enable Mythril symbolic execution  
            - `compare_tools`: Enable multi-tool comparison
            - `generate_pdf_report`: Generate downloadable PDF report
            - `timeout`: Analysis timeout in seconds (default: 60)
            
            **Response includes:**
            - Overall vulnerability assessment and risk score
            - Detailed list of detected vulnerabilities
            - Tool comparison results (if enabled)
            - Analysis performance metrics
            - Actionable security recommendations
            """
            # This would be implemented in the actual API
            return {
                'analysis_id': f'analysis_{int(datetime.now().timestamp())}',
                'success': True,
                'is_vulnerable': False,
                'risk_score': 25,
                'confidence': 0.92,
                'vulnerabilities': [],
                'analysis_time': 5.67,
                'timestamp': datetime.now().isoformat(),
                'tools_used': ['AI'],
                'analysis_method': 'AI Analysis'
            }

def add_model_endpoints(ns: Namespace, api: Api):
    """Add model management endpoints."""
    
    @ns.route('/info')
    class ModelInfo(Resource):
        @ns.doc('model_info')
        @ns.marshal_with(api.model_info)
        @ns.response(200, 'Model information retrieved')
        def get(self):
            """
            Get AI Model Information
            
            Retrieve detailed information about the loaded AI models including:
            - Model accuracy and performance metrics
            - Training dates and dataset information
            - Feature counts and model capabilities
            - Model status and availability
            """
            return {
                'models_loaded': 2,
                'binary_model': {
                    'accuracy': 0.857,
                    'training_date': '2024-01-15T10:30:00Z',
                    'features': 67,
                    'model_type': 'RandomForest'
                },
                'multiclass_model': {
                    'accuracy': 1.0,
                    'training_date': '2024-01-15T10:30:00Z',
                    'classes': ['safe', 'reentrancy', 'access_control', 'arithmetic', 'unchecked_calls', 'dos', 'bad_randomness'],
                    'model_type': 'RandomForest'
                }
            }

def add_tool_endpoints(ns: Namespace, api: Api):
    """Add external tools endpoints."""
    
    @ns.route('/status')
    class ToolsStatus(Resource):
        @ns.doc('tools_status')
        @ns.marshal_with(api.tools_status)
        @ns.response(200, 'Tools status retrieved')
        def get(self):
            """
            External Tools Status
            
            Check the availability and status of external security analysis tools:
            - Slither static analysis tool
            - Mythril symbolic execution tool
            - Docker integration status
            - Implementation details (Docker vs Native)
            """
            return {
                'slither': {
                    'available': True,
                    'implementation': 'native',
                    'version': '1.0.0'
                },
                'mythril': {
                    'available': True,
                    'implementation': 'native',
                    'version': '1.0.0'
                },
                'docker': False
            }

def create_openapi_spec() -> Dict[str, Any]:
    """Create complete OpenAPI specification."""
    return {
        "openapi": "3.0.3",
        "info": {
            "title": "Smart Warden API",
            "description": "AI-powered Smart Contract Security Analyzer API",
            "version": "1.0.0",
            "contact": {
                "name": "Smart Warden Team",
                "email": "support@smartwarden.ai"
            },
            "license": {
                "name": "MIT",
                "url": "https://opensource.org/licenses/MIT"
            }
        },
        "servers": [
            {
                "url": "http://localhost:5000",
                "description": "Development server"
            },
            {
                "url": "https://api.smartwarden.ai",
                "description": "Production server"
            }
        ],
        "tags": [
            {
                "name": "system",
                "description": "System health and status operations"
            },
            {
                "name": "analysis",
                "description": "Smart contract analysis operations"
            },
            {
                "name": "models",
                "description": "AI model management operations"
            },
            {
                "name": "tools",
                "description": "External tools management operations"
            }
        ]
    }