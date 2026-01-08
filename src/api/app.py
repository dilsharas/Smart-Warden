"""
Flask application factory for the Smart Contract Security Analyzer API.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
import os
from datetime import datetime
from typing import Dict, Any, Optional
import traceback

from .middleware import setup_middleware
from .routes import register_routes
from .utils import setup_logging, load_configuration

# Import logging config with absolute import
import sys
from pathlib import Path
src_path = Path(__file__).parent.parent
sys.path.insert(0, str(src_path))

try:
    from utils.logging_config import setup_logging as setup_app_logging
except ImportError:
    # Fallback logging setup
    def setup_app_logging(level='INFO'):
        logging.basicConfig(level=getattr(logging, level.upper(), logging.INFO))

logger = logging.getLogger(__name__)


def create_app(config_name: str = 'development') -> Flask:
    """
    Create and configure Flask application.
    
    Args:
        config_name: Configuration environment name
        
    Returns:
        Configured Flask application
    """
    app = Flask(__name__)
    
    # Load configuration
    config = load_configuration(config_name)
    app.config.update(config)
    
    # Setup logging
    setup_app_logging(app.config.get('LOG_LEVEL', 'INFO'))
    
    # Setup CORS
    CORS(app, 
         origins=app.config.get('CORS_ORIGINS', ['http://localhost:3000', 'http://localhost:8501']),
         methods=['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
         allow_headers=['Content-Type', 'Authorization'])
    
    # Setup middleware
    setup_middleware(app)
    
    # Register routes
    register_routes(app)
    
    # Global error handlers
    register_error_handlers(app)
    
    # Health check endpoint
    @app.route('/health', methods=['GET'])
    def health_check():
        """Health check endpoint."""
        return jsonify({
            'status': 'healthy',
            'timestamp': datetime.utcnow().isoformat(),
            'version': app.config.get('VERSION', '1.0.0')
        })
    
    logger.info(f"Flask application created with config: {config_name}")
    return app


def register_error_handlers(app: Flask):
    """
    Register global error handlers.
    
    Args:
        app: Flask application instance
    """
    
    @app.errorhandler(400)
    def bad_request(error):
        """Handle bad request errors."""
        return jsonify({
            'success': False,
            'error': {
                'code': 'BAD_REQUEST',
                'message': 'Invalid request data',
                'details': str(error.description) if hasattr(error, 'description') else None
            },
            'timestamp': datetime.utcnow().isoformat()
        }), 400
    
    @app.errorhandler(404)
    def not_found(error):
        """Handle not found errors."""
        return jsonify({
            'success': False,
            'error': {
                'code': 'NOT_FOUND',
                'message': 'Resource not found',
                'details': f"The requested URL {request.url} was not found on the server"
            },
            'timestamp': datetime.utcnow().isoformat()
        }), 404
    
    @app.errorhandler(405)
    def method_not_allowed(error):
        """Handle method not allowed errors."""
        return jsonify({
            'success': False,
            'error': {
                'code': 'METHOD_NOT_ALLOWED',
                'message': 'Method not allowed',
                'details': f"The method {request.method} is not allowed for the requested URL"
            },
            'timestamp': datetime.utcnow().isoformat()
        }), 405
    
    @app.errorhandler(413)
    def request_entity_too_large(error):
        """Handle file too large errors."""
        return jsonify({
            'success': False,
            'error': {
                'code': 'FILE_TOO_LARGE',
                'message': 'File size exceeds maximum allowed size',
                'details': f"Maximum file size is {app.config.get('MAX_CONTENT_LENGTH', 1048576)} bytes"
            },
            'timestamp': datetime.utcnow().isoformat()
        }), 413
    
    @app.errorhandler(429)
    def rate_limit_exceeded(error):
        """Handle rate limit exceeded errors."""
        return jsonify({
            'success': False,
            'error': {
                'code': 'RATE_LIMIT_EXCEEDED',
                'message': 'Rate limit exceeded',
                'details': 'Too many requests. Please try again later.'
            },
            'timestamp': datetime.utcnow().isoformat()
        }), 429
    
    @app.errorhandler(500)
    def internal_server_error(error):
        """Handle internal server errors."""
        logger.error(f"Internal server error: {error}")
        logger.error(traceback.format_exc())
        
        return jsonify({
            'success': False,
            'error': {
                'code': 'INTERNAL_SERVER_ERROR',
                'message': 'Internal server error',
                'details': 'An unexpected error occurred. Please try again later.' if not app.debug else str(error)
            },
            'timestamp': datetime.utcnow().isoformat()
        }), 500
    
    @app.errorhandler(Exception)
    def handle_unexpected_error(error):
        """Handle unexpected errors."""
        logger.error(f"Unexpected error: {error}")
        logger.error(traceback.format_exc())
        
        return jsonify({
            'success': False,
            'error': {
                'code': 'UNEXPECTED_ERROR',
                'message': 'An unexpected error occurred',
                'details': str(error) if app.debug else 'Please try again later.'
            },
            'timestamp': datetime.utcnow().isoformat()
        }), 500


def create_production_app() -> Flask:
    """
    Create production-ready Flask application.
    
    Returns:
        Production Flask application
    """
    app = create_app('production')
    
    # Additional production configurations
    app.config.update({
        'DEBUG': False,
        'TESTING': False,
        'PROPAGATE_EXCEPTIONS': False
    })
    
    return app


def create_development_app() -> Flask:
    """
    Create development Flask application.
    
    Returns:
        Development Flask application
    """
    app = create_app('development')
    
    # Additional development configurations
    app.config.update({
        'DEBUG': True,
        'TESTING': False
    })
    
    return app


def create_testing_app() -> Flask:
    """
    Create testing Flask application.
    
    Returns:
        Testing Flask application
    """
    app = create_app('testing')
    
    # Additional testing configurations
    app.config.update({
        'DEBUG': True,
        'TESTING': True,
        'WTF_CSRF_ENABLED': False
    })
    
    return app


# Application factory for different environments
app_factories = {
    'production': create_production_app,
    'development': create_development_app,
    'testing': create_testing_app
}


def get_app(environment: str = None) -> Flask:
    """
    Get Flask application for specified environment.
    
    Args:
        environment: Environment name (production, development, testing)
        
    Returns:
        Flask application instance
    """
    if environment is None:
        environment = os.getenv('FLASK_ENV', 'development')
    
    factory = app_factories.get(environment, create_development_app)
    return factory()


# Default application instance - only create when needed
app = None

def get_default_app():
    """Get or create the default application instance."""
    global app
    if app is None:
        app = get_app()
    return app


if __name__ == '__main__':
    # Run development server
    app = create_development_app()
    app.run(
        host=app.config.get('HOST', '0.0.0.0'),
        port=app.config.get('PORT', 5000),
        debug=True
    )