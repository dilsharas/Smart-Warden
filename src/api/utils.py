"""
Utility functions for the Flask API.
"""

import os
import yaml
import json
import logging
from typing import Dict, Any, Optional
from pathlib import Path
from datetime import datetime


def load_configuration(config_name: str = 'development') -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_name: Configuration environment name
        
    Returns:
        Configuration dictionary
    """
    # Default configuration
    default_config = {
        'DEBUG': False,
        'TESTING': False,
        'SECRET_KEY': os.getenv('SECRET_KEY', 'dev-secret-key-change-in-production'),
        'HOST': '0.0.0.0',
        'PORT': 5000,
        'MAX_CONTENT_LENGTH': 10 * 1024 * 1024,  # 10MB
        'CORS_ORIGINS': ['http://localhost:3000', 'http://localhost:8501'],
        'ENABLE_RATE_LIMITING': True,
        'RATE_LIMIT_REQUESTS': 100,
        'RATE_LIMIT_WINDOW': 15,
        'ENABLE_SECURITY_HEADERS': True,
        'LOG_LEVEL': 'INFO',
        'VERSION': '1.0.0',
        'API_TITLE': 'Smart Contract Security Analyzer API',
        'API_DESCRIPTION': 'AI-enhanced smart contract vulnerability detection API',
        'MODELS_PATH': 'models/',
        'CACHE_ENABLED': True,
        'CACHE_TTL': 3600,  # 1 hour
        'ANALYSIS_TIMEOUT': 300,  # 5 minutes
        'ENABLE_TOOL_COMPARISON': True,
        'SLITHER_ENABLED': True,
        'MYTHRIL_ENABLED': True
    }
    
    # Environment-specific configurations
    config_overrides = {
        'development': {
            'DEBUG': True,
            'LOG_LEVEL': 'DEBUG',
            'RATE_LIMIT_REQUESTS': 1000,  # More lenient for development
        },
        'testing': {
            'DEBUG': True,
            'TESTING': True,
            'LOG_LEVEL': 'DEBUG',
            'ENABLE_RATE_LIMITING': False,
            'CACHE_ENABLED': False
        },
        'production': {
            'DEBUG': False,
            'LOG_LEVEL': 'WARNING',
            'SECRET_KEY': os.getenv('SECRET_KEY'),
            'HOST': '0.0.0.0',
            'PORT': int(os.getenv('PORT', 5000))
        }
    }
    
    # Start with default config
    config = default_config.copy()
    
    # Apply environment-specific overrides
    if config_name in config_overrides:
        config.update(config_overrides[config_name])
    
    # Load from YAML file if exists
    config_file = Path(f'configs/api_config.yaml')
    if config_file.exists():
        try:
            with open(config_file, 'r') as f:
                file_config = yaml.safe_load(f) or {}
                
                # Apply environment-specific config from file
                if config_name in file_config:
                    config.update(file_config[config_name])
                
                # Apply common config from file
                if 'common' in file_config:
                    common_config = file_config['common']
                    # Don't override environment-specific settings
                    for key, value in common_config.items():
                        if key not in config_overrides.get(config_name, {}):
                            config[key] = value
                            
        except Exception as e:
            logging.warning(f"Failed to load config file {config_file}: {e}")
    
    # Override with environment variables
    env_overrides = {
        'SECRET_KEY': os.getenv('SECRET_KEY'),
        'HOST': os.getenv('HOST'),
        'PORT': os.getenv('PORT'),
        'LOG_LEVEL': os.getenv('LOG_LEVEL'),
        'MAX_CONTENT_LENGTH': os.getenv('MAX_CONTENT_LENGTH'),
        'RATE_LIMIT_REQUESTS': os.getenv('RATE_LIMIT_REQUESTS'),
        'RATE_LIMIT_WINDOW': os.getenv('RATE_LIMIT_WINDOW'),
        'ANALYSIS_TIMEOUT': os.getenv('ANALYSIS_TIMEOUT'),
        'MODELS_PATH': os.getenv('MODELS_PATH')
    }
    
    for key, value in env_overrides.items():
        if value is not None:
            # Convert to appropriate type
            if key in ['PORT', 'MAX_CONTENT_LENGTH', 'RATE_LIMIT_REQUESTS', 'RATE_LIMIT_WINDOW', 'ANALYSIS_TIMEOUT']:
                try:
                    config[key] = int(value)
                except ValueError:
                    logging.warning(f"Invalid integer value for {key}: {value}")
            elif key in ['DEBUG', 'TESTING', 'ENABLE_RATE_LIMITING', 'ENABLE_SECURITY_HEADERS', 'CACHE_ENABLED']:
                config[key] = value.lower() in ('true', '1', 'yes', 'on')
            else:
                config[key] = value
    
    return config


def setup_logging(log_level: str = 'INFO'):
    """
    Setup logging configuration.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
    """
    # Convert string level to logging constant
    level_map = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR,
        'CRITICAL': logging.CRITICAL
    }
    
    level = level_map.get(log_level.upper(), logging.INFO)
    
    # Configure logging format
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Setup basic logging
    logging.basicConfig(
        level=level,
        format=log_format,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('logs/api.log', mode='a') if Path('logs').exists() else logging.NullHandler()
        ]
    )
    
    # Set specific logger levels
    logging.getLogger('werkzeug').setLevel(logging.WARNING)  # Reduce Flask request logs
    logging.getLogger('urllib3').setLevel(logging.WARNING)   # Reduce HTTP client logs


def validate_contract_code(code: str) -> tuple[bool, Optional[str]]:
    """
    Validate Solidity contract code.
    
    Args:
        code: Solidity source code
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not code or not code.strip():
        return False, "Contract code cannot be empty"
    
    # Check minimum length
    if len(code.strip()) < 10:
        return False, "Contract code is too short"
    
    # Check maximum length (10MB)
    if len(code) > 10 * 1024 * 1024:
        return False, "Contract code is too large (max 10MB)"
    
    # Check for pragma statement
    if 'pragma solidity' not in code.lower():
        return False, "Contract must include 'pragma solidity' statement"
    
    # Check for contract keyword
    if 'contract ' not in code:
        return False, "Contract must include 'contract' keyword"
    
    # Check for balanced braces
    open_braces = code.count('{')
    close_braces = code.count('}')
    if open_braces != close_braces:
        return False, "Unbalanced braces in contract code"
    
    # Check for balanced parentheses
    open_parens = code.count('(')
    close_parens = code.count(')')
    if open_parens != close_parens:
        return False, "Unbalanced parentheses in contract code"
    
    return True, None


def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename for safe storage.
    
    Args:
        filename: Original filename
        
    Returns:
        Sanitized filename
    """
    # Remove path components
    filename = os.path.basename(filename)
    
    # Replace unsafe characters
    unsafe_chars = '<>:"/\\|?*'
    for char in unsafe_chars:
        filename = filename.replace(char, '_')
    
    # Limit length
    if len(filename) > 255:
        name, ext = os.path.splitext(filename)
        filename = name[:250] + ext
    
    # Ensure it's not empty
    if not filename:
        filename = 'contract.sol'
    
    return filename


def format_error_response(error_code: str, message: str, details: Optional[str] = None) -> Dict[str, Any]:
    """
    Format standardized error response.
    
    Args:
        error_code: Error code identifier
        message: Human-readable error message
        details: Additional error details
        
    Returns:
        Formatted error response dictionary
    """
    return {
        'success': False,
        'error': {
            'code': error_code,
            'message': message,
            'details': details
        },
        'timestamp': datetime.utcnow().isoformat()
    }


def format_success_response(data: Any, message: Optional[str] = None) -> Dict[str, Any]:
    """
    Format standardized success response.
    
    Args:
        data: Response data
        message: Optional success message
        
    Returns:
        Formatted success response dictionary
    """
    response = {
        'success': True,
        'data': data,
        'timestamp': datetime.utcnow().isoformat()
    }
    
    if message:
        response['message'] = message
    
    return response


def get_file_hash(content: str) -> str:
    """
    Generate hash for file content.
    
    Args:
        content: File content
        
    Returns:
        SHA-256 hash of content
    """
    import hashlib
    return hashlib.sha256(content.encode('utf-8')).hexdigest()


def parse_solidity_version(code: str) -> Optional[str]:
    """
    Parse Solidity version from pragma statement.
    
    Args:
        code: Solidity source code
        
    Returns:
        Solidity version string or None
    """
    import re
    
    # Look for pragma solidity statement
    pragma_match = re.search(r'pragma\s+solidity\s+([^;]+);', code, re.IGNORECASE)
    if pragma_match:
        version_spec = pragma_match.group(1).strip()
        
        # Extract version number
        version_match = re.search(r'(\d+\.\d+\.\d+)', version_spec)
        if version_match:
            return version_match.group(1)
        
        # Extract major.minor version
        version_match = re.search(r'(\d+\.\d+)', version_spec)
        if version_match:
            return version_match.group(1)
    
    return None


def estimate_analysis_time(code_length: int, enable_tools: bool = True) -> int:
    """
    Estimate analysis time based on code length and enabled tools.
    
    Args:
        code_length: Length of contract code
        enable_tools: Whether external tools are enabled
        
    Returns:
        Estimated time in seconds
    """
    # Base time for AI analysis (fast)
    base_time = max(1, code_length // 1000)  # 1 second per 1000 characters
    
    # Add time for external tools
    if enable_tools:
        tool_time = max(5, code_length // 500)  # 5-20 seconds for tools
        return base_time + tool_time
    
    return base_time


def create_analysis_id() -> str:
    """
    Create unique analysis ID.
    
    Returns:
        Unique analysis identifier
    """
    import uuid
    return str(uuid.uuid4())


def get_model_info(model_path: str) -> Optional[Dict[str, Any]]:
    """
    Get information about a saved model.
    
    Args:
        model_path: Path to model file
        
    Returns:
        Model information dictionary or None
    """
    try:
        import joblib
        
        if not os.path.exists(model_path):
            return None
        
        # Load model metadata
        model_data = joblib.load(model_path)
        
        if isinstance(model_data, dict) and 'model_metadata' in model_data:
            metadata = model_data['model_metadata']
            return {
                'model_type': metadata.get('model_type', 'Unknown'),
                'classes': metadata.get('classes', []),
                'feature_count': len(metadata.get('feature_columns', [])),
                'file_size': os.path.getsize(model_path),
                'last_modified': datetime.fromtimestamp(os.path.getmtime(model_path)).isoformat()
            }
    
    except Exception as e:
        logging.error(f"Error loading model info from {model_path}: {e}")
    
    return None


def check_system_health() -> Dict[str, Any]:
    """
    Check system health and resource usage.
    
    Returns:
        System health information
    """
    import psutil
    
    try:
        # Get system metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        return {
            'status': 'healthy',
            'cpu_usage_percent': cpu_percent,
            'memory_usage_percent': memory.percent,
            'memory_available_mb': memory.available // (1024 * 1024),
            'disk_usage_percent': disk.percent,
            'disk_free_gb': disk.free // (1024 * 1024 * 1024),
            'timestamp': datetime.utcnow().isoformat()
        }
    
    except Exception as e:
        return {
            'status': 'error',
            'error': str(e),
            'timestamp': datetime.utcnow().isoformat()
        }