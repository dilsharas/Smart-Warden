"""
Middleware components for the Flask API.
"""

from flask import Flask, request, jsonify, g
from functools import wraps
import time
import logging
from typing import Dict, Any, Optional, Callable
from datetime import datetime, timedelta
import hashlib
import json

logger = logging.getLogger(__name__)


class RateLimiter:
    """Simple in-memory rate limiter."""
    
    def __init__(self, max_requests: int = 100, window_minutes: int = 15):
        """
        Initialize rate limiter.
        
        Args:
            max_requests: Maximum requests per window
            window_minutes: Time window in minutes
        """
        self.max_requests = max_requests
        self.window_minutes = window_minutes
        self.requests = {}  # {client_id: [(timestamp, count), ...]}
    
    def is_allowed(self, client_id: str) -> bool:
        """
        Check if request is allowed for client.
        
        Args:
            client_id: Client identifier
            
        Returns:
            True if request is allowed, False otherwise
        """
        now = datetime.utcnow()
        window_start = now - timedelta(minutes=self.window_minutes)
        
        # Clean old entries
        if client_id in self.requests:
            self.requests[client_id] = [
                (timestamp, count) for timestamp, count in self.requests[client_id]
                if timestamp > window_start
            ]
        else:
            self.requests[client_id] = []
        
        # Count requests in current window
        total_requests = sum(count for _, count in self.requests[client_id])
        
        if total_requests >= self.max_requests:
            return False
        
        # Add current request
        self.requests[client_id].append((now, 1))
        return True
    
    def get_remaining_requests(self, client_id: str) -> int:
        """
        Get remaining requests for client.
        
        Args:
            client_id: Client identifier
            
        Returns:
            Number of remaining requests
        """
        now = datetime.utcnow()
        window_start = now - timedelta(minutes=self.window_minutes)
        
        if client_id not in self.requests:
            return self.max_requests
        
        # Count requests in current window
        current_requests = sum(
            count for timestamp, count in self.requests[client_id]
            if timestamp > window_start
        )
        
        return max(0, self.max_requests - current_requests)


class RequestLogger:
    """Request logging middleware."""
    
    def __init__(self):
        """Initialize request logger."""
        self.logger = logging.getLogger('api.requests')
    
    def log_request(self, start_time: float, response_status: int):
        """
        Log request details.
        
        Args:
            start_time: Request start time
            response_status: HTTP response status code
        """
        duration = time.time() - start_time
        
        log_data = {
            'method': request.method,
            'url': request.url,
            'remote_addr': request.remote_addr,
            'user_agent': request.headers.get('User-Agent', ''),
            'duration_ms': round(duration * 1000, 2),
            'status_code': response_status,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        # Log request size for POST requests
        if request.method == 'POST' and request.content_length:
            log_data['content_length'] = request.content_length
        
        self.logger.info(f"Request processed", extra=log_data)


class SecurityHeaders:
    """Security headers middleware."""
    
    @staticmethod
    def add_security_headers(response):
        """
        Add security headers to response.
        
        Args:
            response: Flask response object
            
        Returns:
            Response with security headers
        """
        # Prevent clickjacking
        response.headers['X-Frame-Options'] = 'DENY'
        
        # Prevent MIME type sniffing
        response.headers['X-Content-Type-Options'] = 'nosniff'
        
        # Enable XSS protection
        response.headers['X-XSS-Protection'] = '1; mode=block'
        
        # Strict transport security (HTTPS only)
        response.headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains'
        
        # Content security policy
        response.headers['Content-Security-Policy'] = "default-src 'self'"
        
        # Referrer policy
        response.headers['Referrer-Policy'] = 'strict-origin-when-cross-origin'
        
        return response


def setup_middleware(app: Flask):
    """
    Setup all middleware for the Flask application.
    
    Args:
        app: Flask application instance
    """
    # Initialize middleware components
    rate_limiter = RateLimiter(
        max_requests=app.config.get('RATE_LIMIT_REQUESTS', 100),
        window_minutes=app.config.get('RATE_LIMIT_WINDOW', 15)
    )
    request_logger = RequestLogger()
    security_headers = SecurityHeaders()
    
    @app.before_request
    def before_request():
        """Execute before each request."""
        # Store request start time
        g.start_time = time.time()
        
        # Generate client ID for rate limiting
        client_id = get_client_id(request)
        g.client_id = client_id
        
        # Check rate limiting (skip for health check)
        if request.endpoint != 'health_check' and app.config.get('ENABLE_RATE_LIMITING', True):
            if not rate_limiter.is_allowed(client_id):
                remaining = rate_limiter.get_remaining_requests(client_id)
                return jsonify({
                    'success': False,
                    'error': {
                        'code': 'RATE_LIMIT_EXCEEDED',
                        'message': 'Rate limit exceeded',
                        'details': f'Maximum {rate_limiter.max_requests} requests per {rate_limiter.window_minutes} minutes',
                        'remaining_requests': remaining
                    },
                    'timestamp': datetime.utcnow().isoformat()
                }), 429
        
        # Validate content type for POST requests
        if request.method == 'POST':
            content_type = request.headers.get('Content-Type', '')
            if not content_type.startswith(('application/json', 'multipart/form-data')):
                return jsonify({
                    'success': False,
                    'error': {
                        'code': 'INVALID_CONTENT_TYPE',
                        'message': 'Invalid content type',
                        'details': 'Content-Type must be application/json or multipart/form-data'
                    },
                    'timestamp': datetime.utcnow().isoformat()
                }), 400
        
        # Validate request size
        max_size = app.config.get('MAX_CONTENT_LENGTH', 10 * 1024 * 1024)  # 10MB default
        if request.content_length and request.content_length > max_size:
            return jsonify({
                'success': False,
                'error': {
                    'code': 'REQUEST_TOO_LARGE',
                    'message': 'Request entity too large',
                    'details': f'Maximum request size is {max_size} bytes'
                },
                'timestamp': datetime.utcnow().isoformat()
            }), 413
    
    @app.after_request
    def after_request(response):
        """Execute after each request."""
        # Add security headers
        if app.config.get('ENABLE_SECURITY_HEADERS', True):
            response = security_headers.add_security_headers(response)
        
        # Log request
        if hasattr(g, 'start_time'):
            request_logger.log_request(g.start_time, response.status_code)
        
        # Add rate limiting headers
        if hasattr(g, 'client_id') and app.config.get('ENABLE_RATE_LIMITING', True):
            remaining = rate_limiter.get_remaining_requests(g.client_id)
            response.headers['X-RateLimit-Limit'] = str(rate_limiter.max_requests)
            response.headers['X-RateLimit-Remaining'] = str(remaining)
            response.headers['X-RateLimit-Window'] = f"{rate_limiter.window_minutes}m"
        
        return response


def get_client_id(request) -> str:
    """
    Generate client ID for rate limiting.
    
    Args:
        request: Flask request object
        
    Returns:
        Client identifier string
    """
    # Use IP address and User-Agent for client identification
    ip_address = request.remote_addr or 'unknown'
    user_agent = request.headers.get('User-Agent', 'unknown')
    
    # Create hash of IP + User-Agent for privacy
    client_data = f"{ip_address}:{user_agent}"
    client_id = hashlib.sha256(client_data.encode()).hexdigest()[:16]
    
    return client_id


def require_json(f: Callable) -> Callable:
    """
    Decorator to require JSON content type.
    
    Args:
        f: Function to decorate
        
    Returns:
        Decorated function
    """
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not request.is_json:
            return jsonify({
                'success': False,
                'error': {
                    'code': 'INVALID_CONTENT_TYPE',
                    'message': 'Request must be JSON',
                    'details': 'Content-Type must be application/json'
                },
                'timestamp': datetime.utcnow().isoformat()
            }), 400
        return f(*args, **kwargs)
    return decorated_function


def validate_json_schema(schema: Dict[str, Any]) -> Callable:
    """
    Decorator to validate JSON request against schema.
    
    Args:
        schema: JSON schema for validation
        
    Returns:
        Decorator function
    """
    def decorator(f: Callable) -> Callable:
        @wraps(f)
        def decorated_function(*args, **kwargs):
            try:
                data = request.get_json()
                if data is None:
                    return jsonify({
                        'success': False,
                        'error': {
                            'code': 'INVALID_JSON',
                            'message': 'Invalid JSON data',
                            'details': 'Request body must contain valid JSON'
                        },
                        'timestamp': datetime.utcnow().isoformat()
                    }), 400
                
                # Basic schema validation (simplified)
                for field, field_type in schema.items():
                    if field not in data:
                        return jsonify({
                            'success': False,
                            'error': {
                                'code': 'MISSING_FIELD',
                                'message': f'Missing required field: {field}',
                                'details': f'Field "{field}" is required'
                            },
                            'timestamp': datetime.utcnow().isoformat()
                        }), 400
                    
                    if not isinstance(data[field], field_type):
                        return jsonify({
                            'success': False,
                            'error': {
                                'code': 'INVALID_FIELD_TYPE',
                                'message': f'Invalid type for field: {field}',
                                'details': f'Field "{field}" must be of type {field_type.__name__}'
                            },
                            'timestamp': datetime.utcnow().isoformat()
                        }), 400
                
                return f(*args, **kwargs)
                
            except Exception as e:
                logger.error(f"JSON validation error: {e}")
                return jsonify({
                    'success': False,
                    'error': {
                        'code': 'VALIDATION_ERROR',
                        'message': 'Request validation failed',
                        'details': str(e)
                    },
                    'timestamp': datetime.utcnow().isoformat()
                }), 400
        
        return decorated_function
    return decorator


def handle_file_upload(max_size: int = 1024 * 1024, allowed_extensions: set = None) -> Callable:
    """
    Decorator to handle file uploads with validation.
    
    Args:
        max_size: Maximum file size in bytes
        allowed_extensions: Set of allowed file extensions
        
    Returns:
        Decorator function
    """
    if allowed_extensions is None:
        allowed_extensions = {'.sol', '.txt'}
    
    def decorator(f: Callable) -> Callable:
        @wraps(f)
        def decorated_function(*args, **kwargs):
            # Check if file is present
            if 'file' not in request.files:
                return jsonify({
                    'success': False,
                    'error': {
                        'code': 'NO_FILE',
                        'message': 'No file provided',
                        'details': 'Request must include a file upload'
                    },
                    'timestamp': datetime.utcnow().isoformat()
                }), 400
            
            file = request.files['file']
            
            # Check if file is selected
            if file.filename == '':
                return jsonify({
                    'success': False,
                    'error': {
                        'code': 'NO_FILE_SELECTED',
                        'message': 'No file selected',
                        'details': 'Please select a file to upload'
                    },
                    'timestamp': datetime.utcnow().isoformat()
                }), 400
            
            # Check file extension
            if file.filename:
                file_ext = '.' + file.filename.rsplit('.', 1)[1].lower() if '.' in file.filename else ''
                if file_ext not in allowed_extensions:
                    return jsonify({
                        'success': False,
                        'error': {
                            'code': 'INVALID_FILE_TYPE',
                            'message': 'Invalid file type',
                            'details': f'Allowed extensions: {", ".join(allowed_extensions)}'
                        },
                        'timestamp': datetime.utcnow().isoformat()
                    }), 400
            
            # Check file size
            file.seek(0, 2)  # Seek to end
            file_size = file.tell()
            file.seek(0)  # Reset to beginning
            
            if file_size > max_size:
                return jsonify({
                    'success': False,
                    'error': {
                        'code': 'FILE_TOO_LARGE',
                        'message': 'File too large',
                        'details': f'Maximum file size is {max_size} bytes'
                    },
                    'timestamp': datetime.utcnow().isoformat()
                }), 400
            
            return f(*args, **kwargs)
        
        return decorated_function
    return decorator