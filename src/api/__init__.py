"""
Smart Contract Security Analyzer API module.
"""

from .app import create_app, get_app, get_default_app
from .routes import register_routes
from .middleware import setup_middleware
from .utils import load_configuration

__all__ = [
    'create_app',
    'get_app', 
    'register_routes',
    'setup_middleware',
    'load_configuration'
]