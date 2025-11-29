"""Utility modules for ShikshaSetu application.

This package contains utility functions and classes for:
- Logging configuration
- Environment validation
- Input sanitization
- Circuit breaker pattern implementation
- Model loading utilities (lazy loading with memory management)
- Request context management
- Authentication helpers
"""

from .logging import get_logger, setup_logging
from .sanitizer import InputSanitizer, ValidationError
from .circuit_breaker import CircuitBreaker, CircuitBreakerOpen, circuit_breaker
from .request_context import get_request_id, set_request_id
from .env import EnvironmentValidator, validate_environment

# Lazy imports to avoid circular dependencies
def get_lazy_model_loader():
    """Get lazy model loader instance."""
    from .lazy_loader import get_model_loader
    return get_model_loader()

__all__ = [
    'get_logger',
    'setup_logging', 
    'InputSanitizer',
    'ValidationError',
    'CircuitBreaker',
    'CircuitBreakerOpen',
    'circuit_breaker',
    'get_request_id',
    'set_request_id',
    'EnvironmentValidator',
    'validate_environment',
    'get_lazy_model_loader',
]
