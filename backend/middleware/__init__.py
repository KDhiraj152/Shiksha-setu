"""Middleware package for ShikshaSetu.

This package provides middleware components for:
- Request/response logging
- Rate limiting
- Error handling
- Authentication
- CORS
"""

from .rate_limiter import (
    RateLimiter,
    RateLimitConfig,
    RateLimitDependency,
    rate_limit_middleware,
    get_rate_limiter,
    RATE_LIMITS,
)

__all__ = [
    'RateLimiter',
    'RateLimitConfig',
    'RateLimitDependency',
    'rate_limit_middleware',
    'get_rate_limiter',
    'RATE_LIMITS',
]
