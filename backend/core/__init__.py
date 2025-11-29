"""Core configuration and utilities for ShikshaSetu.

This package contains core infrastructure components:
- Application configuration (settings)
- Database connection management
- Exception classes
- Rate limiting
- Caching (Redis)
- Monitoring and metrics
"""

from .config import settings, Settings
from .database import get_db_session, SessionLocal, engine
from .exceptions import ShikshaSetuException
from .cache import get_redis

__all__ = [
    'settings',
    'Settings',
    'get_db_session',
    'SessionLocal', 
    'engine',
    'ShikshaSetuException',
    'get_redis',
]
