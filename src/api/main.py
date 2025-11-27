"""
ShikshaSetu Main Application - Streamlined and modular.

This is the new main application file that uses the refactored structure:
- Route modules in src/api/routes/
- Core configuration in src/core/
- Centralized logging in src/utils/logging_config.py
- Middleware in src/api/middleware.py
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from ..core.config import settings
from ..core.exceptions import ShikshaSetuException
from ..utils.logging import setup_logging
from .middleware import (
    SecurityHeadersMiddleware,
    RequestTimingMiddleware,
    RequestLoggingMiddleware,
    exception_handler,
    generic_exception_handler
)
from ..core.rate_limiter import create_rate_limiter
from ..cache import get_redis
from .routes import health_router, auth_router, content_router, qa_router

# Initialize logging
logger = setup_logging()

# Initialize FastAPI app
app = FastAPI(
    title=settings.APP_NAME,
    description="Production-grade multilingual education content processing with AI/ML pipeline",
    version=settings.APP_VERSION,
    debug=settings.DEBUG
)

# Add custom middleware (order matters - they execute in reverse order of addition)
# Security headers middleware (added first to apply last)
app.add_middleware(SecurityHeadersMiddleware)

# Timing middleware
app.add_middleware(RequestTimingMiddleware)

# Logging middleware (only in development)
if settings.ENVIRONMENT != "production":
    app.add_middleware(RequestLoggingMiddleware)

# Add CORS middleware (added last to apply first)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=settings.ALLOW_CREDENTIALS,
    allow_methods=settings.ALLOWED_METHODS,
    allow_headers=settings.ALLOWED_HEADERS,
    expose_headers=["X-Process-Time", "X-RateLimit-Limit", "X-RateLimit-Remaining"]
)

# Add rate limiting middleware (production ready)
if settings.RATE_LIMIT_ENABLED:
    try:
        redis_client = get_redis()
        rate_limiter = create_rate_limiter(redis_client)
        app.add_middleware(rate_limiter)
        logger.info("Rate limiting enabled with Redis backend")
    except Exception as e:
        logger.warning(f"Failed to initialize Redis for rate limiting, using in-memory: {e}")
        rate_limiter = create_rate_limiter(None)
        app.add_middleware(rate_limiter)
        logger.info("Rate limiting enabled with in-memory backend")

# Exception handlers
app.add_exception_handler(ShikshaSetuException, exception_handler)
app.add_exception_handler(Exception, generic_exception_handler)

# Include route modules
app.include_router(health_router)
app.include_router(auth_router)
app.include_router(content_router)
app.include_router(qa_router)


@app.on_event("startup")
async def startup_event():
    """Application startup tasks."""
    logger.info(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}")
    logger.info(f"Environment: {settings.ENVIRONMENT}")
    logger.info(f"Debug mode: {settings.DEBUG}")
    
    # Initialize database
    from ..database import init_db
    try:
        init_db()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")


@app.on_event("shutdown")
async def shutdown_event():
    """Application shutdown tasks."""
    logger.info(f"Shutting down {settings.APP_NAME}")


# For backward compatibility, expose the app as the default export
__all__ = ["app"]
