"""API middleware for security, logging, and performance monitoring."""
import time
import uuid
from typing import Callable
from datetime import datetime, timezone

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from ..core.config import settings
from ..core.exceptions import ShikshaSetuException
from ..utils.logging import get_logger
from ..utils.request_context import set_request_id

logger = get_logger(__name__)


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Add unique request ID for distributed tracing."""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Get or generate request ID
        request_id = request.headers.get("X-Request-ID")
        if not request_id:
            request_id = str(uuid.uuid4())
        
        # Store in request state for access in route handlers
        request.state.request_id = request_id
        
        # Store in context var for access in Celery tasks and logging
        set_request_id(request_id)
        
        # Process request
        response = await call_next(request)
        
        # Add request ID to response headers
        response.headers["X-Request-ID"] = request_id
        
        return response


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Add security headers to all responses."""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        response = await call_next(request)
        
        # Security headers - prevent MIME type sniffing
        response.headers["X-Content-Type-Options"] = "nosniff"
        
        # Security headers - prevent clickjacking
        response.headers["X-Frame-Options"] = "SAMEORIGIN"
        
        # Security headers - XSS protection
        response.headers["X-XSS-Protection"] = "1; mode=block"
        
        # Security headers - Referrer policy
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        
        # Content Security Policy
        response.headers["Content-Security-Policy"] = "default-src 'self'; script-src 'self' 'unsafe-inline' 'unsafe-eval'; style-src 'self' 'unsafe-inline';"
        
        # HSTS header for production
        if settings.ENVIRONMENT == "production":
            response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains; preload"
        
        # Permissions Policy
        response.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"
        
        return response


class RequestTimingMiddleware(BaseHTTPMiddleware):
    """Add request timing and log slow requests."""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start_time = time.time()
        
        # Process request
        response = await call_next(request)
        
        # Calculate processing time
        process_time = time.time() - start_time
        response.headers["X-Process-Time"] = f"{process_time:.4f}"
        
        # Log slow requests
        if process_time > settings.SLOW_REQUEST_THRESHOLD:
            logger.warning(
                f"Slow request detected: {request.method} {request.url.path} "
                f"took {process_time:.2f}s"
            )
        
        return response


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Log all API requests with request ID correlation."""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Get request ID from request state (set by RequestIDMiddleware)
        request_id = getattr(request.state, "request_id", "unknown")
        
        # Log request with request ID
        logger.info(
            f"Request: {request.method} {request.url.path} "
            f"from {request.client.host if request.client else 'unknown'}",
            extra={"request_id": request_id}
        )
        
        # Process request
        response = await call_next(request)
        
        # Log response with request ID
        logger.info(
            f"Response: {request.method} {request.url.path} "
            f"status={response.status_code}",
            extra={"request_id": request_id}
        )
        
        return response


def exception_handler(request: Request, exc: ShikshaSetuException) -> JSONResponse:
    """Handle custom ShikshaSetu exceptions."""
    logger.error(
        f"ShikshaSetuException: {exc.error_code} - {exc.detail} "
        f"(status={exc.status_code}) for {request.method} {request.url.path}"
    )
    
    return JSONResponse(
        status_code=exc.status_code,
        content=exc.to_dict()
    )


def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle unexpected exceptions."""
    logger.exception(
        f"Unhandled exception for {request.method} {request.url.path}: {str(exc)}"
    )
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "INTERNAL_SERVER_ERROR",
            "detail": "An unexpected error occurred" if settings.ENVIRONMENT == "production" else str(exc),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    )
