"""
Middleware for Error Tracking Context

Automatically adds request context to Sentry error reports
"""

from typing import Callable
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from ..services.error_tracking import (
    set_user_context,
    set_tag,
    set_context,
    add_breadcrumb
)


class SentryContextMiddleware(BaseHTTPMiddleware):
    """Middleware to add request context to Sentry errors."""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Add request context before processing."""
        
        # Add request context
        set_context("request", {
            "url": str(request.url),
            "method": request.method,
            "headers": dict(request.headers),
            "query_params": dict(request.query_params)
        })
        
        # Add tags
        set_tag("request_method", request.method)
        set_tag("endpoint", request.url.path)
        
        # Get request ID if available
        request_id = request.headers.get("X-Request-ID")
        if request_id:
            set_tag("request_id", request_id)
        
        # Get user info from request state (set by auth middleware)
        if hasattr(request.state, "user"):
            user = request.state.user
            set_user_context(
                user_id=str(user.get("id")),
                email=user.get("email"),
                username=user.get("username"),
                ip_address=request.client.host if request.client else None
            )
            set_tag("user_role", user.get("role", "unknown"))
        
        # Add breadcrumb
        add_breadcrumb(
            message=f"{request.method} {request.url.path}",
            category="http",
            level="info",
            data={
                "method": request.method,
                "url": str(request.url),
                "ip": request.client.host if request.client else None
            }
        )
        
        # Process request
        response = await call_next(request)
        
        # Add response context
        set_context("response", {
            "status_code": response.status_code,
            "headers": dict(response.headers)
        })
        
        set_tag("response_status", str(response.status_code))
        
        return response
