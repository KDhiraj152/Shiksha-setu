"""
Request Logging Middleware

Issue: CODE-REVIEW-GPT #16 (MEDIUM)
Purpose: Comprehensive HTTP request/response logging
"""

import time
import json
import logging
from typing import Callable
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import Message

logger = logging.getLogger("request_logger")


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware for comprehensive request/response logging.
    
    Logs:
    - Request method, path, headers, query params
    - Response status, headers, duration
    - Request/response body (configurable)
    - User information (if authenticated)
    """
    
    def __init__(
        self,
        app,
        log_request_body: bool = False,
        log_response_body: bool = False,
        exclude_paths: list = None
    ):
        super().__init__(app)
        self.log_request_body = log_request_body
        self.log_response_body = log_response_body
        self.exclude_paths = exclude_paths or ["/health", "/metrics", "/docs"]
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request and log details."""
        
        # Skip excluded paths
        if any(request.url.path.startswith(path) for path in self.exclude_paths):
            return await call_next(request)
        
        # Record start time
        start_time = time.time()
        
        # Extract request info
        request_info = {
            "method": request.method,
            "path": request.url.path,
            "query_params": dict(request.query_params),
            "client_host": request.client.host if request.client else None,
            "user_agent": request.headers.get("user-agent"),
            "request_id": request.headers.get("x-request-id")
        }
        
        # Log request body if enabled
        if self.log_request_body and request.method in ["POST", "PUT", "PATCH"]:
            try:
                body = await request.body()
                request_info["body_size"] = len(body)
                # Store body for later re-reading
                request.state.body = body
            except Exception as e:
                logger.warning(f"Could not read request body: {e}")
        
        # Get user info if authenticated
        if hasattr(request.state, "user"):
            request_info["user_id"] = request.state.user.get("id")
            request_info["user_role"] = request.state.user.get("role")
        
        logger.info(f"Request started: {json.dumps(request_info)}")
        
        # Process request
        try:
            response = await call_next(request)
            duration = time.time() - start_time
            
            # Log response
            response_info = {
                **request_info,
                "status_code": response.status_code,
                "duration_ms": round(duration * 1000, 2)
            }
            
            log_level = logging.INFO if response.status_code < 400 else logging.WARNING
            logger.log(
                log_level,
                f"Request completed: {json.dumps(response_info)}"
            )
            
            return response
            
        except Exception as e:
            duration = time.time() - start_time
            logger.error(
                f"Request failed: {json.dumps({**request_info, 'error': str(e), 'duration_ms': round(duration * 1000, 2)})}"
            )
            raise


# Helper function to configure request logging
def setup_request_logging(log_level: str = "INFO"):
    """Configure request logger."""
    request_logger = logging.getLogger("request_logger")
    request_logger.setLevel(getattr(logging, log_level.upper()))
    
    # Create handler if not exists
    if not request_logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        request_logger.addHandler(handler)
    
    return request_logger
