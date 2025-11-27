"""
Rate limiting middleware for production environments.

Implements distributed rate limiting using Redis to protect against:
- DDoS attacks
- API abuse
- Resource exhaustion
- Credential stuffing

Features:
- Per-user and per-IP rate limiting
- Token bucket algorithm with burst support
- Distributed rate limiting via Redis
- Configurable limits per endpoint
- Graceful degradation if Redis is unavailable
"""

import time
import hashlib
from typing import Callable, Optional
from fastapi import Request, Response, HTTPException, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from redis import Redis
from redis.exceptions import RedisError
import logging

from src.core.config import settings

logger = logging.getLogger(__name__)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Rate limiting middleware using token bucket algorithm.
    
    Implements distributed rate limiting with Redis for production environments.
    Falls back to in-memory rate limiting if Redis is unavailable.
    """
    
    def __init__(self, app, redis_client: Optional[Redis] = None):
        """
        Initialize rate limiter.
        
        Args:
            app: FastAPI application instance
            redis_client: Redis client for distributed rate limiting
        """
        super().__init__(app)
        self.redis_client = redis_client
        self.enabled = settings.RATE_LIMIT_ENABLED
        self.per_minute_limit = settings.RATE_LIMIT_PER_MINUTE
        self.per_hour_limit = settings.RATE_LIMIT_PER_HOUR
        self.burst_multiplier = settings.RATE_LIMIT_BURST_MULTIPLIER
        
        # In-memory fallback (for development or Redis unavailable)
        self.memory_store = {}
        
        # Endpoints exempt from rate limiting
        self.exempt_paths = [
            "/health",
            "/docs",
            "/redoc",
            "/openapi.json",
            "/metrics",  # Prometheus metrics
        ]
        
        # Custom limits for specific endpoints
        self.endpoint_limits = {
            "/api/v1/auth/login": (10, 100),  # (per_minute, per_hour)
            "/api/v1/auth/register": (5, 50),
            "/api/v1/upload": (5, 50),
            "/api/v1/translate": (30, 500),
            "/api/v1/simplify": (30, 500),
            "/api/v1/qa/search": (60, 1000),
        }
        
        logger.info(
            f"Rate limiting {'enabled' if self.enabled else 'disabled'}: "
            f"{self.per_minute_limit}/min, {self.per_hour_limit}/hour"
        )
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Process request and apply rate limiting.
        
        Args:
            request: Incoming HTTP request
            call_next: Next middleware in chain
            
        Returns:
            HTTP response or 429 if rate limited
        """
        # Skip if rate limiting is disabled
        if not self.enabled:
            return await call_next(request)
        
        # Skip exempt paths
        if any(request.url.path.startswith(path) for path in self.exempt_paths):
            return await call_next(request)
        
        # Get identifier (user ID or IP address)
        identifier = self._get_identifier(request)
        
        # Get rate limits for this endpoint
        minute_limit, hour_limit = self._get_endpoint_limits(request.url.path)
        
        # Check rate limits
        try:
            # Check minute limit
            minute_key = f"rate_limit:minute:{identifier}:{int(time.time() / 60)}"
            if not self._check_limit(minute_key, minute_limit, 60):
                return self._rate_limit_response(minute_limit, "minute")
            
            # Check hour limit
            hour_key = f"rate_limit:hour:{identifier}:{int(time.time() / 3600)}"
            if not self._check_limit(hour_key, hour_limit, 3600):
                return self._rate_limit_response(hour_limit, "hour")
            
            # Process request
            response = await call_next(request)
            
            # Add rate limit headers
            response.headers["X-RateLimit-Limit-Minute"] = str(minute_limit)
            response.headers["X-RateLimit-Limit-Hour"] = str(hour_limit)
            
            # Get remaining counts
            minute_count = self._get_count(minute_key)
            hour_count = self._get_count(hour_key)
            
            response.headers["X-RateLimit-Remaining-Minute"] = str(max(0, minute_limit - minute_count))
            response.headers["X-RateLimit-Remaining-Hour"] = str(max(0, hour_limit - hour_count))
            
            return response
            
        except Exception as e:
            logger.error(f"Rate limiting error: {e}", exc_info=True)
            # Fail open - allow request if rate limiting fails
            return await call_next(request)
    
    def _get_identifier(self, request: Request) -> str:
        """
        Get unique identifier for rate limiting.
        
        Prefers authenticated user ID, falls back to IP address.
        
        Args:
            request: HTTP request
            
        Returns:
            Unique identifier string
        """
        # Try to get user ID from request state (set by auth middleware)
        if hasattr(request.state, "user") and request.state.user:
            user_id = getattr(request.state.user, "id", None)
            if user_id:
                return f"user:{user_id}"
        
        # Fall back to IP address
        # Handle X-Forwarded-For header for proxied requests
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            # Get first IP in chain (client IP)
            client_ip = forwarded_for.split(",")[0].strip()
        else:
            client_ip = request.client.host if request.client else "unknown"
        
        return f"ip:{client_ip}"
    
    def _get_endpoint_limits(self, path: str) -> tuple[int, int]:
        """
        Get rate limits for specific endpoint.
        
        Args:
            path: Request path
            
        Returns:
            Tuple of (per_minute_limit, per_hour_limit)
        """
        # Check for exact match
        if path in self.endpoint_limits:
            return self.endpoint_limits[path]
        
        # Check for prefix match
        for endpoint_path, limits in self.endpoint_limits.items():
            if path.startswith(endpoint_path):
                return limits
        
        # Return default limits
        return self.per_minute_limit, self.per_hour_limit
    
    def _check_limit(self, key: str, limit: int, window: int) -> bool:
        """
        Check if request is within rate limit.
        
        Uses token bucket algorithm with burst support.
        
        Args:
            key: Redis key for this time window
            limit: Maximum requests allowed
            window: Time window in seconds
            
        Returns:
            True if request is allowed, False if rate limited
        """
        # Apply burst multiplier
        max_tokens = int(limit * self.burst_multiplier)
        
        try:
            if self.redis_client and settings.RATE_LIMIT_STORAGE == "redis":
                # Use Redis for distributed rate limiting
                return self._check_limit_redis(key, limit, max_tokens, window)
            else:
                # Use in-memory fallback
                return self._check_limit_memory(key, limit, max_tokens, window)
        except RedisError as e:
            logger.warning(f"Redis error, falling back to memory: {e}")
            return self._check_limit_memory(key, limit, max_tokens, window)
    
    def _check_limit_redis(self, key: str, limit: int, max_tokens: int, window: int) -> bool:
        """
        Check rate limit using Redis.
        
        Args:
            key: Redis key
            limit: Rate limit
            max_tokens: Maximum burst tokens
            window: Time window in seconds
            
        Returns:
            True if allowed, False if rate limited
        """
        # Increment counter
        count = self.redis_client.incr(key)
        
        # Set expiration on first request
        if count == 1:
            self.redis_client.expire(key, window)
        
        # Check against limit with burst allowance
        return count <= max_tokens
    
    def _check_limit_memory(self, key: str, limit: int, max_tokens: int, window: int) -> bool:
        """
        Check rate limit using in-memory store.
        
        Args:
            key: Memory key
            limit: Rate limit
            max_tokens: Maximum burst tokens
            window: Time window in seconds
            
        Returns:
            True if allowed, False if rate limited
        """
        current_time = time.time()
        
        if key not in self.memory_store:
            self.memory_store[key] = {"count": 0, "reset_at": current_time + window}
        
        # Reset if window expired
        if current_time >= self.memory_store[key]["reset_at"]:
            self.memory_store[key] = {"count": 0, "reset_at": current_time + window}
        
        # Increment and check
        self.memory_store[key]["count"] += 1
        
        # Clean up old entries
        self._cleanup_memory_store(current_time)
        
        return self.memory_store[key]["count"] <= max_tokens
    
    def _get_count(self, key: str) -> int:
        """
        Get current request count for key.
        
        Args:
            key: Rate limit key
            
        Returns:
            Current count
        """
        try:
            if self.redis_client and settings.RATE_LIMIT_STORAGE == "redis":
                count = self.redis_client.get(key)
                return int(count) if count else 0
            else:
                return self.memory_store.get(key, {}).get("count", 0)
        except Exception:
            return 0
    
    def _cleanup_memory_store(self, current_time: float):
        """
        Clean up expired entries from memory store.
        
        Args:
            current_time: Current timestamp
        """
        # Only cleanup occasionally to avoid overhead
        if len(self.memory_store) > 10000:
            expired_keys = [
                k for k, v in self.memory_store.items()
                if current_time >= v.get("reset_at", float("inf"))
            ]
            for key in expired_keys:
                del self.memory_store[key]
    
    def _rate_limit_response(self, limit: int, period: str) -> JSONResponse:
        """
        Create rate limit exceeded response.
        
        Args:
            limit: Rate limit that was exceeded
            period: Time period (minute/hour)
            
        Returns:
            429 JSON response
        """
        return JSONResponse(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            content={
                "detail": f"Rate limit exceeded. Maximum {limit} requests per {period}.",
                "error": "rate_limit_exceeded",
                "limit": limit,
                "period": period
            },
            headers={
                "Retry-After": "60" if period == "minute" else "3600"
            }
        )


def create_rate_limiter(redis_client: Optional[Redis] = None) -> RateLimitMiddleware:
    """
    Factory function to create rate limiter middleware.
    
    Args:
        redis_client: Redis client for distributed rate limiting
        
    Returns:
        Configured rate limiter middleware
    """
    def middleware_factory(app):
        return RateLimitMiddleware(app, redis_client)
    
    return middleware_factory
