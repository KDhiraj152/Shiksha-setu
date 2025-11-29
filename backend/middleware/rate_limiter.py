"""
Rate Limiter Implementation

Token bucket algorithm for rate limiting API requests.
Supports per-user, per-IP, and global rate limits.

Features:
- Redis-backed for distributed rate limiting
- In-memory fallback when Redis unavailable
- Per-endpoint configuration
- Sliding window support
"""
import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class RateLimitConfig:
    """Rate limit configuration."""
    requests_per_minute: int = 60
    requests_per_hour: int = 1000
    burst_size: int = 10  # Max burst above rate limit
    

# Default rate limits per endpoint category
RATE_LIMITS = {
    "default": RateLimitConfig(requests_per_minute=60, requests_per_hour=1000),
    "process": RateLimitConfig(requests_per_minute=10, requests_per_hour=100),  # Heavy processing
    "tts": RateLimitConfig(requests_per_minute=20, requests_per_hour=200),
    "qa": RateLimitConfig(requests_per_minute=30, requests_per_hour=500),
    "upload": RateLimitConfig(requests_per_minute=5, requests_per_hour=50),
    "auth": RateLimitConfig(requests_per_minute=10, requests_per_hour=100),
}


class TokenBucket:
    """
    Token bucket for rate limiting.
    
    Tokens are added at a fixed rate up to a maximum.
    Each request consumes one token.
    """
    
    def __init__(
        self,
        rate: float,  # Tokens per second
        capacity: int  # Maximum tokens
    ):
        self.rate = rate
        self.capacity = capacity
        self.tokens = capacity
        self.last_update = time.time()
    
    def _refill(self):
        """Add tokens based on elapsed time."""
        now = time.time()
        elapsed = now - self.last_update
        tokens_to_add = elapsed * self.rate
        self.tokens = min(self.capacity, self.tokens + tokens_to_add)
        self.last_update = now
    
    def consume(self, tokens: int = 1) -> Tuple[bool, float]:
        """
        Try to consume tokens.
        
        Returns:
            Tuple of (success, retry_after_seconds)
        """
        self._refill()
        
        if self.tokens >= tokens:
            self.tokens -= tokens
            return True, 0
        
        # Calculate wait time for tokens to regenerate
        tokens_needed = tokens - self.tokens
        wait_time = tokens_needed / self.rate
        return False, wait_time


class InMemoryRateLimiter:
    """Simple in-memory rate limiter."""
    
    def __init__(self):
        self._buckets: Dict[str, TokenBucket] = {}
    
    def _get_bucket(self, key: str, config: RateLimitConfig) -> TokenBucket:
        """Get or create bucket for key."""
        if key not in self._buckets:
            rate = config.requests_per_minute / 60.0
            capacity = config.requests_per_minute + config.burst_size
            self._buckets[key] = TokenBucket(rate, capacity)
        return self._buckets[key]
    
    def check(
        self,
        key: str,
        config: RateLimitConfig
    ) -> Tuple[bool, float, Dict[str, int]]:
        """
        Check if request is allowed.
        
        Returns:
            Tuple of (allowed, retry_after, headers)
        """
        bucket = self._get_bucket(key, config)
        allowed, retry_after = bucket.consume()
        
        headers = {
            "X-RateLimit-Limit": config.requests_per_minute,
            "X-RateLimit-Remaining": int(bucket.tokens),
            "X-RateLimit-Reset": int(time.time() + (config.requests_per_minute / bucket.rate)),
        }
        
        if not allowed:
            headers["Retry-After"] = int(retry_after) + 1
        
        return allowed, retry_after, headers


class RedisRateLimiter:
    """Redis-backed distributed rate limiter."""
    
    def __init__(self, redis_client=None):
        self._redis = redis_client
    
    def _get_redis(self):
        """Get Redis client."""
        if self._redis is not None:
            return self._redis
        
        try:
            from ..core.cache import get_redis
            return get_redis()
        except Exception:
            return None
    
    async def check_async(
        self,
        key: str,
        config: RateLimitConfig
    ) -> Tuple[bool, float, Dict[str, int]]:
        """
        Async check if request is allowed using sliding window.
        
        Returns:
            Tuple of (allowed, retry_after, headers)
        """
        redis = self._get_redis()
        if not redis:
            # Fallback to basic check
            return True, 0, {}
        
        try:
            now = time.time()
            window_start = now - 60  # 1-minute window
            
            # Use sorted set for sliding window
            rate_key = f"ratelimit:{key}:minute"
            
            # Remove old entries
            redis.zremrangebyscore(rate_key, 0, window_start)
            
            # Count current entries
            current_count = redis.zcard(rate_key)
            
            headers = {
                "X-RateLimit-Limit": config.requests_per_minute,
                "X-RateLimit-Remaining": max(0, config.requests_per_minute - current_count),
                "X-RateLimit-Reset": int(now + 60),
            }
            
            if current_count >= config.requests_per_minute:
                # Get oldest entry to calculate retry time
                oldest = redis.zrange(rate_key, 0, 0, withscores=True)
                retry_after = 60 - (now - oldest[0][1]) if oldest else 60
                headers["Retry-After"] = int(retry_after) + 1
                return False, retry_after, headers
            
            # Add current request
            redis.zadd(rate_key, {f"{now}": now})
            redis.expire(rate_key, 120)  # 2-minute expiry for safety
            
            return True, 0, headers
            
        except Exception as e:
            logger.warning(f"Redis rate limit error: {e}")
            return True, 0, {}
    
    def check(
        self,
        key: str,
        config: RateLimitConfig
    ) -> Tuple[bool, float, Dict[str, int]]:
        """
        Sync check if request is allowed.
        
        Returns:
            Tuple of (allowed, retry_after, headers)
        """
        redis = self._get_redis()
        if not redis:
            return True, 0, {}
        
        try:
            now = time.time()
            window_start = now - 60
            
            rate_key = f"ratelimit:{key}:minute"
            
            pipe = redis.pipeline()
            pipe.zremrangebyscore(rate_key, 0, window_start)
            pipe.zcard(rate_key)
            pipe.zrange(rate_key, 0, 0, withscores=True)
            pipe.zadd(rate_key, {f"{now}": now})
            pipe.expire(rate_key, 120)
            
            results = pipe.execute()
            current_count = results[1]
            oldest = results[2]
            
            headers = {
                "X-RateLimit-Limit": config.requests_per_minute,
                "X-RateLimit-Remaining": max(0, config.requests_per_minute - current_count),
                "X-RateLimit-Reset": int(now + 60),
            }
            
            if current_count >= config.requests_per_minute:
                retry_after = 60 - (now - oldest[0][1]) if oldest else 60
                headers["Retry-After"] = int(retry_after) + 1
                return False, retry_after, headers
            
            return True, 0, headers
            
        except Exception as e:
            logger.warning(f"Redis rate limit error: {e}")
            return True, 0, {}


class RateLimiter:
    """
    Unified rate limiter with Redis and in-memory support.
    """
    
    def __init__(self, redis_client=None):
        self._redis_limiter = RedisRateLimiter(redis_client)
        self._memory_limiter = InMemoryRateLimiter()
    
    def _make_key(
        self,
        identifier: str,
        endpoint_type: str = "default"
    ) -> str:
        """Create rate limit key."""
        return f"{endpoint_type}:{identifier}"
    
    def check(
        self,
        identifier: str,
        endpoint_type: str = "default"
    ) -> Tuple[bool, float, Dict[str, int]]:
        """
        Check if request is allowed.
        
        Args:
            identifier: User ID, IP address, or API key
            endpoint_type: Type of endpoint for config lookup
            
        Returns:
            Tuple of (allowed, retry_after, headers)
        """
        config = RATE_LIMITS.get(endpoint_type, RATE_LIMITS["default"])
        key = self._make_key(identifier, endpoint_type)
        
        # Try Redis first
        allowed, retry_after, headers = self._redis_limiter.check(key, config)
        
        if not headers:
            # Redis unavailable, use in-memory
            allowed, retry_after, headers = self._memory_limiter.check(key, config)
        
        return allowed, retry_after, headers
    
    async def check_async(
        self,
        identifier: str,
        endpoint_type: str = "default"
    ) -> Tuple[bool, float, Dict[str, int]]:
        """
        Async check if request is allowed.
        
        Args:
            identifier: User ID, IP address, or API key
            endpoint_type: Type of endpoint for config lookup
            
        Returns:
            Tuple of (allowed, retry_after, headers)
        """
        config = RATE_LIMITS.get(endpoint_type, RATE_LIMITS["default"])
        key = self._make_key(identifier, endpoint_type)
        
        # Try Redis first
        allowed, retry_after, headers = await self._redis_limiter.check_async(key, config)
        
        if not headers:
            # Redis unavailable, use in-memory
            allowed, retry_after, headers = self._memory_limiter.check(key, config)
        
        return allowed, retry_after, headers


# Global rate limiter
_rate_limiter: Optional[RateLimiter] = None


def get_rate_limiter() -> RateLimiter:
    """Get global rate limiter."""
    global _rate_limiter
    if _rate_limiter is None:
        _rate_limiter = RateLimiter()
    return _rate_limiter


# FastAPI middleware
from fastapi import Request, Response
from fastapi.responses import JSONResponse


async def rate_limit_middleware(request: Request, call_next):
    """
    FastAPI middleware for rate limiting.
    
    Add to app:
        app.middleware("http")(rate_limit_middleware)
    """
    # Get identifier (prefer user ID, fallback to IP)
    identifier = getattr(request.state, "user_id", None)
    if not identifier:
        identifier = request.client.host if request.client else "unknown"
    
    # Determine endpoint type
    endpoint_type = "default"
    path = request.url.path
    
    if "/process" in path or "/pipeline" in path:
        endpoint_type = "process"
    elif "/tts" in path or "/audio" in path:
        endpoint_type = "tts"
    elif "/qa" in path or "/question" in path:
        endpoint_type = "qa"
    elif "/upload" in path:
        endpoint_type = "upload"
    elif "/auth" in path or "/login" in path:
        endpoint_type = "auth"
    
    # Check rate limit
    limiter = get_rate_limiter()
    allowed, retry_after, headers = await limiter.check_async(identifier, endpoint_type)
    
    if not allowed:
        response = JSONResponse(
            status_code=429,
            content={
                "error": "rate_limit_exceeded",
                "message": f"Rate limit exceeded. Try again in {int(retry_after)} seconds.",
                "retry_after": int(retry_after)
            }
        )
        for key, value in headers.items():
            response.headers[key] = str(value)
        return response
    
    # Process request
    response = await call_next(request)
    
    # Add rate limit headers
    for key, value in headers.items():
        response.headers[key] = str(value)
    
    return response


# Dependency for per-route rate limiting
from fastapi import Depends, HTTPException


class RateLimitDependency:
    """
    FastAPI dependency for per-route rate limiting.
    
    Usage:
        @app.get("/expensive", dependencies=[Depends(RateLimitDependency("process"))])
        async def expensive_endpoint():
            ...
    """
    
    def __init__(self, endpoint_type: str = "default"):
        self.endpoint_type = endpoint_type
    
    async def __call__(self, request: Request):
        identifier = getattr(request.state, "user_id", None)
        if not identifier:
            identifier = request.client.host if request.client else "unknown"
        
        limiter = get_rate_limiter()
        allowed, retry_after, headers = await limiter.check_async(identifier, self.endpoint_type)
        
        if not allowed:
            raise HTTPException(
                status_code=429,
                detail={
                    "error": "rate_limit_exceeded",
                    "message": f"Rate limit exceeded. Try again in {int(retry_after)} seconds.",
                    "retry_after": int(retry_after)
                },
                headers={k: str(v) for k, v in headers.items()}
            )
        
        # Store headers for response
        request.state.rate_limit_headers = headers
