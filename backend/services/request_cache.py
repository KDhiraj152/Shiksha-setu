"""
Request Caching Service

Redis-based caching for expensive API operations like:
- Text simplification
- Translation
- Content validation

Features:
- Content-based cache keys (same input = cache hit)
- Configurable TTL per operation type
- Automatic cache invalidation
- Fallback to in-memory cache if Redis unavailable
"""
import hashlib
import json
import logging
import time
from functools import wraps
from typing import Any, Callable, Optional, TypeVar, Dict
from dataclasses import dataclass

logger = logging.getLogger(__name__)

T = TypeVar('T')


@dataclass
class CacheConfig:
    """Configuration for a cache operation."""
    ttl_seconds: int = 3600  # 1 hour default
    prefix: str = "shiksha"
    include_version: bool = True
    compress: bool = False


# Default cache configurations per operation type
CACHE_CONFIGS = {
    "simplify": CacheConfig(ttl_seconds=86400, prefix="simplify"),      # 24 hours
    "translate": CacheConfig(ttl_seconds=86400, prefix="translate"),    # 24 hours
    "validate": CacheConfig(ttl_seconds=3600, prefix="validate"),       # 1 hour
    "embedding": CacheConfig(ttl_seconds=604800, prefix="embed"),       # 7 days
    "tts": CacheConfig(ttl_seconds=86400, prefix="tts"),               # 24 hours
    "qa_answer": CacheConfig(ttl_seconds=3600, prefix="qa"),           # 1 hour
}


class InMemoryCache:
    """Simple in-memory cache as fallback."""
    
    def __init__(self, max_size: int = 1000):
        self._cache: Dict[str, tuple] = {}  # key -> (value, expiry_time)
        self._max_size = max_size
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        if key not in self._cache:
            return None
        
        value, expiry = self._cache[key]
        if time.time() > expiry:
            del self._cache[key]
            return None
        
        return value
    
    def set(self, key: str, value: Any, ttl: int):
        """Set value in cache with TTL."""
        # Evict if at capacity (simple LRU-ish eviction)
        if len(self._cache) >= self._max_size:
            # Remove oldest 10%
            to_remove = sorted(
                self._cache.keys(),
                key=lambda k: self._cache[k][1]
            )[:self._max_size // 10]
            for k in to_remove:
                del self._cache[k]
        
        self._cache[key] = (value, time.time() + ttl)
    
    def delete(self, key: str):
        """Delete key from cache."""
        self._cache.pop(key, None)
    
    def clear(self):
        """Clear all cache."""
        self._cache.clear()


class RequestCache:
    """
    Redis-based request cache with fallback.
    
    Caches expensive computation results to improve response times
    and reduce load on ML models.
    """
    
    def __init__(self, redis_client=None):
        """
        Initialize request cache.
        
        Args:
            redis_client: Optional Redis client. If None, uses global or fallback.
        """
        self._redis = redis_client
        self._fallback = InMemoryCache()
        self._stats = {
            "hits": 0,
            "misses": 0,
            "sets": 0,
            "errors": 0,
        }
        
        # Get version for cache invalidation
        try:
            from ..core.config import settings
            self._version = getattr(settings, 'APP_VERSION', '1.0.0')
        except ImportError:
            self._version = '1.0.0'
    
    def _get_redis(self):
        """Get Redis client, trying global if not set."""
        if self._redis is not None:
            return self._redis
        
        try:
            from ..core.cache import get_redis
            return get_redis()
        except Exception:
            return None
    
    def _make_key(
        self,
        operation: str,
        content_hash: str,
        config: CacheConfig
    ) -> str:
        """Create cache key."""
        parts = [config.prefix, operation, content_hash]
        if config.include_version:
            parts.append(f"v{self._version}")
        return ":".join(parts)
    
    def _hash_content(self, *args, **kwargs) -> str:
        """Create content hash from arguments."""
        # Combine all arguments into a hashable string
        content = json.dumps(
            {"args": args, "kwargs": kwargs},
            sort_keys=True,
            default=str
        )
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def get(
        self,
        operation: str,
        *args,
        **kwargs
    ) -> Optional[Any]:
        """
        Get cached result for an operation.
        
        Args:
            operation: Operation type (e.g., "simplify", "translate")
            *args, **kwargs: Arguments used to generate cache key
            
        Returns:
            Cached result or None
        """
        config = CACHE_CONFIGS.get(operation, CacheConfig())
        content_hash = self._hash_content(*args, **kwargs)
        key = self._make_key(operation, content_hash, config)
        
        # Try Redis first
        redis = self._get_redis()
        if redis:
            try:
                data = redis.get(key)
                if data:
                    self._stats["hits"] += 1
                    logger.debug(f"Cache hit: {key}")
                    return json.loads(data)
            except Exception as e:
                logger.warning(f"Redis get error: {e}")
                self._stats["errors"] += 1
        
        # Fall back to in-memory
        result = self._fallback.get(key)
        if result is not None:
            self._stats["hits"] += 1
            logger.debug(f"Fallback cache hit: {key}")
            return result
        
        self._stats["misses"] += 1
        return None
    
    def set(
        self,
        operation: str,
        result: Any,
        *args,
        ttl: Optional[int] = None,
        **kwargs
    ):
        """
        Cache result for an operation.
        
        Args:
            operation: Operation type
            result: Result to cache
            *args, **kwargs: Arguments used to generate cache key
            ttl: Optional TTL override in seconds
        """
        config = CACHE_CONFIGS.get(operation, CacheConfig())
        content_hash = self._hash_content(*args, **kwargs)
        key = self._make_key(operation, content_hash, config)
        ttl = ttl or config.ttl_seconds
        
        # Serialize result
        try:
            data = json.dumps(result, default=str)
        except Exception as e:
            logger.warning(f"Failed to serialize cache data: {e}")
            return
        
        # Try Redis first
        redis = self._get_redis()
        if redis:
            try:
                redis.setex(key, ttl, data)
                self._stats["sets"] += 1
                logger.debug(f"Cached to Redis: {key} (TTL: {ttl}s)")
                return
            except Exception as e:
                logger.warning(f"Redis set error: {e}")
                self._stats["errors"] += 1
        
        # Fall back to in-memory
        self._fallback.set(key, result, ttl)
        self._stats["sets"] += 1
        logger.debug(f"Cached to memory: {key} (TTL: {ttl}s)")
    
    def invalidate(self, operation: str, *args, **kwargs):
        """
        Invalidate cache for specific operation and arguments.
        
        Args:
            operation: Operation type
            *args, **kwargs: Arguments used to generate cache key
        """
        config = CACHE_CONFIGS.get(operation, CacheConfig())
        content_hash = self._hash_content(*args, **kwargs)
        key = self._make_key(operation, content_hash, config)
        
        # Delete from Redis
        redis = self._get_redis()
        if redis:
            try:
                redis.delete(key)
            except Exception as e:
                logger.warning(f"Redis delete error: {e}")
        
        # Delete from fallback
        self._fallback.delete(key)
        logger.debug(f"Invalidated cache: {key}")
    
    def invalidate_operation(self, operation: str):
        """
        Invalidate all cache for an operation type.
        
        Args:
            operation: Operation type to invalidate
        """
        config = CACHE_CONFIGS.get(operation, CacheConfig())
        pattern = f"{config.prefix}:{operation}:*"
        
        redis = self._get_redis()
        if redis:
            try:
                keys = list(redis.scan_iter(pattern))
                if keys:
                    redis.delete(*keys)
                    logger.info(f"Invalidated {len(keys)} cache keys for {operation}")
            except Exception as e:
                logger.warning(f"Redis invalidate error: {e}")
        
        # Clear fallback (full clear since we can't pattern match)
        self._fallback.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total = self._stats["hits"] + self._stats["misses"]
        hit_rate = self._stats["hits"] / total if total > 0 else 0
        
        return {
            **self._stats,
            "total_requests": total,
            "hit_rate": round(hit_rate * 100, 2),
            "redis_available": self._get_redis() is not None,
        }


# Global cache instance
_cache: Optional[RequestCache] = None


def get_request_cache() -> RequestCache:
    """Get or create global request cache."""
    global _cache
    if _cache is None:
        _cache = RequestCache()
    return _cache


def cached(
    operation: str,
    ttl: Optional[int] = None,
    key_params: Optional[list] = None
):
    """
    Decorator to cache function results.
    
    Args:
        operation: Cache operation type (e.g., "simplify", "translate")
        ttl: Optional TTL override
        key_params: List of parameter names to include in cache key
                   (if None, uses all parameters)
    
    Usage:
        @cached("simplify", ttl=3600)
        async def simplify_text(text: str, grade_level: int):
            # Expensive operation
            return simplified_text
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            cache = get_request_cache()
            
            # Build cache key from parameters
            if key_params:
                cache_kwargs = {k: kwargs.get(k) for k in key_params if k in kwargs}
            else:
                cache_kwargs = kwargs
            
            # Check cache
            cached_result = cache.get(operation, *args, **cache_kwargs)
            if cached_result is not None:
                return cached_result
            
            # Execute function
            result = await func(*args, **kwargs)
            
            # Cache result
            if result is not None:
                cache.set(operation, result, *args, ttl=ttl, **cache_kwargs)
            
            return result
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            cache = get_request_cache()
            
            # Build cache key from parameters
            if key_params:
                cache_kwargs = {k: kwargs.get(k) for k in key_params if k in kwargs}
            else:
                cache_kwargs = kwargs
            
            # Check cache
            cached_result = cache.get(operation, *args, **cache_kwargs)
            if cached_result is not None:
                return cached_result
            
            # Execute function
            result = func(*args, **kwargs)
            
            # Cache result
            if result is not None:
                cache.set(operation, result, *args, ttl=ttl, **cache_kwargs)
            
            return result
        
        # Return appropriate wrapper based on function type
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator
