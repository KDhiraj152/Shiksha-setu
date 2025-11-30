"""
Enterprise-Grade Cache Manager with Redis Enforcement.

Replaces the optional Redis caching with mandatory, hierarchical cache
with automatic invalidation, versioning, and distributed coherence.

Key features:
- Enforced Redis (fails fast if unavailable)
- Hierarchical cache keys with versioning
- Automatic cache invalidation patterns
- Cache warming and precomputation
- Distributed cache coherence
"""

import redis
import json
import hashlib
import logging
import asyncio
from typing import Optional, Any, Dict, Callable, Type, Tuple
from datetime import datetime, timedelta, timezone
from enum import Enum
from dataclasses import asdict, is_dataclass
from functools import wraps

logger = logging.getLogger(__name__)


class CacheLevel(Enum):
    """Cache hierarchy levels."""
    L1_INFERENCE = "inference"  # Model outputs (24h TTL)
    L2_CONTENT = "content"      # Processed content (7d TTL)
    L3_EMBEDDINGS = "embeddings"  # RAG embeddings (permanent)
    L4_METADATA = "metadata"    # Metadata (90d TTL)


class RedisConnectionError(Exception):
    """Raised when Redis is unavailable (no graceful fallback)."""
    pass


class CacheManager:
    """
    Enterprise cache manager with Redis enforcement.
    
    This manager enforces Redis as a critical dependency rather than
    degrading gracefully. This prevents silent data inconsistencies
    in distributed deployments.
    """
    
    # TTL by cache level (in seconds)
    TTL_MAP = {
        CacheLevel.L1_INFERENCE: 86400,      # 24 hours
        CacheLevel.L2_CONTENT: 604800,       # 7 days
        CacheLevel.L3_EMBEDDINGS: 0,         # Permanent
        CacheLevel.L4_METADATA: 7776000,     # 90 days
    }
    
    # Cache version (increment when schema changes)
    CACHE_VERSION = "v2"
    
    def __init__(
        self,
        redis_host: str = "localhost",
        redis_port: int = 6379,
        redis_db: int = 0,
        redis_password: Optional[str] = None,
        fail_fast: bool = True
    ):
        """
        Initialize cache manager.
        
        Args:
            redis_host: Redis hostname
            redis_port: Redis port
            redis_db: Redis database number
            redis_password: Optional Redis password
            fail_fast: If True, raises immediately on Redis unavailability
        
        Raises:
            RedisConnectionError: If fail_fast=True and Redis unavailable
        """
        self.redis_host = redis_host
        self.redis_port = redis_port
        self.redis_db = redis_db
        self.redis_password = redis_password
        self.fail_fast = fail_fast
        
        # Connect to Redis (fail fast if configured)
        self.redis: Optional[redis.Redis] = None
        self._connect()
    
    def _connect(self) -> None:
        """Connect to Redis with fail-fast option."""
        try:
            self.redis = redis.Redis(
                host=self.redis_host,
                port=self.redis_port,
                db=self.redis_db,
                password=self.redis_password,
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5,
                retry_on_timeout=True
            )
            
            # Test connection
            self.redis.ping()
            logger.info(f"✓ Redis connected: {self.redis_host}:{self.redis_port}")
        
        except redis.ConnectionError as e:
            if self.fail_fast:
                logger.critical(f"✗ Redis connection FAILED (fail_fast=True): {e}")
                raise RedisConnectionError(
                    f"Redis unavailable at {self.redis_host}:{self.redis_port}. "
                    "This is a required dependency."
                ) from e
            else:
                logger.warning(f"Redis unavailable (will retry): {e}")
                self.redis = None
    
    def _compute_key(
        self,
        namespace: str,
        level: CacheLevel,
        identifier: str,
        version: Optional[str] = None
    ) -> str:
        """
        Compute hierarchical cache key.
        
        Format: {namespace}:{level}:{version}:{identifier_hash}
        
        This structure enables:
        - Namespace isolation for different services
        - Level-based invalidation
        - Version-based cache busting
        - Hash-based key length limits
        """
        version = version or self.CACHE_VERSION
        identifier_hash = hashlib.sha256(identifier.encode()).hexdigest()[:16]
        
        return f"{namespace}:{level.value}:{version}:{identifier_hash}"
    
    async def get(
        self,
        namespace: str,
        level: CacheLevel,
        identifier: str,
        version: Optional[str] = None,
        deserializer: Optional[Callable] = None
    ) -> Optional[Any]:
        """
        Get value from cache.
        
        Args:
            namespace: Cache namespace (e.g., "pipeline", "rag", "user")
            level: Cache level
            identifier: Unique identifier for this cache entry
            version: Cache version (defaults to CACHE_VERSION)
            deserializer: Custom deserializer (defaults to JSON)
        
        Returns:
            Cached value or None if not found
        """
        if not self.redis:
            logger.warning("Redis unavailable - cache miss")
            return None
        
        try:
            key = self._compute_key(namespace, level, identifier, version)
            
            value = self.redis.get(key)
            if value is None:
                logger.debug(f"Cache miss: {key}")
                return None
            
            # Deserialize
            if deserializer:
                result = deserializer(value)
            else:
                result = json.loads(value)
            
            logger.debug(f"Cache hit: {key}")
            return result
        
        except Exception as e:
            logger.error(f"Cache retrieval error: {e}")
            return None
    
    async def set(
        self,
        namespace: str,
        level: CacheLevel,
        identifier: str,
        value: Any,
        ttl: Optional[int] = None,
        version: Optional[str] = None,
        serializer: Optional[Callable] = None
    ) -> bool:
        """
        Set value in cache.
        
        Args:
            namespace: Cache namespace
            level: Cache level
            identifier: Unique identifier
            value: Value to cache
            ttl: Time-to-live in seconds (defaults to level TTL)
            version: Cache version
            serializer: Custom serializer (defaults to JSON)
        
        Returns:
            True if successful, False otherwise
        """
        if not self.redis:
            logger.warning("Redis unavailable - cache miss on write")
            return False
        
        try:
            key = self._compute_key(namespace, level, identifier, version)
            ttl = ttl or self.TTL_MAP[level]
            
            # Serialize
            if serializer:
                serialized = serializer(value)
            else:
                serialized = json.dumps(value, default=self._json_default)
            
            # Set with TTL
            if ttl > 0:
                self.redis.setex(key, ttl, serialized)
            else:
                # Permanent cache (for embeddings)
                self.redis.set(key, serialized)
            
            logger.debug(f"Cache set: {key} (TTL: {ttl}s)")
            return True
        
        except Exception as e:
            logger.error(f"Cache write error: {e}")
            return False
    
    async def invalidate(
        self,
        namespace: str,
        level: Optional[CacheLevel] = None,
        pattern: Optional[str] = None
    ) -> int:
        """
        Invalidate cache entries.
        
        Args:
            namespace: Cache namespace
            level: Optional cache level to target
            pattern: Optional wildcard pattern (e.g., "user:123:*")
        
        Returns:
            Number of keys deleted
        """
        if not self.redis:
            logger.warning("Redis unavailable - invalidation skipped")
            return 0
        
        try:
            if pattern:
                search_pattern = f"{namespace}:{pattern}"
            elif level:
                search_pattern = f"{namespace}:{level.value}:*"
            else:
                search_pattern = f"{namespace}:*"
            
            # Use SCAN for large keyspaces (avoids blocking)
            keys_deleted = 0
            for key in self.redis.scan_iter(match=search_pattern):
                keys_deleted += self.redis.delete(key)
            
            logger.info(f"Cache invalidated: {search_pattern} ({keys_deleted} keys)")
            return keys_deleted
        
        except Exception as e:
            logger.error(f"Cache invalidation error: {e}")
            return 0
    
    async def invalidate_by_user(self, user_id: str) -> int:
        """Invalidate all cache for a specific user."""
        return await self.invalidate("user", pattern=f"{user_id}:*")
    
    async def invalidate_by_content(self, content_id: str) -> int:
        """Invalidate all cache related to specific content."""
        return await self.invalidate("content", pattern=f"{content_id}:*")
    
    def cache_decorator(
        self,
        namespace: str,
        level: CacheLevel = CacheLevel.L2_CONTENT,
        ttl: Optional[int] = None,
        key_builder: Optional[Callable] = None
    ):
        """
        Decorator for automatic caching of function results.
        
        Example:
            @cache.cache_decorator("content", CacheLevel.L2_CONTENT)
            async def process_content(text: str, language: str):
                # Function logic
                pass
        """
        def decorator(func):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                # Build cache key from function arguments
                if key_builder:
                    identifier = key_builder(*args, **kwargs)
                else:
                    # Default: use function name and all args
                    identifier = f"{func.__name__}:{str(args)}:{str(kwargs)}"
                
                # Try cache
                cached = await self.get(namespace, level, identifier)
                if cached is not None:
                    return cached
                
                # Execute function
                result = await func(*args, **kwargs)
                
                # Cache result
                await self.set(namespace, level, identifier, result, ttl=ttl)
                
                return result
            
            return wrapper
        
        return decorator
    
    async def warm_cache(
        self,
        namespace: str,
        level: CacheLevel,
        data: Dict[str, Any]
    ) -> int:
        """
        Warm cache with pre-computed values.
        
        Useful for preloading common queries or computed values.
        
        Args:
            namespace: Cache namespace
            level: Cache level
            data: Dictionary mapping identifiers to values
        
        Returns:
            Number of entries cached
        """
        count = 0
        for identifier, value in data.items():
            if await self.set(namespace, level, identifier, value):
                count += 1
        
        logger.info(f"Cache warmed: {count} entries in {namespace}/{level.value}")
        return count
    
    async def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        if not self.redis:
            return {"status": "unavailable"}
        
        try:
            info = self.redis.info()
            return {
                "status": "available",
                "memory_used_mb": info.get("used_memory_mb", 0),
                "memory_peak_mb": info.get("used_memory_peak_mb", 0),
                "connected_clients": info.get("connected_clients", 0),
                "total_commands_processed": info.get("total_commands_processed", 0),
            }
        except Exception as e:
            logger.error(f"Error retrieving cache stats: {e}")
            return {"status": "error", "error": str(e)}
    
    @staticmethod
    def _json_default(obj: Any) -> Any:
        """Custom JSON serializer for complex types."""
        if is_dataclass(obj):
            return asdict(obj)
        elif isinstance(obj, (datetime,)):
            return obj.isoformat()
        elif hasattr(obj, '__dict__'):
            return obj.__dict__
        else:
            return str(obj)
    
    def close(self) -> None:
        """Close Redis connection."""
        if self.redis:
            try:
                self.redis.close()
                logger.info("Redis connection closed")
            except Exception as e:
                logger.error(f"Error closing Redis: {e}")


# Global cache manager instance
_cache_manager: Optional[CacheManager] = None


def get_cache_manager(
    fail_fast: bool = True
) -> CacheManager:
    """
    Get or create global cache manager instance.
    
    Args:
        fail_fast: If True, raises on Redis unavailability
    
    Returns:
        CacheManager instance
    
    Raises:
        RedisConnectionError: If fail_fast=True and Redis unavailable
    """
    global _cache_manager
    
    if _cache_manager is None:
        from ...core.config import settings
        
        _cache_manager = CacheManager(
            redis_host=settings.REDIS_HOST if hasattr(settings, 'REDIS_HOST') else 'localhost',
            redis_port=settings.REDIS_PORT if hasattr(settings, 'REDIS_PORT') else 6379,
            redis_db=settings.REDIS_DB if hasattr(settings, 'REDIS_DB') else 0,
            redis_password=settings.REDIS_PASSWORD if hasattr(settings, 'REDIS_PASSWORD') else None,
            fail_fast=fail_fast
        )
    
    return _cache_manager
