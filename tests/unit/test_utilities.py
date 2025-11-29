"""
Unit Tests for Utility Modules

Tests for:
- Lazy model loader
- Request cache
- Circuit breaker
- Rate limiter
"""
import asyncio
import time
import pytest
from unittest.mock import MagicMock, patch, AsyncMock


# Test Lazy Model Loader
class TestLazyModelLoader:
    """Tests for LazyModelLoader."""
    
    @pytest.fixture
    def loader(self):
        """Create test loader."""
        from backend.utils.lazy_loader import LazyModelLoader
        return LazyModelLoader(max_memory_gb=1.0)
    
    def test_initialization(self, loader):
        """Test loader initialization."""
        assert loader.max_memory_gb == 1.0
        assert len(loader._models) == 0
    
    def test_available_memory(self, loader):
        """Test available memory property."""
        assert loader.available_memory_gb == 1.0
    
    def test_loaded_models_empty(self, loader):
        """Test loaded models initially empty."""
        assert loader.loaded_models == {}
    
    def test_estimate_model_size(self, loader):
        """Test model size estimation."""
        # Known model
        size = loader.estimate_model_size("google/flan-t5-base")
        assert size == 1.0
        
        # Pattern-based estimation
        size = loader.estimate_model_size("some-mini-model")
        assert size == 0.2
    
    def test_should_evict(self, loader):
        """Test eviction threshold check."""
        # Initially no eviction needed
        assert loader._should_evict() == False


class TestRequestCache:
    """Tests for RequestCache."""
    
    @pytest.fixture
    def cache(self):
        """Create test cache."""
        from backend.services.request_cache import RequestCache
        c = RequestCache()
        # Clear any existing cached data for test isolation
        c.invalidate("simplify", "test_cache_miss_unique_key_12345", grade_level=99)
        return c
    
    def test_cache_miss(self, cache):
        """Test cache miss returns None."""
        # Use unique key to avoid collision with other tests
        result = cache.get("simplify", "test_cache_miss_unique_key_12345", grade_level=99)
        assert result is None
    
    def test_cache_set_and_get(self, cache):
        """Test setting and getting cache."""
        cache.set("simplify", {"result": "simplified"}, "test_text", grade_level=6)
        result = cache.get("simplify", "test_text", grade_level=6)
        
        assert result == {"result": "simplified"}
    
    def test_cache_different_args(self, cache):
        """Test different args create different keys."""
        cache.set("simplify", {"result": "v1"}, "text1", grade_level=6)
        cache.set("simplify", {"result": "v2"}, "text2", grade_level=6)
        
        assert cache.get("simplify", "text1", grade_level=6) == {"result": "v1"}
        assert cache.get("simplify", "text2", grade_level=6) == {"result": "v2"}
    
    def test_cache_stats(self, cache):
        """Test cache statistics."""
        cache.get("simplify", "miss_text")  # Miss
        cache.set("simplify", {"result": "data"}, "hit_text")
        cache.get("simplify", "hit_text")  # Hit
        
        stats = cache.get_stats()
        
        assert stats["hits"] >= 1
        assert stats["misses"] >= 1
    
    def test_cache_invalidate(self, cache):
        """Test cache invalidation."""
        cache.set("simplify", {"result": "data"}, "test_text")
        cache.invalidate("simplify", "test_text")
        
        result = cache.get("simplify", "test_text")
        assert result is None


class TestCachedDecorator:
    """Tests for @cached decorator."""
    
    @pytest.mark.asyncio
    async def test_cached_async_function(self):
        """Test caching async function."""
        from backend.services.request_cache import cached, get_request_cache
        
        call_count = 0
        
        @cached("test_op")
        async def expensive_operation(text: str):
            nonlocal call_count
            call_count += 1
            return f"result_{text}"
        
        # Clear cache
        cache = get_request_cache()
        cache.invalidate("test_op", "hello")
        call_count = 0
        
        # First call
        result1 = await expensive_operation("hello")
        assert result1 == "result_hello"
        assert call_count == 1
        
        # Second call should be cached
        result2 = await expensive_operation("hello")
        assert result2 == "result_hello"
        assert call_count == 1  # Not incremented
    
    def test_cached_sync_function(self):
        """Test caching sync function."""
        from backend.services.request_cache import cached, get_request_cache
        
        call_count = 0
        
        @cached("test_sync_op")
        def sync_operation(text: str):
            nonlocal call_count
            call_count += 1
            return f"sync_result_{text}"
        
        # Clear cache
        cache = get_request_cache()
        cache.invalidate("test_sync_op", "world")
        call_count = 0
        
        result1 = sync_operation("world")
        result2 = sync_operation("world")
        
        assert result1 == result2 == "sync_result_world"
        assert call_count == 1


class TestCircuitBreaker:
    """Tests for CircuitBreaker."""
    
    @pytest.fixture
    def breaker(self):
        """Create test circuit breaker."""
        from backend.utils.circuit_breaker import CircuitBreaker
        return CircuitBreaker(failure_threshold=3, timeout=1, success_threshold=2)
    
    def test_initial_state_closed(self, breaker):
        """Test initial state is closed."""
        assert breaker.state == "CLOSED"
    
    def test_successful_call(self, breaker):
        """Test successful call passes through."""
        def success_fn():
            return "success"
        
        result = breaker.call(success_fn)
        assert result == "success"
        assert breaker.state == "CLOSED"
    
    def test_opens_after_failures(self, breaker):
        """Test circuit opens after threshold failures."""
        def failing_fn():
            raise Exception("Failure")
        
        for _ in range(3):
            with pytest.raises(Exception):
                breaker.call(failing_fn)
        
        assert breaker.state == "OPEN"
    
    def test_rejects_when_open(self, breaker):
        """Test requests rejected when open."""
        from backend.utils.circuit_breaker import CircuitBreakerOpen
        
        def failing_fn():
            raise Exception("Failure")
        
        # Open the circuit
        for _ in range(3):
            with pytest.raises(Exception):
                breaker.call(failing_fn)
        
        # Should reject
        with pytest.raises(CircuitBreakerOpen):
            breaker.call(lambda: "success")
    
    def test_half_open_after_timeout(self, breaker):
        """Test circuit goes half-open after timeout."""
        from backend.utils.circuit_breaker import CircuitBreakerOpen
        
        def failing_fn():
            raise Exception("Failure")
        
        # Open the circuit
        for _ in range(3):
            with pytest.raises(Exception):
                breaker.call(failing_fn)
        
        assert breaker.state == "OPEN"
        
        # Wait for timeout
        time.sleep(1.1)
        
        # Next call should be allowed (half-open)
        result = breaker.call(lambda: "success")
        assert breaker.state in ["HALF_OPEN", "CLOSED"]
    
    def test_reset(self, breaker):
        """Test manual reset."""
        def failing_fn():
            raise Exception("Failure")
        
        for _ in range(3):
            with pytest.raises(Exception):
                breaker.call(failing_fn)
        
        assert breaker.state == "OPEN"
        
        breaker.reset()
        assert breaker.state == "CLOSED"
    
    @pytest.mark.asyncio
    async def test_async_call(self, breaker):
        """Test async call through breaker."""
        async def async_success():
            return "async_success"
        
        result = await breaker.call_async(async_success)
        assert result == "async_success"


class TestRateLimiter:
    """Tests for RateLimiter."""
    
    @pytest.fixture
    def limiter(self):
        """Create test rate limiter."""
        from backend.middleware.rate_limiter import RateLimiter
        return RateLimiter()
    
    def test_allows_initial_requests(self, limiter):
        """Test initial requests are allowed."""
        allowed, retry_after, headers = limiter.check("user123", "default")
        assert allowed is True
        assert retry_after == 0
    
    def test_rate_limit_headers(self, limiter):
        """Test rate limit headers are returned."""
        _, _, headers = limiter.check("user123", "default")
        
        assert "X-RateLimit-Limit" in headers
        assert "X-RateLimit-Remaining" in headers
        assert "X-RateLimit-Reset" in headers
    
    @pytest.mark.asyncio
    async def test_async_check(self, limiter):
        """Test async rate limit check."""
        allowed, retry_after, headers = await limiter.check_async("user456", "default")
        assert allowed is True


class TestTokenBucket:
    """Tests for TokenBucket algorithm."""
    
    @pytest.fixture
    def bucket(self):
        """Create test bucket."""
        from backend.middleware.rate_limiter import TokenBucket
        return TokenBucket(rate=1.0, capacity=5)  # 1 token per second, max 5
    
    def test_initial_capacity(self, bucket):
        """Test bucket starts at capacity."""
        assert bucket.tokens == 5
    
    def test_consume_tokens(self, bucket):
        """Test consuming tokens."""
        success, _ = bucket.consume(1)
        assert success is True
        assert bucket.tokens == 4
    
    def test_consume_too_many(self, bucket):
        """Test consuming more than available."""
        success, _ = bucket.consume(10)
        assert success is False
    
    def test_refill_over_time(self, bucket):
        """Test tokens refill over time."""
        bucket.tokens = 0
        bucket.last_update = time.time() - 2  # 2 seconds ago
        
        success, _ = bucket.consume(1)
        # Should have refilled ~2 tokens
        assert success is True


class TestInMemoryCache:
    """Tests for InMemoryCache."""
    
    @pytest.fixture
    def cache(self):
        """Create test cache."""
        from backend.services.request_cache import InMemoryCache
        return InMemoryCache(max_size=10)
    
    def test_set_and_get(self, cache):
        """Test basic set and get."""
        cache.set("key1", "value1", ttl=60)
        assert cache.get("key1") == "value1"
    
    def test_expiry(self, cache):
        """Test TTL expiry."""
        cache.set("key1", "value1", ttl=0)  # Immediate expiry
        time.sleep(0.1)
        assert cache.get("key1") is None
    
    def test_eviction_at_capacity(self, cache):
        """Test eviction when at capacity."""
        # Fill cache
        for i in range(15):
            cache.set(f"key{i}", f"value{i}", ttl=60)
        
        # Should have evicted some entries
        assert len(cache._cache) <= 10
    
    def test_delete(self, cache):
        """Test delete."""
        cache.set("key1", "value1", ttl=60)
        cache.delete("key1")
        assert cache.get("key1") is None
    
    def test_clear(self, cache):
        """Test clear all."""
        cache.set("key1", "value1", ttl=60)
        cache.set("key2", "value2", ttl=60)
        cache.clear()
        assert len(cache._cache) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
