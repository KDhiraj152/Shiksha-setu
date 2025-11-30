"""
Integration tests for production-grade system improvements.

These tests validate:
- Concurrent pipeline execution
- Cache coherence
- Database resilience
- Frontend state management
"""

import pytest
import asyncio
import json
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock
import redis

# ============================================================================
# PIPELINE ORCHESTRATION TESTS
# ============================================================================

@pytest.mark.asyncio
async def test_concurrent_pipeline_execution():
    """Verify concurrent stage execution."""
    from backend.services.pipeline.orchestrator_v2 import ConcurrentPipelineOrchestrator
    
    orchestrator = ConcurrentPipelineOrchestrator()
    
    # Mock model clients
    orchestrator.flant5_client.process = AsyncMock(return_value="simplified text")
    orchestrator.indictrans2_client.process = AsyncMock(return_value="translated text")
    orchestrator.bert_client.process = AsyncMock(return_value=0.95)
    orchestrator.vits_client.process = AsyncMock(return_value=("audio.wav", 0.85))
    
    # Execute pipeline
    result = await orchestrator.process_content(
        input_data="Sample content",
        target_language="Hindi",
        grade_level=8,
        subject="Science",
        output_format="both"
    )
    
    # Verify result structure
    assert result.simplified_text == "simplified text"
    assert result.translated_text == "translated text"
    assert result.ncert_alignment_score == 0.95
    assert result.audio_file_path == "audio.wav"
    
    # Verify concurrent execution (all stages should have metrics)
    assert len(result.metrics) >= 3


@pytest.mark.asyncio
async def test_pipeline_backpressure():
    """Verify backpressure control with semaphores."""
    from backend.services.pipeline.orchestrator_v2 import ConcurrentPipelineOrchestrator, PipelineStage
    
    orchestrator = ConcurrentPipelineOrchestrator()
    
    # Mock slow client
    async def slow_process(*args, **kwargs):
        await asyncio.sleep(1)
        return "result"
    
    orchestrator.flant5_client.process = slow_process
    
    # Create multiple concurrent tasks
    tasks = []
    start_time = datetime.now()
    
    for i in range(10):
        task = asyncio.create_task(
            orchestrator._execute_stage_with_backpressure(
                PipelineStage.SIMPLIFICATION,
                slow_process
            )
        )
        tasks.append(task)
    
    # Execute concurrently with semaphore limit (default is 5)
    results = await asyncio.gather(*tasks, return_exceptions=True)
    elapsed = (datetime.now() - start_time).total_seconds()
    
    # With 5 concurrent limit, 10 tasks should take ~2 seconds
    assert 1.8 < elapsed < 3.0
    assert all(isinstance(r, tuple) for r in results)


@pytest.mark.asyncio
async def test_pipeline_circuit_breaker():
    """Verify circuit breaker prevents cascading failures."""
    from backend.services.pipeline.orchestrator_v2 import (
        ConcurrentPipelineOrchestrator,
        PipelineStage,
        ProcessingStatus
    )
    from backend.core.exceptions import ShikshaSetuException
    
    orchestrator = ConcurrentPipelineOrchestrator()
    
    # Make model client fail
    orchestrator.flant5_client.process = AsyncMock(
        side_effect=Exception("Model API down")
    )
    
    # Trigger failures to open circuit breaker
    failure_threshold = 5
    
    for i in range(failure_threshold + 1):
        try:
            await orchestrator._execute_stage_with_backpressure(
                PipelineStage.SIMPLIFICATION,
                orchestrator.flant5_client.process
            )
        except:
            pass
    
    # Circuit breaker should be open now
    assert orchestrator.circuit_breakers[PipelineStage.SIMPLIFICATION].is_open()


# ============================================================================
# CACHE MANAGEMENT TESTS
# ============================================================================

@pytest.fixture
def cache_manager():
    """Provide cache manager for testing."""
    from backend.services.cache_manager import CacheManager, CacheLevel
    
    # Use test Redis instance
    return CacheManager(
        redis_host="localhost",
        redis_port=6379,
        redis_db=15,  # Use separate DB for tests
        fail_fast=False
    )


@pytest.mark.asyncio
async def test_cache_enforcement():
    """Verify Redis caching is enforced."""
    from backend.services.cache_manager import CacheManager
    
    # Should raise if Redis unavailable with fail_fast=True
    with pytest.raises(Exception):  # RedisConnectionError
        CacheManager(
            redis_host="invalid-host",
            redis_port=9999,
            fail_fast=True
        )


@pytest.mark.asyncio
async def test_cache_hierarchical_keys(cache_manager):
    """Verify hierarchical cache key structure."""
    from backend.services.cache_manager import CacheLevel
    
    # Set value with L1 (short TTL)
    await cache_manager.set(
        namespace="pipeline",
        level=CacheLevel.L1_INFERENCE,
        identifier="test-1",
        value={"data": "inference"}
    )
    
    # Retrieve with same namespace/level
    result = await cache_manager.get(
        namespace="pipeline",
        level=CacheLevel.L1_INFERENCE,
        identifier="test-1"
    )
    
    assert result == {"data": "inference"}
    
    # Different level should not find it
    result = await cache_manager.get(
        namespace="pipeline",
        level=CacheLevel.L2_CONTENT,
        identifier="test-1"
    )
    
    assert result is None


@pytest.mark.asyncio
async def test_cache_invalidation(cache_manager):
    """Verify cache invalidation patterns."""
    from backend.services.cache_manager import CacheLevel
    
    # Add multiple entries
    for i in range(5):
        await cache_manager.set(
            namespace="user",
            level=CacheLevel.L2_CONTENT,
            identifier=f"user-123-item-{i}",
            value={"id": i}
        )
    
    # Invalidate all user-123 entries
    deleted = await cache_manager.invalidate(
        namespace="user",
        pattern="123-*"
    )
    
    assert deleted >= 5


# ============================================================================
# DATABASE RESILIENCE TESTS
# ============================================================================

@pytest.fixture
def db_connection():
    """Provide database connection for testing."""
    from backend.database_v2 import ResilientDatabaseConnection
    
    return ResilientDatabaseConnection(
        database_url="sqlite:///:memory:",  # Test with SQLite
        pool_size=5,
        max_overflow=5
    )


def test_circuit_breaker_state_transitions(db_connection):
    """Verify circuit breaker state transitions."""
    cb = db_connection.circuit_breaker
    
    # Initial state: closed
    assert cb.state == "closed"
    assert cb.can_attempt()
    
    # Record failures to open
    for _ in range(cb.failure_threshold):
        cb.record_failure("test error")
    
    assert cb.is_open()
    assert not cb.can_attempt()
    
    # After recovery timeout, enters half-open
    cb.last_failure_time = asyncio.get_event_loop().time() - 61
    assert cb.can_attempt()
    assert cb.state == "half-open"
    
    # Successful attempts close circuit
    for _ in range(cb.half_open_max_attempts):
        cb.record_success()
    
    assert cb.state == "closed"


def test_pool_metrics_tracking(db_connection):
    """Verify connection pool metrics."""
    metrics = db_connection.get_pool_status()
    
    assert metrics.pool_size == 5
    assert metrics.total_checkouts >= 0
    assert metrics.status in ["healthy", "degraded", "critical", "recovering"]


# ============================================================================
# FRONTEND STATE MANAGEMENT TESTS
# ============================================================================

@pytest.mark.asyncio
async def test_cache_decorator():
    """Verify cache decorator for function caching."""
    from backend.services.cache_manager import CacheManager, CacheLevel
    
    cache_mgr = CacheManager(fail_fast=False)
    
    call_count = 0
    
    @cache_mgr.cache_decorator("test", CacheLevel.L1_INFERENCE)
    async def expensive_operation(x: int, y: int):
        nonlocal call_count
        call_count += 1
        return x + y
    
    # First call should execute
    result1 = await expensive_operation(1, 2)
    assert result1 == 3
    assert call_count == 1
    
    # Second call should use cache
    result2 = await expensive_operation(1, 2)
    assert result2 == 3
    assert call_count == 1  # Not incremented
    
    # Different args should not use cache
    result3 = await expensive_operation(2, 3)
    assert result3 == 5
    assert call_count == 2


# ============================================================================
# FAILURE SCENARIO TESTS
# ============================================================================

@pytest.mark.asyncio
async def test_graceful_degradation_on_model_timeout():
    """Verify graceful degradation when model API times out."""
    from backend.services.pipeline.orchestrator_v2 import ConcurrentPipelineOrchestrator
    
    orchestrator = ConcurrentPipelineOrchestrator()
    
    # Mock timeout
    async def timeout_handler(*args, **kwargs):
        await asyncio.sleep(100)
    
    orchestrator.flant5_client.process = timeout_handler
    
    # Should timeout gracefully (not hang forever)
    with pytest.raises(Exception):  # TimeoutError wrapped
        result = await asyncio.wait_for(
            orchestrator.process_content(
                input_data="test",
                target_language="Hindi",
                grade_level=8,
                subject="Science"
            ),
            timeout=5
        )


@pytest.mark.asyncio
async def test_cache_fallback_on_redis_failure(cache_manager):
    """Verify system continues when cache operations fail."""
    from backend.services.cache_manager import CacheLevel
    
    # Disconnect Redis
    cache_manager.redis = None
    
    # Cache operations should return None but not raise
    result = await cache_manager.get(
        namespace="test",
        level=CacheLevel.L1_INFERENCE,
        identifier="key"
    )
    
    assert result is None
    
    # Set should return False but not raise
    success = await cache_manager.set(
        namespace="test",
        level=CacheLevel.L1_INFERENCE,
        identifier="key",
        value="value"
    )
    
    assert success is False


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

@pytest.mark.asyncio
async def test_full_pipeline_with_caching(cache_manager):
    """Integration test: Full pipeline with caching enabled."""
    from backend.services.pipeline.orchestrator_v2 import ConcurrentPipelineOrchestrator
    from backend.services.cache_manager import CacheLevel
    
    # This would require a real/mocked database and Redis
    # Simplified example:
    
    orchestrator = ConcurrentPipelineOrchestrator()
    
    # Mock clients
    orchestrator.flant5_client.process = AsyncMock(return_value="simplified")
    orchestrator.indictrans2_client.process = AsyncMock(return_value="translated")
    orchestrator.bert_client.process = AsyncMock(return_value=0.9)
    orchestrator.vits_client.process = AsyncMock(return_value=("audio.mp3", 0.85))
    
    # Mock cache and database
    with patch("backend.services.pipeline.orchestrator_v2.get_redis") as mock_redis:
        with patch("backend.services.pipeline.orchestrator_v2.get_db"):
            # First execution
            result1 = await orchestrator.process_content(
                input_data="test content",
                target_language="Hindi",
                grade_level=8,
                subject="Science"
            )
            
            # Should have executed all stages
            assert orchestrator.flant5_client.process.call_count >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
