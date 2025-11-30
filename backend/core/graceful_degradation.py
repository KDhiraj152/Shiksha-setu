"""
Graceful Degradation (Principles U, V, W)
==========================================
Timeout handling, circuit breaker, and queue shedding for fault tolerance.

Principles:
- U: Timeout + Fallback (e.g., return cached or partial result)
- V: Circuit Breaker for services (open after 3 fails in 1 min)
- W: Queue-length shedding (reject new tasks if queue > 100)

Strategy:
- Configurable timeouts with fallback logic
- Circuit breaker pattern for external services
- Load shedding when queues are overloaded
- Graceful degradation to maintain service availability
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, TypeVar, Generic
from enum import Enum
from functools import wraps
from collections import deque

logger = logging.getLogger(__name__)

T = TypeVar('T')


# =============================================================================
# Principle U: Timeout & Fallback
# =============================================================================

@dataclass
class TimeoutConfig:
    """Configuration for timeout handling."""
    default_timeout_seconds: float = 30.0
    
    # Per-task timeouts
    task_timeouts: Dict[str, float] = field(default_factory=lambda: {
        "simplify": 60.0,
        "translate": 30.0,
        "ocr": 120.0,
        "embed": 15.0,
        "rag": 45.0,
    })
    
    # Enable fallback on timeout
    enable_fallback: bool = True
    
    # Return partial results on timeout
    return_partial: bool = True


class TimeoutError(Exception):
    """Raised when operation times out."""
    pass


class FallbackResult:
    """Wrapper for fallback results."""
    
    def __init__(self, value: Any, is_fallback: bool = False, reason: str = ""):
        self.value = value
        self.is_fallback = is_fallback
        self.reason = reason


async def with_timeout(
    coro: Any,
    timeout_seconds: float,
    fallback_value: Any = None,
    fallback_fn: Optional[Callable] = None,
    task_name: str = "unknown"
) -> FallbackResult:
    """
    Execute coroutine with timeout and fallback (Principle U).
    
    Args:
        coro: Coroutine to execute
        timeout_seconds: Timeout in seconds
        fallback_value: Value to return on timeout (if no fallback_fn)
        fallback_fn: Function to call for fallback value
        task_name: Name of task for logging
        
    Returns:
        FallbackResult with value and fallback indicator
    """
    try:
        result = await asyncio.wait_for(coro, timeout=timeout_seconds)
        return FallbackResult(result, is_fallback=False)
        
    except asyncio.TimeoutError:
        logger.warning(f"Task {task_name} timed out after {timeout_seconds}s")
        
        if fallback_fn:
            try:
                if asyncio.iscoroutinefunction(fallback_fn):
                    fallback = await fallback_fn()
                else:
                    fallback = fallback_fn()
                return FallbackResult(fallback, is_fallback=True, reason="timeout")
            except Exception as e:
                logger.error(f"Fallback function failed: {e}")
        
        return FallbackResult(fallback_value, is_fallback=True, reason="timeout")


def timeout_with_fallback(
    timeout_seconds: Optional[float] = None,
    fallback_value: Any = None,
    fallback_fn: Optional[Callable] = None
):
    """
    Decorator for timeout with fallback (Principle U).
    
    Usage:
        @timeout_with_fallback(timeout_seconds=30, fallback_value="Error occurred")
        async def slow_operation():
            ...
    """
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            nonlocal timeout_seconds
            
            # Get timeout from config if not specified
            if timeout_seconds is None:
                config = TimeoutConfig()
                timeout_seconds = config.default_timeout_seconds
            
            coro = func(*args, **kwargs)
            result = await with_timeout(
                coro,
                timeout_seconds,
                fallback_value,
                fallback_fn,
                func.__name__
            )
            
            return result.value
        
        return wrapper
    return decorator


# =============================================================================
# Principle V: Circuit Breaker
# =============================================================================

class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker (Principle V)."""
    # Failure threshold to open circuit
    failure_threshold: int = 3  # Principle V: open after 3 fails
    
    # Time window for counting failures
    failure_window_seconds: float = 60.0  # Principle V: in 1 min
    
    # Time to wait before trying again
    recovery_timeout_seconds: float = 30.0
    
    # Success threshold to close circuit
    success_threshold: int = 2
    
    # Half-open max requests
    half_open_max_requests: int = 1


class CircuitBreaker:
    """
    Circuit breaker for fault tolerance (Principle V).
    
    States:
    - CLOSED: Normal operation, requests go through
    - OPEN: Service is failing, requests are rejected
    - HALF_OPEN: Testing if service has recovered
    
    Opens after 3 failures in 1 minute (configurable).
    """
    
    def __init__(self, name: str, config: Optional[CircuitBreakerConfig] = None):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        
        self._state = CircuitState.CLOSED
        self._failures: deque = deque()
        self._last_failure_time: Optional[float] = None
        self._half_open_successes = 0
        self._half_open_requests = 0
    
    @property
    def state(self) -> CircuitState:
        """Get current circuit state."""
        self._check_state_transition()
        return self._state
    
    def _check_state_transition(self):
        """Check if state should transition."""
        now = time.time()
        
        if self._state == CircuitState.OPEN:
            # Check if recovery timeout has passed
            if self._last_failure_time:
                elapsed = now - self._last_failure_time
                if elapsed >= self.config.recovery_timeout_seconds:
                    self._transition_to_half_open()
        
        elif self._state == CircuitState.CLOSED:
            # Clean old failures outside window
            window_start = now - self.config.failure_window_seconds
            while self._failures and self._failures[0] < window_start:
                self._failures.popleft()
    
    def _transition_to_half_open(self):
        """Transition to half-open state."""
        logger.info(f"Circuit {self.name}: OPEN -> HALF_OPEN")
        self._state = CircuitState.HALF_OPEN
        self._half_open_successes = 0
        self._half_open_requests = 0
    
    def _transition_to_open(self):
        """Transition to open state."""
        logger.warning(f"Circuit {self.name}: -> OPEN (failures: {len(self._failures)})")
        self._state = CircuitState.OPEN
        self._last_failure_time = time.time()
    
    def _transition_to_closed(self):
        """Transition to closed state."""
        logger.info(f"Circuit {self.name}: -> CLOSED")
        self._state = CircuitState.CLOSED
        self._failures.clear()
        self._half_open_successes = 0
        self._half_open_requests = 0
    
    def can_execute(self) -> bool:
        """Check if request can be executed."""
        state = self.state
        
        if state == CircuitState.CLOSED:
            return True
        
        if state == CircuitState.OPEN:
            return False
        
        # HALF_OPEN - allow limited requests
        if self._half_open_requests < self.config.half_open_max_requests:
            return True
        
        return False
    
    def record_success(self):
        """Record a successful request."""
        if self._state == CircuitState.HALF_OPEN:
            self._half_open_successes += 1
            self._half_open_requests += 1
            
            if self._half_open_successes >= self.config.success_threshold:
                self._transition_to_closed()
    
    def record_failure(self):
        """Record a failed request."""
        now = time.time()
        
        if self._state == CircuitState.HALF_OPEN:
            # Failure in half-open, back to open
            self._transition_to_open()
            return
        
        # Add failure
        self._failures.append(now)
        self._last_failure_time = now
        
        # Check if should open
        self._check_state_transition()
        
        if len(self._failures) >= self.config.failure_threshold:
            self._transition_to_open()
    
    def get_status(self) -> Dict[str, Any]:
        """Get circuit breaker status."""
        return {
            "name": self.name,
            "state": self.state.value,
            "failures_in_window": len(self._failures),
            "failure_threshold": self.config.failure_threshold,
            "last_failure": self._last_failure_time,
        }


class CircuitBreakerRegistry:
    """Registry for circuit breakers."""
    
    def __init__(self):
        self._breakers: Dict[str, CircuitBreaker] = {}
    
    def get(self, name: str, config: Optional[CircuitBreakerConfig] = None) -> CircuitBreaker:
        """Get or create circuit breaker."""
        if name not in self._breakers:
            self._breakers[name] = CircuitBreaker(name, config)
        return self._breakers[name]
    
    def get_all_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all circuit breakers."""
        return {name: cb.get_status() for name, cb in self._breakers.items()}


# Global registry
circuit_registry = CircuitBreakerRegistry()


def circuit_breaker(name: str, config: Optional[CircuitBreakerConfig] = None):
    """
    Decorator for circuit breaker pattern (Principle V).
    
    Usage:
        @circuit_breaker("external_api")
        async def call_external_api():
            ...
    """
    def decorator(func: Callable):
        cb = circuit_registry.get(name, config)
        
        @wraps(func)
        async def wrapper(*args, **kwargs):
            if not cb.can_execute():
                raise CircuitOpenError(f"Circuit {name} is open")
            
            try:
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)
                
                cb.record_success()
                return result
                
            except Exception as e:
                cb.record_failure()
                raise
        
        return wrapper
    return decorator


class CircuitOpenError(Exception):
    """Raised when circuit is open."""
    pass


# =============================================================================
# Principle W: Queue Length Shedding
# =============================================================================

@dataclass
class LoadSheddingConfig:
    """Configuration for load shedding (Principle W)."""
    # Max queue length before shedding
    max_queue_length: int = 100  # Principle W: reject if queue > 100
    
    # Per-queue limits
    queue_limits: Dict[str, int] = field(default_factory=lambda: {
        "simplify": 100,
        "translate": 150,
        "ocr": 50,  # OCR is slower
        "embedding": 200,
        "rag": 100,
    })
    
    # Shedding strategy
    strategy: str = "reject_new"  # or "drop_oldest"
    
    # Priority shedding (shed low priority first)
    enable_priority_shedding: bool = True


class LoadSheddingError(Exception):
    """Raised when request is shed due to overload."""
    pass


class LoadShedder:
    """
    Load shedding for queue management (Principle W).
    
    Monitors queue lengths and rejects/drops requests when overloaded.
    """
    
    def __init__(self, config: Optional[LoadSheddingConfig] = None):
        self.config = config or LoadSheddingConfig()
        self._queue_lengths: Dict[str, int] = {}
        self._shed_count: Dict[str, int] = {}
    
    def check_queue(self, queue_name: str, current_length: int) -> bool:
        """
        Check if queue can accept new requests.
        
        Args:
            queue_name: Name of queue
            current_length: Current queue length
            
        Returns:
            True if request can be accepted
        """
        limit = self.config.queue_limits.get(queue_name, self.config.max_queue_length)
        self._queue_lengths[queue_name] = current_length
        
        if current_length >= limit:
            self._shed_count[queue_name] = self._shed_count.get(queue_name, 0) + 1
            logger.warning(
                f"Load shedding: queue {queue_name} at {current_length}/{limit}"
            )
            return False
        
        return True
    
    def should_accept(self, queue_name: str, priority: int = 5) -> bool:
        """
        Check if request should be accepted based on current load.
        
        Args:
            queue_name: Target queue
            priority: Request priority (1=high, 10=low)
            
        Returns:
            True if request should be accepted
        """
        current = self._queue_lengths.get(queue_name, 0)
        limit = self.config.queue_limits.get(queue_name, self.config.max_queue_length)
        
        if current >= limit:
            return False
        
        # Priority-based shedding
        if self.config.enable_priority_shedding:
            # Start shedding low priority at 80% capacity
            threshold = int(limit * 0.8)
            if current >= threshold:
                # Only accept high priority (1-3)
                if priority > 3:
                    return False
        
        return True
    
    def update_queue_length(self, queue_name: str, length: int):
        """Update tracked queue length."""
        self._queue_lengths[queue_name] = length
    
    def get_status(self) -> Dict[str, Any]:
        """Get load shedding status."""
        return {
            "queue_lengths": self._queue_lengths.copy(),
            "queue_limits": self.config.queue_limits.copy(),
            "shed_counts": self._shed_count.copy(),
            "total_shed": sum(self._shed_count.values())
        }


# Global load shedder
_load_shedder: Optional[LoadShedder] = None


def get_load_shedder() -> LoadShedder:
    """Get or create global load shedder."""
    global _load_shedder
    if _load_shedder is None:
        _load_shedder = LoadShedder()
    return _load_shedder


async def check_queue_and_accept(queue_name: str, priority: int = 5) -> bool:
    """
    Check if request can be accepted (Principle W).
    
    Args:
        queue_name: Target queue
        priority: Request priority
        
    Returns:
        True if request should be accepted
        
    Raises:
        LoadSheddingError if request should be rejected
    """
    shedder = get_load_shedder()
    
    # Get current queue length from Redis/Celery
    try:
        from backend.tasks.celery_config import celery_app
        
        # Get queue length from Celery
        inspect = celery_app.control.inspect()
        queues = inspect.active_queues() or {}
        
        # Count tasks in queue
        reserved = inspect.reserved() or {}
        active = inspect.active() or {}
        
        total = 0
        for worker_tasks in list(reserved.values()) + list(active.values()):
            total += len([t for t in worker_tasks if t.get("delivery_info", {}).get("routing_key") == queue_name])
        
        shedder.update_queue_length(queue_name, total)
        
    except Exception as e:
        logger.debug(f"Could not get queue length: {e}")
    
    if not shedder.should_accept(queue_name, priority):
        raise LoadSheddingError(
            f"Queue {queue_name} is overloaded, request rejected"
        )
    
    return True


def load_shedding_guard(queue_name: str, priority: int = 5):
    """
    Decorator for load shedding (Principle W).
    
    Usage:
        @load_shedding_guard("simplify", priority=3)
        async def simplify_text(text):
            ...
    """
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            await check_queue_and_accept(queue_name, priority)
            
            if asyncio.iscoroutinefunction(func):
                return await func(*args, **kwargs)
            return func(*args, **kwargs)
        
        return wrapper
    return decorator


# =============================================================================
# Combined Graceful Degradation
# =============================================================================

def graceful_degradation(
    timeout_seconds: float = 30.0,
    circuit_name: Optional[str] = None,
    queue_name: Optional[str] = None,
    fallback_value: Any = None,
    priority: int = 5
):
    """
    Combined decorator for graceful degradation (Principles U, V, W).
    
    Applies:
    - Timeout with fallback (U)
    - Circuit breaker (V)
    - Load shedding (W)
    
    Usage:
        @graceful_degradation(
            timeout_seconds=30,
            circuit_name="simplify_service",
            queue_name="simplify",
            fallback_value={"error": "Service unavailable"}
        )
        async def simplify_text(text):
            ...
    """
    def decorator(func: Callable):
        # Get circuit breaker if configured
        cb = circuit_registry.get(circuit_name) if circuit_name else None
        
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Check load shedding (W)
            if queue_name:
                try:
                    await check_queue_and_accept(queue_name, priority)
                except LoadSheddingError:
                    logger.warning(f"Request shed for {func.__name__}")
                    return FallbackResult(
                        fallback_value,
                        is_fallback=True,
                        reason="load_shedding"
                    )
            
            # Check circuit breaker (V)
            if cb and not cb.can_execute():
                logger.warning(f"Circuit open for {func.__name__}")
                return FallbackResult(
                    fallback_value,
                    is_fallback=True,
                    reason="circuit_open"
                )
            
            # Execute with timeout (U)
            try:
                coro = func(*args, **kwargs)
                result = await with_timeout(
                    coro,
                    timeout_seconds,
                    fallback_value,
                    task_name=func.__name__
                )
                
                if cb and not result.is_fallback:
                    cb.record_success()
                
                return result
                
            except Exception as e:
                if cb:
                    cb.record_failure()
                
                logger.error(f"Error in {func.__name__}: {e}")
                return FallbackResult(
                    fallback_value,
                    is_fallback=True,
                    reason=f"error: {type(e).__name__}"
                )
        
        return wrapper
    return decorator
