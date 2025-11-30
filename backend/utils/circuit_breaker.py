"""Circuit breaker pattern for HuggingFace API calls with tenacity."""
import asyncio
from typing import Optional, Callable, Any
from datetime import datetime, timedelta
import logging

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
    after_log
)
import httpx

logger = logging.getLogger(__name__)


class CircuitBreakerOpen(Exception):
    """Exception raised when circuit breaker is open."""
    pass


class CircuitBreaker:
    """
    Circuit breaker implementation for API calls.
    
    States:
    - CLOSED: Normal operation, requests pass through
    - OPEN: Too many failures, requests fail fast
    - HALF_OPEN: Testing if service recovered
    
    Args:
        failure_threshold: Number of failures before opening circuit
        timeout: Seconds to wait before trying again (OPEN -> HALF_OPEN)
        success_threshold: Successful calls needed to close circuit
    """
    
    def __init__(
        self,
        failure_threshold: int = 5,
        timeout: int = 60,
        success_threshold: int = 2
    ):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.success_threshold = success_threshold
        
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection."""
        if self.state == "OPEN":
            if self._should_attempt_reset():
                self.state = "HALF_OPEN"
                logger.info("Circuit breaker entering HALF_OPEN state")
            else:
                raise CircuitBreakerOpen(
                    f"Circuit breaker is OPEN. "
                    f"Try again after {self.timeout}s from last failure."
                )
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise e
    
    async def call_async(self, func: Callable, *args, **kwargs) -> Any:
        """Execute async function with circuit breaker protection."""
        if self.state == "OPEN":
            if self._should_attempt_reset():
                self.state = "HALF_OPEN"
                logger.info("Circuit breaker entering HALF_OPEN state")
            else:
                raise CircuitBreakerOpen(
                    f"Circuit breaker is OPEN. "
                    f"Try again after {self.timeout}s from last failure."
                )
        
        try:
            result = await func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise e
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset."""
        if self.last_failure_time is None:
            return True
        
        return datetime.now() >= self.last_failure_time + timedelta(seconds=self.timeout)
    
    def _on_success(self):
        """Handle successful call."""
        self.failure_count = 0
        
        if self.state == "HALF_OPEN":
            self.success_count += 1
            if self.success_count >= self.success_threshold:
                self.state = "CLOSED"
                self.success_count = 0
                logger.info("Circuit breaker CLOSED after successful recovery")
    
    def _on_failure(self):
        """Handle failed call."""
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        self.success_count = 0
        
        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"
            logger.warning(
                f"Circuit breaker OPEN after {self.failure_count} failures"
            )
    
    def reset(self):
        """Manually reset circuit breaker."""
        self.state = "CLOSED"
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        logger.info("Circuit breaker manually reset")


# Global circuit breakers for different services
hf_circuit_breaker = CircuitBreaker(failure_threshold=5, timeout=60)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type((httpx.TimeoutException, httpx.NetworkError)),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    after=after_log(logger, logging.INFO)
)
async def call_hf_api_with_retry(
    client: httpx.AsyncClient,
    url: str,
    **kwargs
) -> httpx.Response:
    """
    Call HuggingFace API with retry logic and circuit breaker.
    
    Args:
        client: httpx AsyncClient instance
        url: API endpoint URL
        **kwargs: Additional request parameters
        
    Returns:
        HTTP response
        
    Raises:
        CircuitBreakerOpen: If circuit breaker is open
        httpx.HTTPError: If request fails after retries
    """
    async def make_request():
        response = await client.post(url, **kwargs)
        response.raise_for_status()
        return response
    
    return await hf_circuit_breaker.call_async(make_request)


# Decorator for adding circuit breaker to async functions
def with_circuit_breaker(breaker: CircuitBreaker):
    """
    Decorator to add circuit breaker protection to async functions.
    
    Usage:
        @with_circuit_breaker(hf_circuit_breaker)
        async def my_api_call():
            ...
    """
    def decorator(func: Callable):
        async def wrapper(*args, **kwargs):
            return await breaker.call_async(func, *args, **kwargs)
        return wrapper
    return decorator
