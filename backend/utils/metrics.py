"""
Prometheus Metrics Module

Provides metrics collection for monitoring:
- HTTP request metrics (latency, count, errors)
- Model inference metrics
- Cache hit/miss rates
- Background task metrics
- System resource usage
"""
import time
import logging
from functools import wraps
from typing import Callable, Optional, Dict, Any
from dataclasses import dataclass, field
from collections import defaultdict
import threading

logger = logging.getLogger(__name__)

# Try to import prometheus_client, fallback to mock if not available
try:
    from prometheus_client import (
        Counter, Histogram, Gauge, Summary, Info,
        CollectorRegistry, generate_latest, CONTENT_TYPE_LATEST,
        multiprocess, REGISTRY
    )
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    logger.warning("prometheus_client not installed, using mock metrics")


# ============ Mock Metrics for Development ============

class MockMetric:
    """Mock metric for when prometheus_client is not available."""
    def __init__(self, *args, **kwargs):
        self._value = 0
        self._values = defaultdict(float)
    
    def inc(self, value=1):
        self._value += value
    
    def dec(self, value=1):
        self._value -= value
    
    def set(self, value):
        self._value = value
    
    def observe(self, value):
        self._value = value
    
    def labels(self, **kwargs):
        key = tuple(sorted(kwargs.items()))
        return MockLabeledMetric(self._values, key)
    
    def time(self):
        return MockTimer(self)
    
    def info(self, value):
        self._value = value


class MockLabeledMetric:
    def __init__(self, values, key):
        self._values = values
        self._key = key
    
    def inc(self, value=1):
        self._values[self._key] += value
    
    def dec(self, value=1):
        self._values[self._key] -= value
    
    def set(self, value):
        self._values[self._key] = value
    
    def observe(self, value):
        self._values[self._key] = value


class MockTimer:
    def __init__(self, metric):
        self._metric = metric
        self._start = None
    
    def __enter__(self):
        self._start = time.time()
        return self
    
    def __exit__(self, *args):
        self._metric.observe(time.time() - self._start)


# ============ Metrics Registry ============

class MetricsRegistry:
    """
    Central registry for all application metrics.
    
    Provides both Prometheus-compatible metrics and a lightweight
    fallback for development environments.
    """
    
    def __init__(self, prefix: str = "shiksha"):
        self.prefix = prefix
        self._metrics: Dict[str, Any] = {}
        self._lock = threading.Lock()
        
        # Initialize core metrics
        self._init_http_metrics()
        self._init_model_metrics()
        self._init_cache_metrics()
        self._init_task_metrics()
        self._init_system_metrics()
    
    def _create_counter(self, name: str, description: str, labels: list = None) -> Any:
        """Create a counter metric."""
        full_name = f"{self.prefix}_{name}"
        if PROMETHEUS_AVAILABLE:
            return Counter(full_name, description, labels or [])
        return MockMetric()
    
    def _create_histogram(self, name: str, description: str, labels: list = None, buckets: tuple = None) -> Any:
        """Create a histogram metric."""
        full_name = f"{self.prefix}_{name}"
        if PROMETHEUS_AVAILABLE:
            kwargs = {"labelnames": labels} if labels else {}
            if buckets:
                kwargs["buckets"] = buckets
            return Histogram(full_name, description, **kwargs)
        return MockMetric()
    
    def _create_gauge(self, name: str, description: str, labels: list = None) -> Any:
        """Create a gauge metric."""
        full_name = f"{self.prefix}_{name}"
        if PROMETHEUS_AVAILABLE:
            return Gauge(full_name, description, labels or [])
        return MockMetric()
    
    def _create_summary(self, name: str, description: str, labels: list = None) -> Any:
        """Create a summary metric."""
        full_name = f"{self.prefix}_{name}"
        if PROMETHEUS_AVAILABLE:
            return Summary(full_name, description, labels or [])
        return MockMetric()
    
    def _init_http_metrics(self):
        """Initialize HTTP request metrics."""
        # Request counter
        self.http_requests_total = self._create_counter(
            "http_requests_total",
            "Total HTTP requests",
            ["method", "endpoint", "status"]
        )
        
        # Request latency
        self.http_request_duration_seconds = self._create_histogram(
            "http_request_duration_seconds",
            "HTTP request latency in seconds",
            ["method", "endpoint"],
            buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)
        )
        
        # Active requests
        self.http_requests_in_progress = self._create_gauge(
            "http_requests_in_progress",
            "Number of HTTP requests in progress",
            ["method", "endpoint"]
        )
        
        # Request size
        self.http_request_size_bytes = self._create_histogram(
            "http_request_size_bytes",
            "HTTP request size in bytes",
            ["method", "endpoint"],
            buckets=(100, 1000, 10000, 100000, 1000000, 10000000)
        )
        
        # Response size
        self.http_response_size_bytes = self._create_histogram(
            "http_response_size_bytes",
            "HTTP response size in bytes",
            ["method", "endpoint"],
            buckets=(100, 1000, 10000, 100000, 1000000, 10000000)
        )
    
    def _init_model_metrics(self):
        """Initialize model inference metrics."""
        # Inference counter
        self.model_inferences_total = self._create_counter(
            "model_inferences_total",
            "Total model inferences",
            ["model_name", "model_tier", "status"]
        )
        
        # Inference latency
        self.model_inference_duration_seconds = self._create_histogram(
            "model_inference_duration_seconds",
            "Model inference latency in seconds",
            ["model_name", "model_tier"],
            buckets=(0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0)
        )
        
        # Model memory usage
        self.model_memory_bytes = self._create_gauge(
            "model_memory_bytes",
            "Model memory usage in bytes",
            ["model_name"]
        )
        
        # Loaded models count
        self.models_loaded = self._create_gauge(
            "models_loaded",
            "Number of currently loaded models"
        )
        
        # Token throughput
        self.model_tokens_processed = self._create_counter(
            "model_tokens_processed_total",
            "Total tokens processed",
            ["model_name", "direction"]  # direction: input/output
        )
    
    def _init_cache_metrics(self):
        """Initialize cache metrics."""
        # Cache operations
        self.cache_operations_total = self._create_counter(
            "cache_operations_total",
            "Total cache operations",
            ["operation", "cache_type", "status"]  # operation: get/set/delete, status: hit/miss
        )
        
        # Cache size
        self.cache_size_bytes = self._create_gauge(
            "cache_size_bytes",
            "Cache size in bytes",
            ["cache_type"]
        )
        
        # Cache items
        self.cache_items = self._create_gauge(
            "cache_items",
            "Number of items in cache",
            ["cache_type"]
        )
        
        # Cache hit rate (calculated from operations)
        self.cache_hit_ratio = self._create_gauge(
            "cache_hit_ratio",
            "Cache hit ratio",
            ["cache_type"]
        )
    
    def _init_task_metrics(self):
        """Initialize background task metrics."""
        # Task counter
        self.tasks_total = self._create_counter(
            "tasks_total",
            "Total background tasks",
            ["task_name", "status"]  # status: started/completed/failed/retried
        )
        
        # Task duration
        self.task_duration_seconds = self._create_histogram(
            "task_duration_seconds",
            "Background task duration in seconds",
            ["task_name"],
            buckets=(1.0, 5.0, 10.0, 30.0, 60.0, 300.0, 600.0)
        )
        
        # Active tasks
        self.tasks_in_progress = self._create_gauge(
            "tasks_in_progress",
            "Number of tasks in progress",
            ["task_name"]
        )
        
        # Task queue size
        self.task_queue_size = self._create_gauge(
            "task_queue_size",
            "Number of tasks in queue",
            ["queue_name"]
        )
    
    def _init_system_metrics(self):
        """Initialize system resource metrics."""
        # CPU usage
        self.cpu_usage_percent = self._create_gauge(
            "cpu_usage_percent",
            "CPU usage percentage"
        )
        
        # Memory usage
        self.memory_usage_bytes = self._create_gauge(
            "memory_usage_bytes",
            "Memory usage in bytes",
            ["type"]  # type: rss/vms/shared
        )
        
        # GPU memory (if available)
        self.gpu_memory_bytes = self._create_gauge(
            "gpu_memory_bytes",
            "GPU memory usage in bytes",
            ["device"]
        )
        
        # Database connections
        self.db_connections = self._create_gauge(
            "db_connections",
            "Number of database connections",
            ["state"]  # state: active/idle
        )
        
        # Redis connections
        self.redis_connections = self._create_gauge(
            "redis_connections",
            "Number of Redis connections"
        )


# Global metrics registry
_metrics: Optional[MetricsRegistry] = None


def get_metrics() -> MetricsRegistry:
    """Get global metrics registry."""
    global _metrics
    if _metrics is None:
        _metrics = MetricsRegistry()
    return _metrics


# ============ Decorators for Easy Instrumentation ============

def track_request_metrics(endpoint: str = None):
    """
    Decorator to track HTTP request metrics.
    
    Usage:
        @track_request_metrics()
        async def my_endpoint():
            ...
    """
    def decorator(func: Callable):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            metrics = get_metrics()
            ep = endpoint or func.__name__
            method = kwargs.get("request", {}).method if hasattr(kwargs.get("request", {}), "method") else "UNKNOWN"
            
            metrics.http_requests_in_progress.labels(method=method, endpoint=ep).inc()
            start_time = time.time()
            status = "success"
            
            try:
                result = await func(*args, **kwargs)
                return result
            except Exception as e:
                status = "error"
                raise
            finally:
                duration = time.time() - start_time
                metrics.http_request_duration_seconds.labels(method=method, endpoint=ep).observe(duration)
                metrics.http_requests_total.labels(method=method, endpoint=ep, status=status).inc()
                metrics.http_requests_in_progress.labels(method=method, endpoint=ep).dec()
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            metrics = get_metrics()
            ep = endpoint or func.__name__
            method = "SYNC"
            
            metrics.http_requests_in_progress.labels(method=method, endpoint=ep).inc()
            start_time = time.time()
            status = "success"
            
            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                status = "error"
                raise
            finally:
                duration = time.time() - start_time
                metrics.http_request_duration_seconds.labels(method=method, endpoint=ep).observe(duration)
                metrics.http_requests_total.labels(method=method, endpoint=ep, status=status).inc()
                metrics.http_requests_in_progress.labels(method=method, endpoint=ep).dec()
        
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    return decorator


def track_model_inference(model_name: str, model_tier: str = "UNKNOWN"):
    """
    Decorator to track model inference metrics.
    
    Usage:
        @track_model_inference("simplifier", "MEDIUM")
        async def simplify(text):
            ...
    """
    def decorator(func: Callable):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            metrics = get_metrics()
            start_time = time.time()
            status = "success"
            
            try:
                result = await func(*args, **kwargs)
                return result
            except Exception as e:
                status = "error"
                raise
            finally:
                duration = time.time() - start_time
                metrics.model_inference_duration_seconds.labels(
                    model_name=model_name, model_tier=model_tier
                ).observe(duration)
                metrics.model_inferences_total.labels(
                    model_name=model_name, model_tier=model_tier, status=status
                ).inc()
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            metrics = get_metrics()
            start_time = time.time()
            status = "success"
            
            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                status = "error"
                raise
            finally:
                duration = time.time() - start_time
                metrics.model_inference_duration_seconds.labels(
                    model_name=model_name, model_tier=model_tier
                ).observe(duration)
                metrics.model_inferences_total.labels(
                    model_name=model_name, model_tier=model_tier, status=status
                ).inc()
        
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    return decorator


def track_cache_operation(cache_type: str = "default"):
    """
    Decorator to track cache operations.
    
    Usage:
        @track_cache_operation("embedding")
        def get_from_cache(key):
            ...
    """
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            metrics = get_metrics()
            result = func(*args, **kwargs)
            
            status = "hit" if result is not None else "miss"
            metrics.cache_operations_total.labels(
                operation="get", cache_type=cache_type, status=status
            ).inc()
            
            return result
        return wrapper
    return decorator


# ============ FastAPI Middleware ============

async def metrics_middleware(request, call_next):
    """
    FastAPI middleware for automatic metrics collection.
    
    Add to app:
        app.middleware("http")(metrics_middleware)
    """
    metrics = get_metrics()
    method = request.method
    endpoint = request.url.path
    
    # Track in-progress requests
    metrics.http_requests_in_progress.labels(method=method, endpoint=endpoint).inc()
    
    start_time = time.time()
    status_code = 500
    
    try:
        response = await call_next(request)
        status_code = response.status_code
        return response
    except Exception as e:
        raise
    finally:
        duration = time.time() - start_time
        status = "success" if status_code < 400 else "error"
        
        metrics.http_request_duration_seconds.labels(
            method=method, endpoint=endpoint
        ).observe(duration)
        
        metrics.http_requests_total.labels(
            method=method, endpoint=endpoint, status=status
        ).inc()
        
        metrics.http_requests_in_progress.labels(
            method=method, endpoint=endpoint
        ).dec()


# ============ Metrics Endpoint ============

def get_metrics_response():
    """Generate Prometheus metrics response."""
    if PROMETHEUS_AVAILABLE:
        return generate_latest(REGISTRY), CONTENT_TYPE_LATEST
    else:
        # Return JSON format for mock metrics
        metrics = get_metrics()
        return {"status": "prometheus_client not installed"}, "application/json"
