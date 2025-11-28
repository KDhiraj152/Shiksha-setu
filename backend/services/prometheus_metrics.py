"""
Prometheus Metrics Integration

Issue: CODE-REVIEW-GPT #17 (MEDIUM)
Purpose: Application performance monitoring with Prometheus
"""

from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry, generate_latest
from typing import Callable
import time
import logging
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)

# Create registry
registry = CollectorRegistry()

# HTTP Metrics
http_requests_total = Counter(
    'http_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status'],
    registry=registry
)

http_request_duration_seconds = Histogram(
    'http_request_duration_seconds',
    'HTTP request duration in seconds',
    ['method', 'endpoint'],
    registry=registry
)

http_requests_in_progress = Gauge(
    'http_requests_in_progress',
    'HTTP requests currently in progress',
    ['method', 'endpoint'],
    registry=registry
)

# Database Metrics
db_query_duration_seconds = Histogram(
    'db_query_duration_seconds',
    'Database query duration',
    ['operation'],
    registry=registry
)

db_connections_active = Gauge(
    'db_connections_active',
    'Active database connections',
    registry=registry
)

# Cache Metrics
cache_hits_total = Counter(
    'cache_hits_total',
    'Total cache hits',
    ['cache_type'],
    registry=registry
)

cache_misses_total = Counter(
    'cache_misses_total',
    'Total cache misses',
    ['cache_type'],
    registry=registry
)

# Content Pipeline Metrics
pipeline_processing_duration_seconds = Histogram(
    'pipeline_processing_duration_seconds',
    'Content pipeline processing duration',
    ['stage'],
    registry=registry
)

pipeline_errors_total = Counter(
    'pipeline_errors_total',
    'Pipeline processing errors',
    ['stage', 'error_type'],
    registry=registry
)

# Model Inference Metrics
model_inference_duration_seconds = Histogram(
    'model_inference_duration_seconds',
    'Model inference duration',
    ['model_name'],
    registry=registry
)

model_requests_total = Counter(
    'model_requests_total',
    'Total model inference requests',
    ['model_name', 'status'],
    registry=registry
)

# Validation Metrics
validation_checks_total = Counter(
    'validation_checks_total',
    'Total validation checks',
    ['validation_type', 'passed'],
    registry=registry
)

validation_score = Histogram(
    'validation_score',
    'Validation scores',
    ['validation_type'],
    registry=registry
)


class PrometheusMiddleware(BaseHTTPMiddleware):
    """Middleware to collect HTTP metrics."""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Collect metrics for HTTP requests."""
        
        method = request.method
        endpoint = request.url.path
        
        # Track in-progress requests
        http_requests_in_progress.labels(method=method, endpoint=endpoint).inc()
        
        start_time = time.time()
        
        try:
            response = await call_next(request)
            status = response.status_code
            
            # Record metrics
            duration = time.time() - start_time
            http_request_duration_seconds.labels(
                method=method,
                endpoint=endpoint
            ).observe(duration)
            
            http_requests_total.labels(
                method=method,
                endpoint=endpoint,
                status=status
            ).inc()
            
            return response
            
        except Exception as e:
            # Record error
            http_requests_total.labels(
                method=method,
                endpoint=endpoint,
                status=500
            ).inc()
            raise
            
        finally:
            # Decrement in-progress
            http_requests_in_progress.labels(method=method, endpoint=endpoint).dec()


# Helper functions for recording metrics
def record_cache_hit(cache_type: str = "redis"):
    """Record a cache hit."""
    cache_hits_total.labels(cache_type=cache_type).inc()


def record_cache_miss(cache_type: str = "redis"):
    """Record a cache miss."""
    cache_misses_total.labels(cache_type=cache_type).inc()


def record_db_query(operation: str, duration: float):
    """Record database query metrics."""
    db_query_duration_seconds.labels(operation=operation).observe(duration)


def record_pipeline_stage(stage: str, duration: float):
    """Record pipeline stage metrics."""
    pipeline_processing_duration_seconds.labels(stage=stage).observe(duration)


def record_pipeline_error(stage: str, error_type: str):
    """Record pipeline error."""
    pipeline_errors_total.labels(stage=stage, error_type=error_type).inc()


def record_model_inference(model_name: str, duration: float, success: bool = True):
    """Record model inference metrics."""
    model_inference_duration_seconds.labels(model_name=model_name).observe(duration)
    model_requests_total.labels(
        model_name=model_name,
        status="success" if success else "error"
    ).inc()


def record_validation(validation_type: str, passed: bool, score: float):
    """Record validation metrics."""
    validation_checks_total.labels(
        validation_type=validation_type,
        passed=str(passed)
    ).inc()
    validation_score.labels(validation_type=validation_type).observe(score)


# Metrics endpoint
async def metrics_endpoint(request: Request) -> Response:
    """Expose Prometheus metrics."""
    from fastapi.responses import PlainTextResponse
    
    metrics = generate_latest(registry)
    return PlainTextResponse(
        content=metrics.decode('utf-8'),
        media_type="text/plain; version=0.0.4"
    )


logger.info("Prometheus metrics configured")
