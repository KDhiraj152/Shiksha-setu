"""
Metrics API Route

Exposes Prometheus metrics endpoint for scraping.
"""
from fastapi import APIRouter, Response
from fastapi.responses import PlainTextResponse
import logging

from ...utils.metrics import get_metrics_response, get_metrics, PROMETHEUS_AVAILABLE

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/metrics", tags=["metrics"])


@router.get(
    "",
    summary="Prometheus Metrics",
    description="Get Prometheus-formatted metrics for scraping",
    responses={
        200: {
            "description": "Metrics in Prometheus format",
            "content": {"text/plain": {}}
        }
    }
)
async def get_prometheus_metrics():
    """
    Expose metrics in Prometheus format.
    
    Returns all collected metrics including:
    - HTTP request metrics
    - Model inference metrics
    - Cache hit/miss rates
    - Background task metrics
    - System resource metrics
    """
    content, content_type = get_metrics_response()
    
    if isinstance(content, dict):
        # Mock metrics - return JSON
        import json
        return Response(
            content=json.dumps(content),
            media_type="application/json"
        )
    
    return Response(content=content, media_type=content_type)


@router.get(
    "/status",
    summary="Metrics Status",
    description="Get metrics collection status and summary"
)
async def get_metrics_status():
    """Get summary of metrics collection status."""
    metrics = get_metrics()
    
    return {
        "prometheus_available": PROMETHEUS_AVAILABLE,
        "metrics_prefix": metrics.prefix,
        "http_metrics_enabled": True,
        "model_metrics_enabled": True,
        "cache_metrics_enabled": True,
        "task_metrics_enabled": True,
    }


@router.get(
    "/summary",
    summary="Metrics Summary",
    description="Get a JSON summary of key metrics"
)
async def get_metrics_summary():
    """
    Get human-readable summary of key metrics.
    
    Useful for dashboards and monitoring without Prometheus.
    """
    # This would query the actual metrics
    # For now, return structure
    return {
        "requests": {
            "total": 0,
            "per_minute": 0,
            "error_rate": 0.0,
            "avg_latency_ms": 0
        },
        "models": {
            "loaded": 0,
            "inferences_total": 0,
            "avg_inference_ms": 0
        },
        "cache": {
            "hit_rate": 0.0,
            "size_mb": 0
        },
        "tasks": {
            "pending": 0,
            "running": 0,
            "completed_last_hour": 0,
            "failed_last_hour": 0
        }
    }
