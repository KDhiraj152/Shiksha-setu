"""
Monitoring Module (Principles S, T)
====================================
Prometheus metrics and OOM alerting for system monitoring.
"""

from .metrics import (
    # Recording functions
    record_inference_latency,
    record_tokens,
    record_model_memory,
    record_model_load_time,
    record_queue_length,
    record_queue_wait,
    record_error,
    record_cache_hit,
    record_cache_miss,
    record_system_memory,
    record_gpu_memory,
    
    # Context managers and decorators
    track_inference,
    track_latency,
    
    # Metrics endpoint
    get_metrics,
    get_metrics_content_type,
    setup_metrics_endpoint,
    
    # Latency tracking
    log_latency,
    get_latency_stats,
    LatencyTracker,
)

from .oom_alerts import (
    # Alert types
    Alert,
    AlertLevel,
    AlertType,
    
    # Alert handlers
    AlertHandler,
    LogAlertHandler,
    WebhookAlertHandler,
    SlackAlertHandler,
    EvictionAlertHandler,
    
    # Alert manager
    OOMAlertManager,
    OOMAlertConfig,
    get_alert_manager,
    
    # Convenience functions
    start_oom_monitoring,
    stop_oom_monitoring,
    get_memory_status,
)

__all__ = [
    # Metrics (Principle S)
    'record_inference_latency',
    'record_tokens',
    'record_model_memory',
    'record_model_load_time',
    'record_queue_length',
    'record_queue_wait',
    'record_error',
    'record_cache_hit',
    'record_cache_miss',
    'record_system_memory',
    'record_gpu_memory',
    'track_inference',
    'track_latency',
    'get_metrics',
    'get_metrics_content_type',
    'setup_metrics_endpoint',
    'log_latency',
    'get_latency_stats',
    'LatencyTracker',
    
    # OOM Alerts (Principle T)
    'Alert',
    'AlertLevel',
    'AlertType',
    'AlertHandler',
    'LogAlertHandler',
    'WebhookAlertHandler',
    'SlackAlertHandler',
    'EvictionAlertHandler',
    'OOMAlertManager',
    'OOMAlertConfig',
    'get_alert_manager',
    'start_oom_monitoring',
    'stop_oom_monitoring',
    'get_memory_status',
]
