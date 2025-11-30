"""OpenTelemetry distributed tracing instrumentation."""
from typing import Optional
from contextlib import contextmanager

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor
from opentelemetry.sdk.resources import Resource

from ..core.config import settings
from ..utils.logging import get_logger

logger = get_logger(__name__)

# Global tracer
tracer: Optional[trace.Tracer] = None


def setup_telemetry(app):
    """
    Initialize OpenTelemetry distributed tracing.
    
    Args:
        app: FastAPI application instance
    """
    if not settings.OTEL_ENABLED:
        logger.info("OpenTelemetry disabled")
        return
    
    try:
        # Create resource with service information
        resource = Resource.create({
            "service.name": settings.APP_NAME,
            "service.version": settings.APP_VERSION,
            "deployment.environment": settings.ENVIRONMENT
        })
        
        # Create tracer provider
        provider = TracerProvider(resource=resource)
        
        # Configure OTLP exporter
        otlp_exporter = OTLPSpanExporter(
            endpoint=settings.OTEL_EXPORTER_OTLP_ENDPOINT,
            insecure=settings.ENVIRONMENT != "production"
        )
        
        # Add span processor
        provider.add_span_processor(BatchSpanProcessor(otlp_exporter))
        
        # Set global tracer provider
        trace.set_tracer_provider(provider)
        
        # Get tracer
        global tracer
        tracer = trace.get_tracer(__name__)
        
        # Instrument FastAPI
        FastAPIInstrumentor.instrument_app(app)
        
        # Instrument HTTP client
        HTTPXClientInstrumentor().instrument()
        
        # Instrument SQLAlchemy
        SQLAlchemyInstrumentor().instrument()
        
        logger.info(
            f"OpenTelemetry initialized: {settings.OTEL_EXPORTER_OTLP_ENDPOINT}"
        )
    
    except Exception as e:
        logger.error(f"Failed to initialize OpenTelemetry: {e}", exc_info=True)


@contextmanager
def trace_span(name: str, attributes: Optional[dict] = None):
    """
    Context manager for creating trace spans.
    
    Usage:
        with trace_span("my_operation", {"key": "value"}):
            # Your code here
            pass
    
    Args:
        name: Span name
        attributes: Optional span attributes
    """
    if tracer is None:
        yield None
        return
    
    with tracer.start_as_current_span(name) as span:
        if attributes:
            for key, value in attributes.items():
                span.set_attribute(key, str(value))
        yield span


def add_span_attribute(key: str, value: any):
    """Add attribute to current span."""
    current_span = trace.get_current_span()
    if current_span.is_recording():
        current_span.set_attribute(key, str(value))


def add_span_event(name: str, attributes: Optional[dict] = None):
    """Add event to current span."""
    current_span = trace.get_current_span()
    if current_span.is_recording():
        current_span.add_event(name, attributes or {})


def record_exception(exception: Exception):
    """Record exception in current span."""
    current_span = trace.get_current_span()
    if current_span.is_recording():
        current_span.record_exception(exception)
