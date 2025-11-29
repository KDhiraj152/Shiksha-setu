"""
Structured Logging Module

Provides JSON-formatted logging with:
- Correlation IDs for request tracing
- Contextual information (user, request, etc.)
- Log level configuration
- Integration with external logging services
"""
import json
import logging
import sys
import traceback
import uuid
from contextvars import ContextVar
from datetime import datetime
from typing import Any, Dict, Optional
from functools import wraps

# Context variables for request-scoped data
_request_id: ContextVar[Optional[str]] = ContextVar("request_id", default=None)
_user_id: ContextVar[Optional[str]] = ContextVar("user_id", default=None)
_trace_id: ContextVar[Optional[str]] = ContextVar("trace_id", default=None)
_span_id: ContextVar[Optional[str]] = ContextVar("span_id", default=None)


def get_request_id() -> Optional[str]:
    """Get current request ID."""
    return _request_id.get()


def set_request_id(request_id: str) -> None:
    """Set current request ID."""
    _request_id.set(request_id)


def get_user_id() -> Optional[str]:
    """Get current user ID."""
    return _user_id.get()


def set_user_id(user_id: str) -> None:
    """Set current user ID."""
    _user_id.set(user_id)


def get_trace_context() -> Dict[str, Optional[str]]:
    """Get current trace context."""
    return {
        "trace_id": _trace_id.get(),
        "span_id": _span_id.get(),
        "request_id": _request_id.get(),
    }


def set_trace_context(trace_id: str, span_id: str) -> None:
    """Set trace context."""
    _trace_id.set(trace_id)
    _span_id.set(span_id)


class JSONFormatter(logging.Formatter):
    """
    JSON log formatter for structured logging.
    
    Outputs logs in JSON format with consistent fields for
    easy parsing by log aggregation systems.
    """
    
    def __init__(
        self,
        include_timestamp: bool = True,
        include_level: bool = True,
        include_logger: bool = True,
        include_trace: bool = True,
        extra_fields: Optional[Dict[str, Any]] = None
    ):
        super().__init__()
        self.include_timestamp = include_timestamp
        self.include_level = include_level
        self.include_logger = include_logger
        self.include_trace = include_trace
        self.extra_fields = extra_fields or {}
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_data = {}
        
        # Timestamp
        if self.include_timestamp:
            log_data["timestamp"] = datetime.utcnow().isoformat() + "Z"
        
        # Level
        if self.include_level:
            log_data["level"] = record.levelname
            log_data["level_num"] = record.levelno
        
        # Logger name
        if self.include_logger:
            log_data["logger"] = record.name
        
        # Message
        log_data["message"] = record.getMessage()
        
        # Trace context
        if self.include_trace:
            request_id = get_request_id()
            user_id = get_user_id()
            trace_id = _trace_id.get()
            span_id = _span_id.get()
            
            if request_id:
                log_data["request_id"] = request_id
            if user_id:
                log_data["user_id"] = user_id
            if trace_id:
                log_data["trace_id"] = trace_id
            if span_id:
                log_data["span_id"] = span_id
        
        # Location
        log_data["location"] = {
            "file": record.filename,
            "line": record.lineno,
            "function": record.funcName,
        }
        
        # Exception info
        if record.exc_info:
            log_data["exception"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                "message": str(record.exc_info[1]) if record.exc_info[1] else None,
                "traceback": traceback.format_exception(*record.exc_info) if record.exc_info[0] else None,
            }
        
        # Extra fields from record
        for key, value in record.__dict__.items():
            if key not in (
                "name", "msg", "args", "created", "filename", "funcName",
                "levelname", "levelno", "lineno", "module", "msecs",
                "pathname", "process", "processName", "relativeCreated",
                "stack_info", "exc_info", "exc_text", "thread", "threadName",
                "message"
            ):
                try:
                    json.dumps(value)  # Check if serializable
                    log_data[key] = value
                except (TypeError, ValueError):
                    log_data[key] = str(value)
        
        # Static extra fields
        log_data.update(self.extra_fields)
        
        return json.dumps(log_data, default=str)


class StructuredLogger:
    """
    Enhanced logger with structured output support.
    
    Provides methods for logging with extra context and
    automatic correlation ID injection.
    """
    
    def __init__(self, name: str, level: int = logging.INFO):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
    
    def _log(self, level: int, message: str, extra: Optional[Dict] = None, **kwargs):
        """Internal log method with extra context."""
        log_extra = extra or {}
        log_extra.update(kwargs)
        self.logger.log(level, message, extra=log_extra)
    
    def debug(self, message: str, **kwargs):
        """Log debug message."""
        self._log(logging.DEBUG, message, **kwargs)
    
    def info(self, message: str, **kwargs):
        """Log info message."""
        self._log(logging.INFO, message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning message."""
        self._log(logging.WARNING, message, **kwargs)
    
    def error(self, message: str, exc_info: bool = False, **kwargs):
        """Log error message."""
        self.logger.error(message, exc_info=exc_info, extra=kwargs)
    
    def critical(self, message: str, exc_info: bool = False, **kwargs):
        """Log critical message."""
        self.logger.critical(message, exc_info=exc_info, extra=kwargs)
    
    def exception(self, message: str, **kwargs):
        """Log exception with traceback."""
        self.logger.exception(message, extra=kwargs)
    
    # Contextual logging methods
    def log_request(self, method: str, path: str, **kwargs):
        """Log HTTP request."""
        self.info(
            f"HTTP {method} {path}",
            event_type="http_request",
            http_method=method,
            http_path=path,
            **kwargs
        )
    
    def log_response(self, status_code: int, duration_ms: float, **kwargs):
        """Log HTTP response."""
        self.info(
            f"HTTP Response {status_code}",
            event_type="http_response",
            http_status=status_code,
            duration_ms=duration_ms,
            **kwargs
        )
    
    def log_model_inference(
        self,
        model_name: str,
        duration_ms: float,
        input_tokens: int = 0,
        output_tokens: int = 0,
        **kwargs
    ):
        """Log model inference."""
        self.info(
            f"Model inference: {model_name}",
            event_type="model_inference",
            model_name=model_name,
            duration_ms=duration_ms,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            **kwargs
        )
    
    def log_cache_operation(
        self,
        operation: str,
        cache_type: str,
        hit: bool,
        key: str = None,
        **kwargs
    ):
        """Log cache operation."""
        self.debug(
            f"Cache {operation}: {'hit' if hit else 'miss'}",
            event_type="cache_operation",
            cache_operation=operation,
            cache_type=cache_type,
            cache_hit=hit,
            cache_key=key,
            **kwargs
        )
    
    def log_task(
        self,
        task_name: str,
        status: str,
        task_id: str = None,
        duration_ms: float = None,
        **kwargs
    ):
        """Log background task."""
        self.info(
            f"Task {task_name}: {status}",
            event_type="background_task",
            task_name=task_name,
            task_status=status,
            task_id=task_id,
            duration_ms=duration_ms,
            **kwargs
        )
    
    def log_error_event(
        self,
        error_type: str,
        error_message: str,
        **kwargs
    ):
        """Log error event."""
        self.error(
            f"Error: {error_type} - {error_message}",
            event_type="error",
            error_type=error_type,
            error_message=error_message,
            **kwargs
        )


def setup_structured_logging(
    level: str = "INFO",
    json_output: bool = True,
    service_name: str = "shiksha-setu",
    environment: str = "development"
):
    """
    Configure structured logging for the application.
    
    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        json_output: Whether to output JSON format
        service_name: Service name to include in logs
        environment: Environment name (development, staging, production)
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    
    # Remove existing handlers
    root_logger.handlers = []
    
    # Create handler
    handler = logging.StreamHandler(sys.stdout)
    
    if json_output:
        formatter = JSONFormatter(
            extra_fields={
                "service": service_name,
                "environment": environment,
            }
        )
    else:
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
    
    handler.setFormatter(formatter)
    root_logger.addHandler(handler)
    
    # Reduce noise from third-party libraries
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
    
    return root_logger


def get_logger(name: str) -> StructuredLogger:
    """Get a structured logger instance."""
    return StructuredLogger(name)


# ============ Request Logging Middleware ============

async def request_logging_middleware(request, call_next):
    """
    FastAPI middleware for request/response logging.
    
    Automatically logs all requests with correlation IDs.
    """
    import time
    
    # Generate request ID
    request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
    set_request_id(request_id)
    
    # Get user ID if available
    user_id = getattr(request.state, "user_id", None)
    if user_id:
        set_user_id(user_id)
    
    logger = get_logger("http")
    
    # Log request
    logger.log_request(
        method=request.method,
        path=str(request.url.path),
        query_params=dict(request.query_params),
        client_host=request.client.host if request.client else None,
    )
    
    start_time = time.time()
    
    try:
        response = await call_next(request)
        
        # Add request ID to response
        response.headers["X-Request-ID"] = request_id
        
        # Log response
        duration_ms = (time.time() - start_time) * 1000
        logger.log_response(
            status_code=response.status_code,
            duration_ms=round(duration_ms, 2),
        )
        
        return response
        
    except Exception as e:
        duration_ms = (time.time() - start_time) * 1000
        logger.log_error_event(
            error_type=type(e).__name__,
            error_message=str(e),
            duration_ms=round(duration_ms, 2),
        )
        raise


# ============ Logging Decorators ============

def log_function(logger_name: str = None):
    """
    Decorator to log function entry/exit.
    
    Usage:
        @log_function("my_module")
        def my_function(x, y):
            return x + y
    """
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            logger = get_logger(logger_name or func.__module__)
            func_name = func.__name__
            
            logger.debug(f"Entering {func_name}", function=func_name)
            start_time = time.time()
            
            try:
                result = await func(*args, **kwargs)
                duration_ms = (time.time() - start_time) * 1000
                logger.debug(
                    f"Exiting {func_name}",
                    function=func_name,
                    duration_ms=round(duration_ms, 2)
                )
                return result
            except Exception as e:
                duration_ms = (time.time() - start_time) * 1000
                logger.error(
                    f"Error in {func_name}: {e}",
                    function=func_name,
                    duration_ms=round(duration_ms, 2),
                    exc_info=True
                )
                raise
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            import time
            logger = get_logger(logger_name or func.__module__)
            func_name = func.__name__
            
            logger.debug(f"Entering {func_name}", function=func_name)
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                duration_ms = (time.time() - start_time) * 1000
                logger.debug(
                    f"Exiting {func_name}",
                    function=func_name,
                    duration_ms=round(duration_ms, 2)
                )
                return result
            except Exception as e:
                duration_ms = (time.time() - start_time) * 1000
                logger.error(
                    f"Error in {func_name}: {e}",
                    function=func_name,
                    duration_ms=round(duration_ms, 2),
                    exc_info=True
                )
                raise
        
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    return decorator
