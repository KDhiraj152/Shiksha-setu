"""
Unified Logging Module for ShikshaSetu

Provides structured JSON logging with consistent formatting across all modules.
Supports both development (pretty print) and production (JSON) modes.

Usage:
    from backend.utils.logging import get_logger, setup_logging
    
    # Initialize at app startup
    setup_logging()
    
    # Get logger in any module
    logger = get_logger(__name__)
    logger.info("Processing content", extra={"content_id": "123", "stage": "simplification"})
"""
import os
import sys
import json
import logging
import traceback
from datetime import datetime, timezone
from typing import Any, Dict, Optional
from pathlib import Path


# =============================================================================
# Configuration
# =============================================================================

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_FORMAT = os.getenv("LOG_FORMAT", "json")  # json | pretty
LOG_FILE = os.getenv("LOG_FILE", "logs/shiksha_setu.log")
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")


# =============================================================================
# Custom JSON Formatter
# =============================================================================

class JSONFormatter(logging.Formatter):
    """JSON log formatter for structured logging."""
    
    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        # Add extra fields
        if hasattr(record, "extra"):
            log_data.update(record.extra)
        
        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                "message": str(record.exc_info[1]) if record.exc_info[1] else None,
                "traceback": traceback.format_exception(*record.exc_info)
            }
        
        # Add request context if available
        if hasattr(record, "request_id"):
            log_data["request_id"] = record.request_id
        if hasattr(record, "user_id"):
            log_data["user_id"] = record.user_id
        
        return json.dumps(log_data, default=str)


class PrettyFormatter(logging.Formatter):
    """Human-readable formatter for development."""
    
    COLORS = {
        "DEBUG": "\033[36m",     # Cyan
        "INFO": "\033[32m",      # Green
        "WARNING": "\033[33m",   # Yellow
        "ERROR": "\033[31m",     # Red
        "CRITICAL": "\033[35m",  # Magenta
        "RESET": "\033[0m",
    }
    
    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelname, self.COLORS["RESET"])
        reset = self.COLORS["RESET"]
        
        # Format timestamp
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        
        # Base message
        message = f"{color}{timestamp} [{record.levelname:^8}]{reset} {record.name}: {record.getMessage()}"
        
        # Add extra context if available
        extra_fields = {}
        for key in ["request_id", "user_id", "content_id", "stage", "duration_ms"]:
            if hasattr(record, key):
                extra_fields[key] = getattr(record, key)
        
        if extra_fields:
            extras = " | ".join(f"{k}={v}" for k, v in extra_fields.items())
            message += f" {color}({extras}){reset}"
        
        # Add exception if present
        if record.exc_info:
            message += f"\n{self.formatException(record.exc_info)}"
        
        return message


# =============================================================================
# Logger Factory
# =============================================================================

class ContextAdapter(logging.LoggerAdapter):
    """Logger adapter that adds context to all log messages."""
    
    def process(self, msg: str, kwargs: Dict[str, Any]) -> tuple:
        # Merge extra context
        extra = kwargs.get("extra", {})
        extra.update(self.extra)
        kwargs["extra"] = extra
        return msg, kwargs


_loggers: Dict[str, logging.Logger] = {}
_logging_configured = False


def setup_logging(
    level: Optional[str] = None,
    format_type: Optional[str] = None,
    log_file: Optional[str] = None
) -> None:
    """
    Configure logging for the application.
    
    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format_type: Format type (json or pretty)
        log_file: Path to log file (None for stdout only)
    """
    global _logging_configured
    
    if _logging_configured:
        return
    
    level = level or LOG_LEVEL
    format_type = format_type or LOG_FORMAT
    log_file = log_file or LOG_FILE
    
    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level))
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Choose formatter based on environment
    if format_type == "json" or ENVIRONMENT == "production":
        formatter = JSONFormatter()
    else:
        formatter = PrettyFormatter()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5
        )
        file_handler.setFormatter(JSONFormatter())  # Always JSON for files
        root_logger.addHandler(file_handler)
    
    # Reduce noise from third-party libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    
    _logging_configured = True
    
    # Log startup
    logger = get_logger(__name__)
    logger.info(
        f"Logging configured",
        extra={
            "level": level,
            "format": format_type,
            "environment": ENVIRONMENT,
            "log_file": str(log_file) if log_file else None
        }
    )


def get_logger(name: str, **context) -> logging.LoggerAdapter:
    """
    Get a logger with optional context.
    
    Args:
        name: Logger name (usually __name__)
        **context: Additional context to include in all log messages
    
    Returns:
        Logger adapter with context
    
    Usage:
        logger = get_logger(__name__, service="pipeline")
        logger.info("Processing started", extra={"content_id": "123"})
    """
    if name not in _loggers:
        _loggers[name] = logging.getLogger(name)
    
    return ContextAdapter(_loggers[name], context)


# =============================================================================
# Convenience Functions
# =============================================================================

def log_request(
    method: str,
    path: str,
    status_code: int,
    duration_ms: float,
    request_id: Optional[str] = None,
    user_id: Optional[str] = None,
    **extra
) -> None:
    """Log an HTTP request."""
    logger = get_logger("http")
    logger.info(
        f"{method} {path} {status_code}",
        extra={
            "type": "request",
            "method": method,
            "path": path,
            "status_code": status_code,
            "duration_ms": round(duration_ms, 2),
            "request_id": request_id,
            "user_id": user_id,
            **extra
        }
    )


def log_pipeline_stage(
    stage: str,
    status: str,
    duration_ms: float,
    content_id: Optional[str] = None,
    **extra
) -> None:
    """Log a pipeline stage execution."""
    logger = get_logger("pipeline")
    level = logging.INFO if status == "success" else logging.WARNING
    
    logger.log(
        level,
        f"Pipeline stage: {stage} -> {status}",
        extra={
            "type": "pipeline",
            "stage": stage,
            "status": status,
            "duration_ms": round(duration_ms, 2),
            "content_id": content_id,
            **extra
        }
    )


def log_model_inference(
    model_id: str,
    task_type: str,
    duration_ms: float,
    input_tokens: Optional[int] = None,
    output_tokens: Optional[int] = None,
    **extra
) -> None:
    """Log a model inference call."""
    logger = get_logger("ml")
    logger.info(
        f"Model inference: {model_id} ({task_type})",
        extra={
            "type": "inference",
            "model_id": model_id,
            "task_type": task_type,
            "duration_ms": round(duration_ms, 2),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            **extra
        }
    )


# Import logging.handlers for RotatingFileHandler
import logging.handlers


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    "setup_logging",
    "get_logger",
    "log_request",
    "log_pipeline_stage",
    "log_model_inference",
    "JSONFormatter",
    "PrettyFormatter",
]
