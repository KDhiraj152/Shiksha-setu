"""Request context utilities for distributed tracing."""
import contextvars
from typing import Optional

# Context variable for request ID
request_id_var: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    'request_id',
    default=None
)


def get_request_id() -> Optional[str]:
    """Get the current request ID from context."""
    return request_id_var.get()


def set_request_id(request_id: str) -> None:
    """Set the request ID in context."""
    request_id_var.set(request_id)


class RequestContextFilter:
    """Logging filter to add request ID to all log records."""
    
    def filter(self, record):
        """Add request_id to log record."""
        record.request_id = get_request_id() or "no-request-id"
        return True
