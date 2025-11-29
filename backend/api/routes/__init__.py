"""
API route modules initialization.

Handles graceful fallback if optional route modules fail to import.
Core routes (health, auth, content) are always required.
"""
import logging
from fastapi import APIRouter

logger = logging.getLogger(__name__)

# Create placeholder router for failed imports
def create_placeholder_router(name: str, error: Exception) -> APIRouter:
    """Create a placeholder router that returns service unavailable."""
    router = APIRouter()
    
    @router.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
    async def placeholder(path: str):
        from fastapi import HTTPException
        raise HTTPException(
            status_code=503,
            detail=f"{name} service unavailable: {str(error)[:100]}"
        )
    
    return router


# Core routes - REQUIRED (will raise on import failure)
from .health import router as health_router
from .auth import router as auth_router
from .content import router as content_router

# Optional routes - graceful fallback
try:
    from .qa import router as qa_router
except ImportError as e:
    logger.warning(f"Q&A routes unavailable: {e}")
    qa_router = create_placeholder_router("Q&A", e)

try:
    from .streaming import router as streaming_router
except ImportError as e:
    logger.warning(f"Streaming routes unavailable: {e}")
    streaming_router = create_placeholder_router("Streaming", e)

try:
    from .progress import router as progress_router
except ImportError as e:
    logger.warning(f"Progress routes unavailable: {e}")
    progress_router = create_placeholder_router("Progress", e)

try:
    from .review import router as review_router
except ImportError as e:
    logger.warning(f"Review routes unavailable: {e}")
    review_router = create_placeholder_router("Review", e)

try:
    from .admin import router as admin_router
except ImportError as e:
    logger.warning(f"Admin routes unavailable: {e}")
    admin_router = create_placeholder_router("Admin", e)

try:
    from .experiments import router as experiments_router
except ImportError as e:
    logger.warning(f"Experiments routes unavailable: {e}")
    experiments_router = create_placeholder_router("Experiments", e)

try:
    from .audio_upload import router as audio_router
except ImportError as e:
    logger.warning(f"Audio upload routes unavailable: {e}")
    audio_router = create_placeholder_router("Audio", e)

try:
    from .quantization import router as quantization_router
except ImportError as e:
    logger.warning(f"Quantization routes unavailable: {e}")
    quantization_router = create_placeholder_router("Quantization", e)

__all__ = [
    "health_router",
    "auth_router",
    "content_router",
    "qa_router",
    "streaming_router",
    "progress_router",
    "review_router",
    "admin_router",
    "experiments_router",
    "audio_router",
    "quantization_router",
]

