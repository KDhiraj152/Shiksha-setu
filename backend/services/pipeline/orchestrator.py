"""
Pipeline Orchestrator Module.

Provides backward-compatible wrapper around the v2 concurrent orchestrator.
This module maintains the original API while leveraging the improved v2 implementation.
"""

from ...core.exceptions import ShikshaSetuException
from .orchestrator_v2 import (
    ConcurrentPipelineOrchestrator,
    PipelineCircuitBreaker,
    PipelineStage,
    ProcessedContentResult,
    ProcessingStatus,
    StageMetrics,
)


class PipelineValidationError(ShikshaSetuException):
    """Raised when pipeline validation fails."""

    pass


class PipelineStageError(ShikshaSetuException):
    """Raised when a pipeline stage fails."""

    def __init__(
        self, stage: str, message: str, original_error: Exception | None = None
    ):
        self.stage = stage
        self.original_error = original_error
        super().__init__(f"Pipeline stage '{stage}' failed: {message}")


# Alias for backward compatibility
ContentPipelineOrchestrator = ConcurrentPipelineOrchestrator


__all__ = [
    "ConcurrentPipelineOrchestrator",
    "ContentPipelineOrchestrator",
    "PipelineCircuitBreaker",
    "PipelineStage",
    "PipelineStageError",
    "PipelineValidationError",
    "ProcessedContentResult",
    "ProcessingStatus",
    "StageMetrics",
]
