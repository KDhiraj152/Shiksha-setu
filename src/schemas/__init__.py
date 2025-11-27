"""Pydantic schemas for ShikshaSetu API."""

from .auth import (
    Token,
    TokenData,
    UserCreate,
    UserLogin,
    UserResponse,
    RefreshTokenRequest,
)
from .content import (
    ChunkedUploadRequest,
    ProcessRequest,
    SimplifyRequest,
    TranslateRequest,
    ValidateRequest,
    TTSRequest,
    FeedbackRequest,
    TaskResponse,
    ContentResponse,
)
from .qa import (
    QAProcessRequest,
    QAQueryRequest,
    QAResponse,
    DocumentChunk,
    QAStatusResponse,
)


__all__ = [
    # Auth schemas
    "Token",
    "TokenData",
    "UserCreate",
    "UserLogin",
    "UserResponse",
    "RefreshTokenRequest",
    # Content schemas
    "ChunkedUploadRequest",
    "ProcessRequest",
    "SimplifyRequest",
    "TranslateRequest",
    "ValidateRequest",
    "TTSRequest",
    "FeedbackRequest",
    "TaskResponse",
    "ContentResponse",
    # Q&A schemas
    "QAProcessRequest",
    "QAQueryRequest",
    "QAResponse",
    "DocumentChunk",
    "QAStatusResponse",
]
