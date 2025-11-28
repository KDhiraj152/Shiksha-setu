"""Content processing request/response schemas."""

from typing import List, Optional
from pydantic import BaseModel, Field


class ChunkedUploadRequest(BaseModel):
    """Chunked upload metadata."""
    filename: str
    chunk_index: int
    total_chunks: int
    upload_id: str
    checksum: Optional[str] = None


class ProcessRequest(BaseModel):
    """Full pipeline processing request."""
    grade_level: int = Field(ge=5, le=12)
    subject: str
    target_languages: List[str]
    output_format: str = Field(default='both', pattern='^(text|audio|both)$')
    validation_threshold: float = Field(default=0.80, ge=0.0, le=1.0)


class SimplifyRequest(BaseModel):
    """Text simplification request."""
    text: str = Field(min_length=10, max_length=50000)
    grade_level: Optional[int] = Field(None, ge=5, le=12)
    target_grade: Optional[int] = Field(None, ge=5, le=12)  # Backward compatibility
    subject: str = Field(default='General')
    
    def get_grade_level(self) -> int:
        """Get grade level from either field."""
        return self.grade_level or self.target_grade or 8


class TranslateRequest(BaseModel):
    """Translation request."""
    text: str = Field(min_length=10, max_length=50000)
    target_languages: Optional[List[str]] = None
    source_language: Optional[str] = None  # Backward compatibility
    target_language: Optional[str] = None  # Backward compatibility
    subject: str = Field(default='General')
    
    def get_target_languages(self) -> List[str]:
        """Get target languages from either format."""
        if self.target_languages:
            return self.target_languages
        elif self.target_language:
            return [self.target_language]
        return ['Hindi']


class ValidateRequest(BaseModel):
    """Content validation request."""
    text: Optional[str] = Field(None, min_length=10)  # Backward compatibility
    original_text: Optional[str] = Field(None, min_length=10)
    processed_text: Optional[str] = Field(None, min_length=10)
    grade_level: int = Field(ge=5, le=12)
    subject: str
    language: Optional[str] = Field(default='English')
    
    def get_texts(self) -> tuple[str, str]:
        """Get original and processed texts."""
        if self.text:
            # Use text for both if only text is provided
            return (self.text, self.text)
        return (self.original_text or '', self.processed_text or '')


class TTSRequest(BaseModel):
    """Text-to-speech request."""
    text: str = Field(min_length=10, max_length=10000)
    language: str
    subject: str = Field(default='General')


class FeedbackRequest(BaseModel):
    """User feedback request."""
    content_id: str
    rating: int = Field(ge=1, le=5)
    feedback_text: Optional[str] = None
    issue_type: Optional[str] = None


class TaskResponse(BaseModel):
    """Task status response."""
    task_id: str
    state: str
    result: Optional[dict] = None
    error: Optional[str] = None
    progress: Optional[int] = Field(None, ge=0, le=100)


class ContentResponse(BaseModel):
    """Content retrieval response."""
    content_id: str
    original_text: str
    simplified_text: Optional[str] = None
    translations: Optional[dict] = None
    validation_score: Optional[float] = None
    audio_files: Optional[dict] = None
    created_at: str
    metadata: Optional[dict] = None


__all__ = [
    "ChunkedUploadRequest",
    "ProcessRequest",
    "SimplifyRequest",
    "TranslateRequest",
    "ValidateRequest",
    "TTSRequest",
    "FeedbackRequest",
    "TaskResponse",
    "ContentResponse",
]
