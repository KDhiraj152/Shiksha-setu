"""
Review schemas for API request/response validation.
"""
from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, Dict, Any, List
from datetime import datetime
import uuid


class ReviewCreate(BaseModel):
    """Schema for creating a new review."""
    translation_id: uuid.UUID
    original_text: str = Field(..., min_length=1)
    translated_text: str = Field(..., min_length=1)
    source_lang: str = Field(..., min_length=2, max_length=50)
    target_lang: str = Field(..., min_length=2, max_length=50)
    metadata: Optional[Dict[str, Any]] = None


class ReviewUpdate(BaseModel):
    """Schema for updating a review."""
    translated_text: Optional[str] = None
    status: Optional[int] = Field(None, ge=0, le=3)  # 0-3 for status codes
    feedback: Optional[str] = None


class ReviewResponse(BaseModel):
    """Schema for review response."""
    id: uuid.UUID
    translation_id: uuid.UUID
    original_text: str
    translated_text: str
    source_lang: str
    target_lang: str
    reviewer_id: Optional[uuid.UUID] = None
    status: int
    feedback: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    created_at: datetime
    updated_at: datetime
    
    model_config = ConfigDict(from_attributes=True)


class CommentCreate(BaseModel):
    """Schema for creating a comment on a review."""
    text: str = Field(..., min_length=1)
    position_start: Optional[int] = None
    position_end: Optional[int] = None


class CommentResponse(BaseModel):
    """Schema for comment response."""
    id: uuid.UUID
    review_id: uuid.UUID
    user_id: uuid.UUID
    text: str
    position_start: Optional[int] = None
    position_end: Optional[int] = None
    resolved: bool
    created_at: datetime
    
    model_config = ConfigDict(from_attributes=True)


class VersionResponse(BaseModel):
    """Schema for version response."""
    id: uuid.UUID
    review_id: uuid.UUID
    version: int
    translated_text: str
    created_by: uuid.UUID
    change_summary: Optional[str] = None
    created_at: datetime
    
    model_config = ConfigDict(from_attributes=True)


class ReviewListResponse(BaseModel):
    """Schema for paginated review list."""
    items: List[ReviewResponse]
    total: int
    skip: int
    limit: int
