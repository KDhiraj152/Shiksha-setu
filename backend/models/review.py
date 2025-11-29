"""
Review models for translation review workflow.
"""
from sqlalchemy import Column, String, Integer, Text, TIMESTAMP, ForeignKey, Boolean, Index
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship
import uuid
from datetime import datetime, timezone

from ..core.database import Base


def utcnow():
    """Get current UTC time with timezone awareness."""
    return datetime.now(timezone.utc)


class TranslationReview(Base):
    """Translation review requests for collaborative review workflow."""
    __tablename__ = 'translation_reviews'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    translation_id = Column(UUID(as_uuid=True), index=True)  # Reference to original translation
    original_text = Column(Text, nullable=False)
    translated_text = Column(Text, nullable=False)
    source_lang = Column(String(50), nullable=False, index=True)
    target_lang = Column(String(50), nullable=False, index=True)
    
    # Review details
    reviewer_id = Column(UUID(as_uuid=True), ForeignKey('users.id'), nullable=True, index=True)
    status = Column(Integer, default=0, nullable=False, index=True)  # 0=pending, 1=approved, 2=rejected, 3=revised
    feedback = Column(Text)
    
    # Metadata (renamed from 'metadata' to avoid SQLAlchemy reserved word)
    review_metadata = Column('metadata', JSONB)
    created_at = Column(TIMESTAMP, default=utcnow, index=True)
    updated_at = Column(TIMESTAMP, default=utcnow, onupdate=utcnow)
    
    # Relationships
    comments = relationship("ReviewComment", back_populates="review", cascade="all, delete-orphan")
    versions = relationship("ReviewVersion", back_populates="review", cascade="all, delete-orphan")
    
    # Indexes
    __table_args__ = (
        Index('idx_review_status_lang', 'status', 'target_lang'),
        Index('idx_review_reviewer_status', 'reviewer_id', 'status'),
    )


class ReviewComment(Base):
    """Comments on translation reviews."""
    __tablename__ = 'review_comments'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    review_id = Column(UUID(as_uuid=True), ForeignKey('translation_reviews.id', ondelete='CASCADE'), nullable=False, index=True)
    user_id = Column(UUID(as_uuid=True), ForeignKey('users.id'), nullable=False)
    
    text = Column(Text, nullable=False)
    position_start = Column(Integer)  # Character position for inline comments
    position_end = Column(Integer)
    resolved = Column(Boolean, default=False)
    
    created_at = Column(TIMESTAMP, default=utcnow)
    
    # Relationships
    review = relationship("TranslationReview", back_populates="comments")


class ReviewVersion(Base):
    """Version history for translations under review."""
    __tablename__ = 'review_versions'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    review_id = Column(UUID(as_uuid=True), ForeignKey('translation_reviews.id', ondelete='CASCADE'), nullable=False, index=True)
    version = Column(Integer, nullable=False)
    
    translated_text = Column(Text, nullable=False)
    created_by = Column(UUID(as_uuid=True), ForeignKey('users.id'), nullable=False)
    change_summary = Column(Text)
    
    created_at = Column(TIMESTAMP, default=utcnow)
    
    # Relationships
    review = relationship("TranslationReview", back_populates="versions")
    
    # Indexes
    __table_args__ = (
        Index('idx_version_review_version', 'review_id', 'version'),
    )
