"""
Translation Review API

Endpoints for collaborative translation review, version control,
and approval workflows.
"""
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List, Optional
import uuid
from datetime import datetime, timezone

from backend.database import get_db
from backend.models import TranslationReview, ReviewComment, ReviewVersion, User
from backend.schemas.review import (
    ReviewCreate, ReviewResponse, ReviewUpdate,
    CommentCreate, CommentResponse,
    VersionResponse
)
from backend.utils.auth import get_current_user
from backend.monitoring import track_review_action

router = APIRouter(prefix="/api/v1/reviews", tags=["reviews"])


@router.post("/", response_model=ReviewResponse, status_code=status.HTTP_201_CREATED)
async def create_review(
    review_data: ReviewCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Create a new translation review request.
    
    Args:
        review_data: Review details
        current_user: Authenticated user
        db: Database session
    
    Returns:
        Created review object
    """
    review = TranslationReview(
        id=uuid.uuid4(),
        translation_id=review_data.translation_id,
        original_text=review_data.original_text,
        translated_text=review_data.translated_text,
        source_lang=review_data.source_lang,
        target_lang=review_data.target_lang,
        reviewer_id=current_user.id,
        status=0,  # Pending
        metadata=review_data.metadata
    )
    
    db.add(review)
    db.commit()
    db.refresh(review)
    
    track_review_action("review_created", review.id, current_user.id)
    
    return review


@router.get("/{review_id}", response_model=ReviewResponse)
async def get_review(
    review_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get review details by ID."""
    review = db.query(TranslationReview).filter(
        TranslationReview.id == review_id
    ).first()
    
    if not review:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Review not found"
        )
    
    return review


@router.get("/", response_model=List[ReviewResponse])
async def list_reviews(
    status_filter: Optional[int] = None,
    reviewer_id: Optional[uuid.UUID] = None,
    skip: int = 0,
    limit: int = 50,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    List translation reviews with filters.
    
    Query params:
        status_filter: Filter by status (0=pending, 1=approved, 2=rejected, 3=revised)
        reviewer_id: Filter by reviewer
        skip: Pagination offset
        limit: Max results (default 50, max 100)
    """
    query = db.query(TranslationReview)
    
    if status_filter is not None:
        query = query.filter(TranslationReview.status == status_filter)
    
    if reviewer_id:
        query = query.filter(TranslationReview.reviewer_id == reviewer_id)
    
    # Order by most recent first
    query = query.order_by(TranslationReview.created_at.desc())
    
    reviews = query.offset(skip).limit(min(limit, 100)).all()
    
    return reviews


@router.patch("/{review_id}", response_model=ReviewResponse)
async def update_review(
    review_id: uuid.UUID,
    update_data: ReviewUpdate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Update review status or revised text.
    
    Automatically creates a version entry when revised_text is updated.
    """
    review = db.query(TranslationReview).filter(
        TranslationReview.id == review_id
    ).first()
    
    if not review:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Review not found"
        )
    
    # Update fields
    if update_data.status is not None:
        review.status = update_data.status
        track_review_action(f"status_changed_{update_data.status}", review.id, current_user.id)
    
    if update_data.revised_text is not None:
        # Create version entry
        version_count = db.query(ReviewVersion).filter(
            ReviewVersion.review_id == review_id
        ).count()
        
        version = ReviewVersion(
            id=uuid.uuid4(),
            review_id=review.id,
            version_number=version_count + 1,
            revised_text=update_data.revised_text,
            revised_by=current_user.id,
            change_description=update_data.change_description
        )
        db.add(version)
        
        review.revised_text = update_data.revised_text
        review.status = 3  # Revised status
        
        track_review_action("revision_created", review.id, current_user.id)
    
    if update_data.quality_score is not None:
        review.quality_score = update_data.quality_score
    
    if update_data.comments is not None:
        review.comments = update_data.comments
    
    review.updated_at = datetime.now(timezone.utc)
    
    db.commit()
    db.refresh(review)
    
    return review


@router.delete("/{review_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_review(
    review_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Delete a review (cascade deletes comments and versions)."""
    review = db.query(TranslationReview).filter(
        TranslationReview.id == review_id
    ).first()
    
    if not review:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Review not found"
        )
    
    # Only reviewer can delete
    if review.reviewer_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to delete this review"
        )
    
    db.delete(review)
    db.commit()
    
    track_review_action("review_deleted", review.id, current_user.id)


@router.post("/{review_id}/comments", response_model=CommentResponse, status_code=status.HTTP_201_CREATED)
async def add_comment(
    review_id: uuid.UUID,
    comment_data: CommentCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Add a comment to a review (supports threading)."""
    # Verify review exists
    review = db.query(TranslationReview).filter(
        TranslationReview.id == review_id
    ).first()
    
    if not review:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Review not found"
        )
    
    comment = ReviewComment(
        id=uuid.uuid4(),
        review_id=review_id,
        user_id=current_user.id,
        parent_comment_id=comment_data.parent_comment_id,
        comment_text=comment_data.comment_text
    )
    
    db.add(comment)
    db.commit()
    db.refresh(comment)
    
    track_review_action("comment_added", review_id, current_user.id)
    
    return comment


@router.get("/{review_id}/comments", response_model=List[CommentResponse])
async def get_comments(
    review_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get all comments for a review."""
    comments = db.query(ReviewComment).filter(
        ReviewComment.review_id == review_id
    ).order_by(ReviewComment.created_at).all()
    
    return comments


@router.get("/{review_id}/versions", response_model=List[VersionResponse])
async def get_versions(
    review_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get version history for a review."""
    versions = db.query(ReviewVersion).filter(
        ReviewVersion.review_id == review_id
    ).order_by(ReviewVersion.version_number).all()
    
    return versions


@router.patch("/{review_id}/comments/{comment_id}/resolve")
async def resolve_comment(
    review_id: uuid.UUID,
    comment_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Mark a comment as resolved."""
    comment = db.query(ReviewComment).filter(
        ReviewComment.id == comment_id,
        ReviewComment.review_id == review_id
    ).first()
    
    if not comment:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Comment not found"
        )
    
    comment.is_resolved = True
    db.commit()
    
    return {"status": "resolved"}
