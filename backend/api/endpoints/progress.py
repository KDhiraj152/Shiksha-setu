"""
Progress Tracking API Endpoints.

Provides endpoints for tracking student progress, quiz scores, and generating parent reports.
"""
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import List, Optional
from datetime import datetime, timedelta, timezone
import logging

from backend.database import get_db
from backend.models import StudentProgress, QuizScore, LearningSession, ParentReport, Achievement
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/progress", tags=["progress"])


# Pydantic models for request/response
class ProgressUpdate(BaseModel):
    """Progress update request."""
    user_id: str
    content_id: str
    progress_percent: float
    time_spent_seconds: int
    progress_data: dict = {}


class QuizSubmission(BaseModel):
    """Quiz submission request."""
    user_id: str
    content_id: str
    quiz_id: str
    score: float
    max_score: float = 100.0
    time_taken_seconds: Optional[int] = None
    answers: dict = {}


class SessionStart(BaseModel):
    """Learning session start request."""
    user_id: str
    content_id: str
    device_type: Optional[str] = None


class SessionEnd(BaseModel):
    """Learning session end request."""
    session_id: int
    interactions: int = 0
    pages_viewed: int = 0
    videos_watched: int = 0
    exercises_completed: int = 0


class ProgressResponse(BaseModel):
    """Progress response."""
    id: int
    user_id: str
    content_id: str
    progress_percent: float
    completed: bool
    time_spent_seconds: int
    started_at: datetime
    last_accessed: datetime
    
    class Config:
        from_attributes = True


@router.post("/update", response_model=ProgressResponse)
async def update_progress(
    progress: ProgressUpdate,
    db: Session = Depends(get_db)
):
    """
    Update student progress on content.
    """
    try:
        # Check if progress exists
        existing_progress = db.query(StudentProgress).filter(
            StudentProgress.user_id == progress.user_id,
            StudentProgress.content_id == progress.content_id
        ).first()
        
        if existing_progress:
            # Update existing progress
            existing_progress.progress_percent = progress.progress_percent
            existing_progress.time_spent_seconds += progress.time_spent_seconds
            existing_progress.progress_data = progress.progress_data
            existing_progress.last_accessed = datetime.now(timezone.utc)
            existing_progress.session_count += 1
            
            # Mark as completed if 100%
            if progress.progress_percent >= 100 and not existing_progress.completed:
                existing_progress.completed = True
                existing_progress.completed_at = datetime.now(timezone.utc)
            
            db.commit()
            db.refresh(existing_progress)
            logger.info(f"Updated progress for user {progress.user_id}, content {progress.content_id}")
            return existing_progress
        else:
            # Create new progress record
            new_progress = StudentProgress(
                user_id=progress.user_id,
                content_id=progress.content_id,
                progress_percent=progress.progress_percent,
                time_spent_seconds=progress.time_spent_seconds,
                progress_data=progress.progress_data,
                session_count=1,
                completed=progress.progress_percent >= 100,
                completed_at=datetime.now(timezone.utc) if progress.progress_percent >= 100 else None
            )
            
            db.add(new_progress)
            db.commit()
            db.refresh(new_progress)
            logger.info(f"Created progress for user {progress.user_id}, content {progress.content_id}")
            return new_progress
            
    except Exception as e:
        logger.error(f"Failed to update progress: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to update progress: {str(e)}")


@router.get("/user/{user_id}", response_model=List[ProgressResponse])
async def get_user_progress(
    user_id: str,
    db: Session = Depends(get_db)
):
    """
    Get all progress for a user.
    """
    try:
        progress_records = db.query(StudentProgress).filter(
            StudentProgress.user_id == user_id
        ).all()
        
        logger.info(f"Retrieved {len(progress_records)} progress records for user {user_id}")
        return progress_records
        
    except Exception as e:
        logger.error(f"Failed to get user progress: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get progress: {str(e)}")


@router.get("/content/{content_id}/users")
async def get_content_progress(
    content_id: str,
    db: Session = Depends(get_db)
):
    """
    Get progress statistics for a content item across all users.
    """
    try:
        progress_records = db.query(StudentProgress).filter(
            StudentProgress.content_id == content_id
        ).all()
        
        total_users = len(progress_records)
        completed_users = len([p for p in progress_records if p.completed])
        avg_progress = sum(p.progress_percent for p in progress_records) / total_users if total_users > 0 else 0
        avg_time = sum(p.time_spent_seconds for p in progress_records) / total_users if total_users > 0 else 0
        
        return {
            "content_id": content_id,
            "total_users": total_users,
            "completed_users": completed_users,
            "completion_rate": (completed_users / total_users * 100) if total_users > 0 else 0,
            "average_progress": avg_progress,
            "average_time_seconds": avg_time
        }
        
    except Exception as e:
        logger.error(f"Failed to get content progress: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get content progress: {str(e)}")


@router.post("/quiz/submit")
async def submit_quiz(
    submission: QuizSubmission,
    db: Session = Depends(get_db)
):
    """
    Submit quiz score.
    """
    try:
        # Get progress record
        progress = db.query(StudentProgress).filter(
            StudentProgress.user_id == submission.user_id,
            StudentProgress.content_id == submission.content_id
        ).first()
        
        if not progress:
            # Create progress if doesn't exist
            progress = StudentProgress(
                user_id=submission.user_id,
                content_id=submission.content_id,
                progress_percent=0,
                time_spent_seconds=0
            )
            db.add(progress)
            db.commit()
            db.refresh(progress)
        
        # Get attempt number
        existing_attempts = db.query(QuizScore).filter(
            QuizScore.user_id == submission.user_id,
            QuizScore.quiz_id == submission.quiz_id
        ).count()
        
        # Create quiz score
        quiz_score = QuizScore(
            progress_id=progress.id,
            user_id=submission.user_id,
            content_id=submission.content_id,
            quiz_id=submission.quiz_id,
            score=submission.score,
            max_score=submission.max_score,
            passed=submission.score >= (submission.max_score * 0.6),  # 60% passing
            attempt_number=existing_attempts + 1,
            time_taken_seconds=submission.time_taken_seconds,
            answers=submission.answers
        )
        
        db.add(quiz_score)
        db.commit()
        db.refresh(quiz_score)
        
        logger.info(f"Saved quiz score for user {submission.user_id}, quiz {submission.quiz_id}: {submission.score}")
        
        return {
            "id": quiz_score.id,
            "score": quiz_score.score,
            "max_score": quiz_score.max_score,
            "passed": quiz_score.passed,
            "attempt_number": quiz_score.attempt_number,
            "submitted_at": quiz_score.submitted_at
        }
        
    except Exception as e:
        logger.error(f"Failed to submit quiz: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to submit quiz: {str(e)}")


@router.get("/quiz/{user_id}/{quiz_id}")
async def get_quiz_scores(
    user_id: str,
    quiz_id: str,
    db: Session = Depends(get_db)
):
    """
    Get all attempts for a quiz by user.
    """
    try:
        scores = db.query(QuizScore).filter(
            QuizScore.user_id == user_id,
            QuizScore.quiz_id == quiz_id
        ).order_by(QuizScore.submitted_at.desc()).all()
        
        return {
            "quiz_id": quiz_id,
            "total_attempts": len(scores),
            "best_score": max([s.score for s in scores]) if scores else 0,
            "latest_score": scores[0].score if scores else 0,
            "passed": any(s.passed for s in scores),
            "attempts": [
                {
                    "attempt": s.attempt_number,
                    "score": s.score,
                    "passed": s.passed,
                    "submitted_at": s.submitted_at
                }
                for s in scores
            ]
        }
        
    except Exception as e:
        logger.error(f"Failed to get quiz scores: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get quiz scores: {str(e)}")


@router.post("/session/start")
async def start_session(
    session: SessionStart,
    db: Session = Depends(get_db)
):
    """
    Start a learning session.
    """
    try:
        new_session = LearningSession(
            user_id=session.user_id,
            content_id=session.content_id,
            device_type=session.device_type,
            started_at=datetime.now(timezone.utc)
        )
        
        db.add(new_session)
        db.commit()
        db.refresh(new_session)
        
        logger.info(f"Started session {new_session.id} for user {session.user_id}")
        
        return {
            "session_id": new_session.id,
            "started_at": new_session.started_at
        }
        
    except Exception as e:
        logger.error(f"Failed to start session: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to start session: {str(e)}")


@router.post("/session/end")
async def end_session(
    session_end: SessionEnd,
    db: Session = Depends(get_db)
):
    """
    End a learning session.
    """
    try:
        session = db.query(LearningSession).filter(
            LearningSession.id == session_end.session_id
        ).first()
        
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Update session
        session.ended_at = datetime.now(timezone.utc)
        session.duration_seconds = int((session.ended_at - session.started_at).total_seconds())
        session.interactions = session_end.interactions
        session.pages_viewed = session_end.pages_viewed
        session.videos_watched = session_end.videos_watched
        session.exercises_completed = session_end.exercises_completed
        
        db.commit()
        
        logger.info(f"Ended session {session.id}, duration: {session.duration_seconds}s")
        
        return {
            "session_id": session.id,
            "duration_seconds": session.duration_seconds,
            "interactions": session.interactions
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to end session: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to end session: {str(e)}")


@router.get("/report/{student_id}")
async def generate_parent_report(
    student_id: str,
    report_type: str = Query("weekly", regex="^(weekly|monthly|quarterly)$"),
    db: Session = Depends(get_db)
):
    """
    Generate parent report for student.
    """
    try:
        # Calculate period
        end_date = datetime.now(timezone.utc)
        if report_type == "weekly":
            start_date = end_date - timedelta(days=7)
        elif report_type == "monthly":
            start_date = end_date - timedelta(days=30)
        else:  # quarterly
            start_date = end_date - timedelta(days=90)
        
        # Get progress data
        progress_records = db.query(StudentProgress).filter(
            StudentProgress.user_id == student_id,
            StudentProgress.last_accessed >= start_date
        ).all()
        
        # Get quiz scores
        quiz_scores = db.query(QuizScore).filter(
            QuizScore.user_id == student_id,
            QuizScore.submitted_at >= start_date
        ).all()
        
        # Calculate statistics
        total_content = len(progress_records)
        completed_content = len([p for p in progress_records if p.completed])
        total_time = sum(p.time_spent_seconds for p in progress_records)
        avg_score = sum(q.score for q in quiz_scores) / len(quiz_scores) if quiz_scores else 0
        
        summary = {
            "period": report_type,
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "total_content_accessed": total_content,
            "completed_content": completed_content,
            "completion_rate": (completed_content / total_content * 100) if total_content > 0 else 0,
            "total_time_hours": total_time / 3600,
            "total_quizzes": len(quiz_scores),
            "average_quiz_score": avg_score,
            "quiz_pass_rate": len([q for q in quiz_scores if q.passed]) / len(quiz_scores) * 100 if quiz_scores else 0
        }
        
        # Get achievements
        achievements = db.query(Achievement).filter(
            Achievement.user_id == student_id,
            Achievement.earned_at >= start_date
        ).all()
        
        # Create report
        report = ParentReport(
            student_id=student_id,
            parent_id=f"parent_{student_id}",  # In production, get from user relationship
            report_type=report_type,
            period_start=start_date,
            period_end=end_date,
            summary=summary,
            statistics={
                "daily_average_minutes": (total_time / 60) / ((end_date - start_date).days or 1),
                "most_active_day": "Monday",  # Would calculate from sessions
                "subjects_covered": list({p.content_id.split('_')[0] for p in progress_records if '_' in p.content_id})
            },
            achievements=[
                {
                    "title": a.title,
                    "description": a.description,
                    "earned_at": a.earned_at.isoformat()
                }
                for a in achievements
            ],
            recommendations={
                "focus_areas": ["Practice more quizzes", "Complete pending content"],
                "suggested_content": [],
                "study_tips": ["Set daily learning goals", "Review completed content"]
            }
        )
        
        db.add(report)
        db.commit()
        db.refresh(report)
        
        logger.info(f"Generated {report_type} report for student {student_id}")
        
        return {
            "report_id": report.id,
            "generated_at": report.generated_at,
            "summary": summary,
            "statistics": report.statistics,
            "achievements": report.achievements,
            "recommendations": report.recommendations
        }
        
    except Exception as e:
        logger.error(f"Failed to generate report: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to generate report: {str(e)}")


@router.get("/stats/{user_id}")
async def get_user_stats(
    user_id: str,
    days: int = Query(30, ge=1, le=365),
    db: Session = Depends(get_db)
):
    """
    Get comprehensive user statistics.
    """
    try:
        start_date = datetime.now(timezone.utc) - timedelta(days=days)
        
        # Progress stats
        progress_records = db.query(StudentProgress).filter(
            StudentProgress.user_id == user_id,
            StudentProgress.last_accessed >= start_date
        ).all()
        
        # Quiz stats
        quiz_scores = db.query(QuizScore).filter(
            QuizScore.user_id == user_id,
            QuizScore.submitted_at >= start_date
        ).all()
        
        # Session stats
        sessions = db.query(LearningSession).filter(
            LearningSession.user_id == user_id,
            LearningSession.started_at >= start_date
        ).all()
        
        return {
            "user_id": user_id,
            "period_days": days,
            "progress": {
                "total_content": len(progress_records),
                "completed": len([p for p in progress_records if p.completed]),
                "in_progress": len([p for p in progress_records if not p.completed and p.progress_percent > 0]),
                "total_time_hours": sum(p.time_spent_seconds for p in progress_records) / 3600
            },
            "quizzes": {
                "total_attempts": len(quiz_scores),
                "passed": len([q for q in quiz_scores if q.passed]),
                "average_score": sum(q.score for q in quiz_scores) / len(quiz_scores) if quiz_scores else 0,
                "best_score": max([q.score for q in quiz_scores]) if quiz_scores else 0
            },
            "sessions": {
                "total_sessions": len(sessions),
                "total_duration_hours": sum(s.duration_seconds for s in sessions if s.duration_seconds) / 3600,
                "average_session_minutes": (sum(s.duration_seconds for s in sessions if s.duration_seconds) / len(sessions) / 60) if sessions else 0
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to get user stats: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")
