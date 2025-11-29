"""
Teacher Performance Evaluation Foundation

Issue: CODE-REVIEW-GPT #2 (CRITICAL)
Problem: PS4 completely missing - no evaluation framework exists

Solution: Implement foundation for teacher evaluation system with:
- Content quality metrics
- Student engagement analytics
- Learning outcome assessment
- Performance scoring algorithms
"""

from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any, Optional
from sqlalchemy import Column, String, Integer, Float, Text, TIMESTAMP, ForeignKey, Boolean, Index
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import Session
import uuid
from enum import Enum

from ..core.database import Base


def utcnow():
    """Get current UTC time with timezone awareness."""
    return datetime.now(timezone.utc)


# =============================================================================
# TEACHER EVALUATION MODELS
# =============================================================================

class TeacherProfile(Base):
    """Teacher profile with evaluation metadata."""
    __tablename__ = 'teacher_profiles'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey('users.id', ondelete='CASCADE'), unique=True, nullable=False, index=True)
    
    # Profile information
    employee_id = Column(String(100), unique=True, index=True)
    qualification = Column(String(255))
    specialization = Column(String(255))
    experience_years = Column(Float)
    
    # Teaching scope
    subjects = Column(JSONB, default=[])  # List of subjects taught
    grade_levels = Column(JSONB, default=[])  # Grade levels taught
    
    # Performance metrics (aggregated)
    overall_rating = Column(Float, default=0.0)  # 0-5 scale
    content_quality_score = Column(Float, default=0.0)  # 0-100
    student_engagement_score = Column(Float, default=0.0)  # 0-100
    learning_outcome_score = Column(Float, default=0.0)  # 0-100
    
    # Status
    is_active = Column(Boolean, default=True)
    created_at = Column(TIMESTAMP, default=utcnow, nullable=False)
    updated_at = Column(TIMESTAMP, default=utcnow, onupdate=utcnow)
    last_evaluated_at = Column(TIMESTAMP)
    
    __table_args__ = (
        Index('idx_teacher_rating', 'overall_rating'),
        Index('idx_teacher_subjects', 'subjects', postgresql_using='gin'),
    )


class ContentQualityMetric(Base):
    """Track quality metrics for teacher-created content."""
    __tablename__ = 'content_quality_metrics'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    teacher_id = Column(UUID(as_uuid=True), ForeignKey('teacher_profiles.id', ondelete='CASCADE'), nullable=False, index=True)
    content_id = Column(UUID(as_uuid=True), ForeignKey('processed_content.id', ondelete='CASCADE'), nullable=False, index=True)
    
    # Quality dimensions
    ncert_alignment_score = Column(Float, nullable=False)  # 0-100
    language_clarity_score = Column(Float, nullable=False)  # 0-100
    pedagogical_effectiveness = Column(Float, nullable=False)  # 0-100
    cultural_sensitivity_score = Column(Float, nullable=False)  # 0-100
    
    # Engagement metrics
    views_count = Column(Integer, default=0)
    completion_rate = Column(Float, default=0.0)  # Percentage
    average_time_spent = Column(Float, default=0.0)  # Seconds
    
    # Feedback
    positive_feedback_count = Column(Integer, default=0)
    negative_feedback_count = Column(Integer, default=0)
    average_rating = Column(Float, default=0.0)  # 1-5 scale
    
    # Aggregate quality score (weighted average)
    overall_quality_score = Column(Float, nullable=False)  # 0-100
    
    evaluated_at = Column(TIMESTAMP, default=utcnow, nullable=False, index=True)
    
    __table_args__ = (
        Index('idx_quality_teacher_date', 'teacher_id', 'evaluated_at'),
        Index('idx_quality_score', 'overall_quality_score'),
    )


class StudentEngagementMetric(Base):
    """Track student engagement with teacher's content."""
    __tablename__ = 'student_engagement_metrics'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    teacher_id = Column(UUID(as_uuid=True), ForeignKey('teacher_profiles.id', ondelete='CASCADE'), nullable=False, index=True)
    content_id = Column(UUID(as_uuid=True), ForeignKey('processed_content.id', ondelete='CASCADE'), nullable=False, index=True)
    
    # Engagement indicators
    total_students = Column(Integer, default=0)
    active_students = Column(Integer, default=0)  # Engaged in last 7 days
    completion_count = Column(Integer, default=0)
    dropout_count = Column(Integer, default=0)
    
    # Interaction metrics
    question_asked_count = Column(Integer, default=0)
    discussion_participation = Column(Integer, default=0)
    resource_download_count = Column(Integer, default=0)
    
    # Time metrics
    average_session_duration = Column(Float, default=0.0)  # Minutes
    total_learning_time = Column(Float, default=0.0)  # Hours
    
    # Engagement score (composite)
    engagement_score = Column(Float, nullable=False)  # 0-100
    
    period_start = Column(TIMESTAMP, nullable=False)
    period_end = Column(TIMESTAMP, nullable=False)
    calculated_at = Column(TIMESTAMP, default=utcnow, nullable=False, index=True)
    
    __table_args__ = (
        Index('idx_engagement_teacher_period', 'teacher_id', 'period_start', 'period_end'),
    )


class LearningOutcomeMetric(Base):
    """Track learning outcomes for teacher's students."""
    __tablename__ = 'learning_outcome_metrics'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    teacher_id = Column(UUID(as_uuid=True), ForeignKey('teacher_profiles.id', ondelete='CASCADE'), nullable=False, index=True)
    content_id = Column(UUID(as_uuid=True), ForeignKey('processed_content.id', ondelete='CASCADE'), nullable=False, index=True)
    
    # Assessment metrics
    total_assessments = Column(Integer, default=0)
    passed_count = Column(Integer, default=0)
    failed_count = Column(Integer, default=0)
    
    # Score statistics
    average_score = Column(Float, default=0.0)  # 0-100
    median_score = Column(Float, default=0.0)
    std_deviation = Column(Float, default=0.0)
    
    # Improvement metrics
    pre_assessment_avg = Column(Float, default=0.0)
    post_assessment_avg = Column(Float, default=0.0)
    improvement_percentage = Column(Float, default=0.0)
    
    # Retention metrics
    knowledge_retention_rate = Column(Float, default=0.0)  # Tested after 30 days
    concept_mastery_rate = Column(Float, default=0.0)  # Students achieving 80%+
    
    # Learning outcome score (composite)
    outcome_score = Column(Float, nullable=False)  # 0-100
    
    period_start = Column(TIMESTAMP, nullable=False)
    period_end = Column(TIMESTAMP, nullable=False)
    calculated_at = Column(TIMESTAMP, default=utcnow, nullable=False, index=True)
    
    __table_args__ = (
        Index('idx_outcome_teacher_period', 'teacher_id', 'period_start', 'period_end'),
    )


class TeacherEvaluation(Base):
    """Comprehensive teacher evaluation report."""
    __tablename__ = 'teacher_evaluations'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    teacher_id = Column(UUID(as_uuid=True), ForeignKey('teacher_profiles.id', ondelete='CASCADE'), nullable=False, index=True)
    
    # Evaluation period
    evaluation_type = Column(String(50), nullable=False)  # 'monthly', 'quarterly', 'annual'
    period_start = Column(TIMESTAMP, nullable=False)
    period_end = Column(TIMESTAMP, nullable=False)
    
    # Aggregated scores
    content_quality_score = Column(Float, nullable=False)  # 0-100
    student_engagement_score = Column(Float, nullable=False)  # 0-100
    learning_outcome_score = Column(Float, nullable=False)  # 0-100
    
    # Weighted overall score
    overall_score = Column(Float, nullable=False)  # 0-100
    overall_rating = Column(Float, nullable=False)  # 0-5 stars
    
    # Performance indicators
    total_content_created = Column(Integer, default=0)
    total_students_taught = Column(Integer, default=0)
    average_completion_rate = Column(Float, default=0.0)
    average_student_satisfaction = Column(Float, default=0.0)
    
    # Strengths and areas for improvement
    strengths = Column(JSONB, default=[])
    improvements_needed = Column(JSONB, default=[])
    recommendations = Column(JSONB, default=[])
    
    # Status
    status = Column(String(50), default='draft')  # 'draft', 'finalized', 'reviewed'
    evaluator_id = Column(UUID(as_uuid=True), ForeignKey('users.id'), nullable=True)  # Admin who finalized
    
    created_at = Column(TIMESTAMP, default=utcnow, nullable=False)
    finalized_at = Column(TIMESTAMP)
    
    __table_args__ = (
        Index('idx_eval_teacher_period', 'teacher_id', 'period_start', 'period_end'),
        Index('idx_eval_overall_score', 'overall_score'),
    )


# =============================================================================
# EVALUATION SERVICE
# =============================================================================

class EvaluationWeights:
    """Configurable weights for evaluation scoring."""
    # Content quality weights
    NCERT_ALIGNMENT = 0.30
    LANGUAGE_CLARITY = 0.25
    PEDAGOGICAL_EFFECTIVENESS = 0.25
    CULTURAL_SENSITIVITY = 0.20
    
    # Overall evaluation weights
    CONTENT_QUALITY = 0.40
    STUDENT_ENGAGEMENT = 0.35
    LEARNING_OUTCOMES = 0.25


class TeacherEvaluationService:
    """Service for calculating teacher performance metrics."""
    
    def __init__(self, db: Session):
        self.db = db
    
    def calculate_content_quality_score(
        self,
        ncert_score: float,
        clarity_score: float,
        pedagogical_score: float,
        cultural_score: float
    ) -> float:
        """Calculate weighted content quality score."""
        return (
            ncert_score * EvaluationWeights.NCERT_ALIGNMENT +
            clarity_score * EvaluationWeights.LANGUAGE_CLARITY +
            pedagogical_score * EvaluationWeights.PEDAGOGICAL_EFFECTIVENESS +
            cultural_score * EvaluationWeights.CULTURAL_SENSITIVITY
        )
    
    def calculate_engagement_score(
        self,
        completion_rate: float,
        active_ratio: float,
        interaction_rate: float,
        session_duration: float
    ) -> float:
        """
        Calculate student engagement score.
        
        Args:
            completion_rate: Percentage of students completing content (0-100)
            active_ratio: Ratio of active to total students (0-1)
            interaction_rate: Questions/discussions per student
            session_duration: Average session duration in minutes
        """
        # Normalize metrics to 0-100 scale
        completion_component = completion_rate  # Already 0-100
        active_component = active_ratio * 100  # Convert to percentage
        
        # Normalize interaction rate (assume 5+ interactions is excellent)
        interaction_component = min(interaction_rate / 5.0, 1.0) * 100
        
        # Normalize session duration (30+ minutes is excellent)
        duration_component = min(session_duration / 30.0, 1.0) * 100
        
        # Weighted average
        return (
            completion_component * 0.35 +
            active_component * 0.30 +
            interaction_component * 0.20 +
            duration_component * 0.15
        )
    
    def calculate_outcome_score(
        self,
        pass_rate: float,
        average_score: float,
        improvement: float,
        mastery_rate: float
    ) -> float:
        """
        Calculate learning outcome score.
        
        Args:
            pass_rate: Percentage of students passing (0-100)
            average_score: Average assessment score (0-100)
            improvement: Improvement from pre to post assessment (0-100)
            mastery_rate: Percentage achieving mastery (80%+)
        """
        return (
            pass_rate * 0.25 +
            average_score * 0.30 +
            improvement * 0.25 +
            mastery_rate * 0.20
        )
    
    def calculate_overall_score(
        self,
        content_quality: float,
        engagement: float,
        outcomes: float
    ) -> float:
        """Calculate overall teacher evaluation score."""
        return (
            content_quality * EvaluationWeights.CONTENT_QUALITY +
            engagement * EvaluationWeights.STUDENT_ENGAGEMENT +
            outcomes * EvaluationWeights.LEARNING_OUTCOMES
        )
    
    def score_to_rating(self, score: float) -> float:
        """Convert 0-100 score to 0-5 star rating."""
        return (score / 100.0) * 5.0
    
    def get_performance_category(self, score: float) -> str:
        """Categorize performance based on score."""
        if score >= 90:
            return "Exceptional"
        elif score >= 75:
            return "Very Good"
        elif score >= 60:
            return "Good"
        elif score >= 45:
            return "Satisfactory"
        else:
            return "Needs Improvement"
