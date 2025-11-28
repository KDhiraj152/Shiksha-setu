"""
A/B Testing Framework

Issue: CODE-REVIEW-GPT #14 (HIGH)
Problem: No A/B testing for content variations

Solution: Framework for content experiments with user segmentation and metrics
"""

from typing import Dict, Any, List, Optional
from datetime import datetime, timezone, timedelta
from enum import Enum
import logging
import hashlib
import random

from sqlalchemy.orm import Session
from sqlalchemy import Column, String, Integer, Float, Boolean, DateTime, JSON, ForeignKey, Enum as SQLEnum
from sqlalchemy.ext.declarative import declarative_base

from ..database import Base
from ..models import User, ProcessedContent

logger = logging.getLogger(__name__)


class ExperimentStatus(str, Enum):
    """Status of A/B test experiment."""
    DRAFT = "draft"
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    ARCHIVED = "archived"


class VariantType(str, Enum):
    """Type of variant in experiment."""
    CONTROL = "control"
    TREATMENT = "treatment"


# Database Models for A/B Testing
class Experiment(Base):
    """A/B test experiment."""
    __tablename__ = "experiments"
    
    id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    description = Column(String)
    hypothesis = Column(String)
    status = Column(SQLEnum(ExperimentStatus), default=ExperimentStatus.DRAFT)
    
    # Experiment configuration
    traffic_allocation = Column(Float, default=1.0)  # 0.0 to 1.0
    start_date = Column(DateTime(timezone=True))
    end_date = Column(DateTime(timezone=True))
    
    # Target criteria
    target_grades = Column(JSON)  # List of grade levels
    target_subjects = Column(JSON)  # List of subjects
    target_languages = Column(JSON)  # List of languages
    
    # Metrics to track
    primary_metric = Column(String)  # e.g., "completion_rate"
    secondary_metrics = Column(JSON)  # List of secondary metrics
    
    # Results
    winner_variant_id = Column(String, nullable=True)
    statistical_significance = Column(Float, nullable=True)
    
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)
    created_by = Column(String, ForeignKey("users.id"))


class ExperimentVariant(Base):
    """Variant in an A/B test."""
    __tablename__ = "experiment_variants"
    
    id = Column(String, primary_key=True)
    experiment_id = Column(String, ForeignKey("experiments.id", ondelete="CASCADE"))
    name = Column(String, nullable=False)
    variant_type = Column(SQLEnum(VariantType))
    
    # Content configuration
    content_id = Column(String, ForeignKey("processed_content.id"), nullable=True)
    configuration = Column(JSON)  # Variant-specific config
    
    # Traffic split
    traffic_percentage = Column(Float, default=50.0)  # Percentage of traffic
    
    # Metrics
    impressions = Column(Integer, default=0)
    conversions = Column(Integer, default=0)
    total_time_spent = Column(Float, default=0.0)
    
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)


class ExperimentAssignment(Base):
    """User assignment to experiment variant."""
    __tablename__ = "experiment_assignments"
    
    id = Column(String, primary_key=True)
    experiment_id = Column(String, ForeignKey("experiments.id", ondelete="CASCADE"))
    variant_id = Column(String, ForeignKey("experiment_variants.id", ondelete="CASCADE"))
    user_id = Column(String, ForeignKey("users.id"), nullable=True)
    session_id = Column(String)  # For anonymous users
    
    assigned_at = Column(DateTime(timezone=True), default=datetime.utcnow)


class ExperimentEvent(Base):
    """Events tracked during experiment."""
    __tablename__ = "experiment_events"
    
    id = Column(String, primary_key=True)
    experiment_id = Column(String, ForeignKey("experiments.id", ondelete="CASCADE"))
    variant_id = Column(String, ForeignKey("experiment_variants.id", ondelete="CASCADE"))
    user_id = Column(String, ForeignKey("users.id"), nullable=True)
    session_id = Column(String)
    
    event_type = Column(String)  # impression, click, conversion, etc.
    event_data = Column(JSON)
    
    timestamp = Column(DateTime(timezone=True), default=datetime.utcnow)


class ABTestingService:
    """Service for A/B testing experiments."""
    
    def __init__(self, db: Session):
        self.db = db
    
    def create_experiment(
        self,
        name: str,
        description: str,
        hypothesis: str,
        variants: List[Dict[str, Any]],
        target_criteria: Dict[str, Any],
        primary_metric: str,
        secondary_metrics: List[str],
        traffic_allocation: float = 1.0,
        created_by: str = None
    ) -> Experiment:
        """
        Create a new A/B test experiment.
        
        Args:
            name: Experiment name
            description: Description
            hypothesis: Hypothesis being tested
            variants: List of variant configurations
            target_criteria: Targeting criteria (grades, subjects, languages)
            primary_metric: Primary success metric
            secondary_metrics: Secondary metrics to track
            traffic_allocation: Percentage of traffic to include (0.0-1.0)
            created_by: User ID who created experiment
            
        Returns:
            Created Experiment object
        """
        import uuid
        
        experiment = Experiment(
            id=str(uuid.uuid4()),
            name=name,
            description=description,
            hypothesis=hypothesis,
            traffic_allocation=traffic_allocation,
            target_grades=target_criteria.get("grades", []),
            target_subjects=target_criteria.get("subjects", []),
            target_languages=target_criteria.get("languages", []),
            primary_metric=primary_metric,
            secondary_metrics=secondary_metrics,
            created_by=created_by
        )
        
        self.db.add(experiment)
        self.db.flush()
        
        # Create variants
        for variant_config in variants:
            self._create_variant(experiment.id, variant_config)
        
        self.db.commit()
        logger.info(f"Created experiment: {name} (ID: {experiment.id})")
        
        return experiment
    
    def _create_variant(
        self,
        experiment_id: str,
        config: Dict[str, Any]
    ) -> ExperimentVariant:
        """Create a variant for an experiment."""
        import uuid
        
        variant = ExperimentVariant(
            id=str(uuid.uuid4()),
            experiment_id=experiment_id,
            name=config.get("name"),
            variant_type=VariantType(config.get("type", "treatment")),
            content_id=config.get("content_id"),
            configuration=config.get("configuration", {}),
            traffic_percentage=config.get("traffic_percentage", 50.0)
        )
        
        self.db.add(variant)
        return variant
    
    def start_experiment(self, experiment_id: str) -> Experiment:
        """Start an experiment."""
        experiment = self.db.query(Experiment).filter(
            Experiment.id == experiment_id
        ).first()
        
        if not experiment:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        experiment.status = ExperimentStatus.ACTIVE
        experiment.start_date = datetime.now(timezone.utc)
        
        self.db.commit()
        logger.info(f"Started experiment: {experiment.name}")
        
        return experiment
    
    def stop_experiment(self, experiment_id: str) -> Experiment:
        """Stop an experiment."""
        experiment = self.db.query(Experiment).filter(
            Experiment.id == experiment_id
        ).first()
        
        if not experiment:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        experiment.status = ExperimentStatus.COMPLETED
        experiment.end_date = datetime.now(timezone.utc)
        
        # Calculate winner
        self._calculate_winner(experiment)
        
        self.db.commit()
        logger.info(f"Stopped experiment: {experiment.name}")
        
        return experiment
    
    def assign_variant(
        self,
        experiment_id: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> ExperimentVariant:
        """
        Assign user/session to a variant.
        
        Uses consistent hashing for stable assignments.
        
        Args:
            experiment_id: Experiment ID
            user_id: User ID (if authenticated)
            session_id: Session ID (for anonymous users)
            context: Request context (grade, subject, language)
            
        Returns:
            Assigned variant
        """
        experiment = self.db.query(Experiment).filter(
            Experiment.id == experiment_id,
            Experiment.status == ExperimentStatus.ACTIVE
        ).first()
        
        if not experiment:
            raise ValueError(f"Active experiment {experiment_id} not found")
        
        # Check if already assigned
        existing = self.db.query(ExperimentAssignment).filter(
            ExperimentAssignment.experiment_id == experiment_id,
            ExperimentAssignment.user_id == user_id if user_id else
            ExperimentAssignment.session_id == session_id
        ).first()
        
        if existing:
            variant = self.db.query(ExperimentVariant).filter(
                ExperimentVariant.id == existing.variant_id
            ).first()
            return variant
        
        # Check targeting criteria
        if context and not self._matches_targeting(experiment, context):
            # Return control variant if doesn't match targeting
            control = self.db.query(ExperimentVariant).filter(
                ExperimentVariant.experiment_id == experiment_id,
                ExperimentVariant.variant_type == VariantType.CONTROL
            ).first()
            return control
        
        # Check traffic allocation
        identifier = user_id or session_id
        if not self._in_traffic_allocation(identifier, experiment.traffic_allocation):
            # Not in traffic, return control
            control = self.db.query(ExperimentVariant).filter(
                ExperimentVariant.experiment_id == experiment_id,
                ExperimentVariant.variant_type == VariantType.CONTROL
            ).first()
            return control
        
        # Assign to variant using consistent hashing
        variants = self.db.query(ExperimentVariant).filter(
            ExperimentVariant.experiment_id == experiment_id
        ).all()
        
        variant = self._hash_to_variant(identifier, variants)
        
        # Record assignment
        import uuid
        assignment = ExperimentAssignment(
            id=str(uuid.uuid4()),
            experiment_id=experiment_id,
            variant_id=variant.id,
            user_id=user_id,
            session_id=session_id
        )
        
        self.db.add(assignment)
        self.db.commit()
        
        logger.info(f"Assigned {identifier} to variant {variant.name} in experiment {experiment.name}")
        
        return variant
    
    def track_event(
        self,
        experiment_id: str,
        variant_id: str,
        event_type: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        event_data: Optional[Dict[str, Any]] = None
    ):
        """Track an event in the experiment."""
        import uuid
        
        event = ExperimentEvent(
            id=str(uuid.uuid4()),
            experiment_id=experiment_id,
            variant_id=variant_id,
            user_id=user_id,
            session_id=session_id,
            event_type=event_type,
            event_data=event_data or {}
        )
        
        self.db.add(event)
        
        # Update variant metrics
        variant = self.db.query(ExperimentVariant).filter(
            ExperimentVariant.id == variant_id
        ).first()
        
        if event_type == "impression":
            variant.impressions += 1
        elif event_type == "conversion":
            variant.conversions += 1
        elif event_type == "time_spent" and event_data:
            variant.total_time_spent += event_data.get("duration", 0)
        
        self.db.commit()
    
    def get_experiment_results(self, experiment_id: str) -> Dict[str, Any]:
        """Get results for an experiment."""
        experiment = self.db.query(Experiment).filter(
            Experiment.id == experiment_id
        ).first()
        
        if not experiment:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        variants = self.db.query(ExperimentVariant).filter(
            ExperimentVariant.experiment_id == experiment_id
        ).all()
        
        results = {
            "experiment": {
                "id": experiment.id,
                "name": experiment.name,
                "status": experiment.status,
                "start_date": experiment.start_date.isoformat() if experiment.start_date else None,
                "end_date": experiment.end_date.isoformat() if experiment.end_date else None,
                "winner": experiment.winner_variant_id
            },
            "variants": []
        }
        
        for variant in variants:
            conversion_rate = (variant.conversions / variant.impressions * 100) if variant.impressions > 0 else 0
            avg_time = (variant.total_time_spent / variant.impressions) if variant.impressions > 0 else 0
            
            results["variants"].append({
                "id": variant.id,
                "name": variant.name,
                "type": variant.variant_type,
                "impressions": variant.impressions,
                "conversions": variant.conversions,
                "conversion_rate": round(conversion_rate, 2),
                "avg_time_spent": round(avg_time, 2)
            })
        
        return results
    
    def _matches_targeting(
        self,
        experiment: Experiment,
        context: Dict[str, Any]
    ) -> bool:
        """Check if context matches experiment targeting."""
        # Check grade level
        if experiment.target_grades and context.get("grade_level"):
            if context["grade_level"] not in experiment.target_grades:
                return False
        
        # Check subject
        if experiment.target_subjects and context.get("subject"):
            if context["subject"] not in experiment.target_subjects:
                return False
        
        # Check language
        if experiment.target_languages and context.get("language"):
            if context["language"] not in experiment.target_languages:
                return False
        
        return True
    
    def _in_traffic_allocation(
        self,
        identifier: str,
        allocation: float
    ) -> bool:
        """Check if identifier is in traffic allocation."""
        hash_val = int(hashlib.md5(identifier.encode()).hexdigest(), 16)
        return (hash_val % 100) < (allocation * 100)
    
    def _hash_to_variant(
        self,
        identifier: str,
        variants: List[ExperimentVariant]
    ) -> ExperimentVariant:
        """Hash identifier to variant using traffic percentages."""
        hash_val = int(hashlib.md5(identifier.encode()).hexdigest(), 16) % 100
        
        cumulative = 0
        for variant in variants:
            cumulative += variant.traffic_percentage
            if hash_val < cumulative:
                return variant
        
        # Fallback to first variant
        return variants[0]
    
    def _calculate_winner(self, experiment: Experiment):
        """Calculate winning variant (simple implementation)."""
        variants = self.db.query(ExperimentVariant).filter(
            ExperimentVariant.experiment_id == experiment.id
        ).all()
        
        # Simple winner: highest conversion rate with minimum impressions
        min_impressions = 100
        best_variant = None
        best_rate = 0
        
        for variant in variants:
            if variant.impressions >= min_impressions:
                rate = variant.conversions / variant.impressions
                if rate > best_rate:
                    best_rate = rate
                    best_variant = variant
        
        if best_variant:
            experiment.winner_variant_id = best_variant.id
            experiment.statistical_significance = 0.95  # Placeholder
            logger.info(f"Winner: {best_variant.name} with {best_rate:.2%} conversion rate")


# Helper function for pipeline integration
def get_experiment_variant(
    db: Session,
    experiment_name: str,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    context: Optional[Dict[str, Any]] = None
) -> Optional[ExperimentVariant]:
    """
    Get variant for active experiment.
    
    Args:
        db: Database session
        experiment_name: Name of experiment
        user_id: User ID
        session_id: Session ID
        context: Request context
        
    Returns:
        Assigned variant or None if no active experiment
    """
    service = ABTestingService(db)
    
    # Find active experiment by name
    experiment = db.query(Experiment).filter(
        Experiment.name == experiment_name,
        Experiment.status == ExperimentStatus.ACTIVE
    ).first()
    
    if not experiment:
        return None
    
    try:
        variant = service.assign_variant(
            experiment_id=experiment.id,
            user_id=user_id,
            session_id=session_id,
            context=context
        )
        return variant
    except Exception as e:
        logger.error(f"Error assigning variant: {e}")
        return None
