"""
NCERT Curriculum Validator

Validates educational content against NCERT curriculum standards.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import logging
import numpy as np
from sqlalchemy.orm import Session

from ..models import NCERTStandard
from .standards import NCERTStandardsLoader
from ..pipeline.model_clients import BERTClient

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of content validation."""
    alignment_score: float
    errors: List[str]
    warnings: List[str]
    suggestions: List[str]
    matched_topics: List[str]
    missing_topics: List[str]
    terminology_issues: List[str]


class NCERTValidator:
    """Validates content against NCERT curriculum standards."""
    
    def __init__(self, bert_client: Optional[BERTClient] = None):
        """
        Initialize NCERT validator.
        
        Args:
            bert_client: Optional BERT client for embeddings
        """
        self.bert_client = bert_client or BERTClient()
        self.standards_loader = NCERTStandardsLoader(self.bert_client)
        self.alignment_threshold = 0.70
        
    async def validate_content(
        self,
        text: str,
        grade_level: int,
        subject: str,
        standards: List[NCERTStandard]
    ) -> Dict[str, Any]:
        """
        Validate content against NCERT curriculum standards.
        
        Args:
            text: Content text to validate
            grade_level: Grade level (1-12)
            subject: Subject area
            standards: List of NCERT standards to validate against
            
        Returns:
            Dictionary with validation results
        """
        logger.info(f"Validating content for Grade {grade_level}, {subject}")
        
        errors = []
        warnings = []
        suggestions = []
        matched_topics = []
        missing_topics = []
        terminology_issues = []
        
        # Find matching standards using embeddings
        matching_standards = self.standards_loader.find_matching_standards(
            content=text,
            grade_level=grade_level,
            subject=subject,
            top_k=5
        )
        
        if not matching_standards:
            errors.append(f"No matching curriculum standards found for Grade {grade_level}, {subject}")
            return {
                "alignment_score": 0.0,
                "errors": errors,
                "warnings": warnings,
                "suggestions": suggestions,
                "matched_topics": matched_topics,
                "missing_topics": missing_topics,
                "terminology_issues": terminology_issues
            }
        
        # Calculate overall alignment score
        alignment_scores = []
        for standard_data, similarity in matching_standards:
            alignment_scores.append(similarity)
            
            if similarity >= self.alignment_threshold:
                matched_topics.append(standard_data.topic)
            else:
                missing_topics.append(standard_data.topic)
        
        overall_alignment = np.mean(alignment_scores) if alignment_scores else 0.0
        
        # Check keyword overlap
        for standard_data, _ in matching_standards:
            keyword_overlap = self.standards_loader.check_keyword_overlap(text, standard_data)
            if keyword_overlap < 0.3:
                warnings.append(
                    f"Low keyword overlap ({keyword_overlap:.1%}) for topic: {standard_data.topic}"
                )
                suggestions.append(
                    f"Consider adding keywords: {', '.join(standard_data.keywords[:3])}"
                )
        
        # Check learning objectives match
        for standard_data, _ in matching_standards[:3]:  # Check top 3 matches
            objectives_match = self.standards_loader.get_learning_objectives_match(
                text, standard_data
            )
            if objectives_match < 0.5:
                warnings.append(
                    f"Content may not fully address learning objectives for: {standard_data.topic}"
                )
        
        # Provide suggestions based on alignment score
        if overall_alignment < 0.5:
            errors.append(
                f"Content alignment too low ({overall_alignment:.1%}). "
                f"Content may not be appropriate for Grade {grade_level} {subject}."
            )
            suggestions.append(
                "Review NCERT curriculum guidelines and adjust content to better match "
                "expected learning outcomes."
            )
        elif overall_alignment < self.alignment_threshold:
            warnings.append(
                f"Content alignment below threshold ({overall_alignment:.1%}). "
                "Consider improvements."
            )
        
        return {
            "alignment_score": float(overall_alignment),
            "errors": errors,
            "warnings": warnings,
            "suggestions": suggestions,
            "matched_topics": matched_topics,
            "missing_topics": missing_topics,
            "terminology_issues": terminology_issues
        }
    
    async def check_factual_accuracy(
        self,
        text: str,
        subject: str
    ) -> Dict[str, Any]:
        """
        Check factual accuracy of content.
        
        Args:
            text: Content text
            subject: Subject area
            
        Returns:
            Dictionary with accuracy results
        """
        logger.info(f"Checking factual accuracy for {subject}")
        
        # This is a simplified implementation
        # In production, you would use:
        # - Fact-checking APIs
        # - Knowledge graph validation
        # - Subject-specific fact databases
        
        factual_errors = []
        accuracy_score = 0.85  # Placeholder score
        
        # Basic checks (expand based on requirements)
        if not text or len(text.strip()) < 10:
            factual_errors.append("Content too short for accuracy validation")
            accuracy_score = 0.0
        
        return {
            "accuracy_score": accuracy_score,
            "factual_errors": factual_errors,
            "confidence": 0.8
        }
    
    def validate_terminology(
        self,
        text: str,
        subject: str,
        grade_level: int
    ) -> List[str]:
        """
        Validate terminology appropriateness for grade level.
        
        Args:
            text: Content text
            subject: Subject area
            grade_level: Grade level
            
        Returns:
            List of terminology issues
        """
        issues = []
        
        # Subject-specific terminology checks
        if subject.lower() == "mathematics":
            complex_terms = ["calculus", "derivative", "integral"]
            if grade_level < 11:
                for term in complex_terms:
                    if term.lower() in text.lower():
                        issues.append(
                            f"Advanced term '{term}' may be inappropriate for Grade {grade_level}"
                        )
        
        elif subject.lower() == "science":
            complex_terms = ["quantum", "thermodynamics", "electromagnetism"]
            if grade_level < 9:
                for term in complex_terms:
                    if term.lower() in text.lower():
                        issues.append(
                            f"Advanced term '{term}' may be inappropriate for Grade {grade_level}"
                        )
        
        return issues
