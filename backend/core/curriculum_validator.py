"""
NCERT/CBSE Curriculum Validation System.

This module validates educational content against NCERT and CBSE curriculum standards,
ensuring alignment with learning objectives and factual accuracy.
"""
import json
import re
import logging
from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass
from pathlib import Path
from enum import Enum

logger = logging.getLogger(__name__)


class ValidationStatus(Enum):
    """Content validation status."""
    APPROVED = "approved"
    NEEDS_REVIEW = "needs_review"
    REJECTED = "rejected"
    PARTIALLY_ALIGNED = "partially_aligned"


class Board(Enum):
    """Educational boards."""
    NCERT = "ncert"
    CBSE = "cbse"
    ICSE = "icse"
    STATE = "state"


@dataclass
class CurriculumStandard:
    """Represents a curriculum standard."""
    grade_level: int
    subject: str
    topic: str
    learning_objectives: List[str]
    keywords: List[str]
    board: Board
    chapter: Optional[str] = None
    unit: Optional[str] = None


@dataclass
class ValidationResult:
    """Result of curriculum validation."""
    status: ValidationStatus
    alignment_score: float  # 0-100
    matched_objectives: List[str]
    missing_objectives: List[str]
    keyword_coverage: float  # 0-100
    matched_keywords: List[str]
    recommendations: List[str]
    factual_issues: List[str]
    metadata: Dict[str, Any]


class CurriculumDatabase:
    """
    Database of curriculum standards from NCERT, CBSE, and state boards.
    """
    
    def __init__(self, data_dir: str = "data/curriculum"):
        """
        Initialize curriculum database.
        
        Args:
            data_dir: Directory containing curriculum JSON files
        """
        self.data_dir = Path(data_dir)
        self.standards: Dict[str, List[CurriculumStandard]] = {}
        self._load_standards()
        logger.info(f"Curriculum database loaded with {len(self.standards)} subjects")
    
    def _load_standards(self):
        """Load curriculum standards from JSON files."""
        # Load NCERT standards
        ncert_file = self.data_dir / "ncert_standards_sample.json"
        if ncert_file.exists():
            with open(ncert_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for standard in data.get('standards', []):
                    self._add_standard(standard, Board.NCERT)
        
        # Load CBSE standards (if available)
        cbse_file = self.data_dir / "cbse_standards.json"
        if cbse_file.exists():
            with open(cbse_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for standard in data.get('standards', []):
                    self._add_standard(standard, Board.CBSE)
        
        # Load state board standards (if available)
        state_dir = self.data_dir / "state_boards"
        if state_dir.exists():
            for state_file in state_dir.glob("*.json"):
                with open(state_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for standard in data.get('standards', []):
                        self._add_standard(standard, Board.STATE)
    
    def _add_standard(self, data: Dict[str, Any], board: Board):
        """Add a curriculum standard to the database."""
        standard = CurriculumStandard(
            grade_level=data['grade_level'],
            subject=data['subject'],
            topic=data['topic'],
            learning_objectives=data['learning_objectives'],
            keywords=data['keywords'],
            board=board,
            chapter=data.get('chapter'),
            unit=data.get('unit')
        )
        
        key = f"{standard.subject}_{standard.grade_level}"
        if key not in self.standards:
            self.standards[key] = []
        self.standards[key].append(standard)
    
    def get_standards(
        self,
        subject: str,
        grade_level: int,
        board: Optional[Board] = None
    ) -> List[CurriculumStandard]:
        """
        Get curriculum standards for a subject and grade level.
        
        Args:
            subject: Subject name
            grade_level: Grade level (1-12)
            board: Optional specific board filter
        
        Returns:
            List of matching curriculum standards
        """
        key = f"{subject}_{grade_level}"
        standards = self.standards.get(key, [])
        
        if board:
            standards = [s for s in standards if s.board == board]
        
        return standards
    
    def search_by_topic(
        self,
        topic: str,
        subject: Optional[str] = None,
        grade_level: Optional[int] = None
    ) -> List[CurriculumStandard]:
        """
        Search standards by topic keyword.
        
        Args:
            topic: Topic to search for
            subject: Optional subject filter
            grade_level: Optional grade level filter
        
        Returns:
            List of matching standards
        """
        results = []
        topic_lower = topic.lower()
        
        for standards_list in self.standards.values():
            for standard in standards_list:
                # Apply filters
                if subject and standard.subject != subject:
                    continue
                if grade_level and standard.grade_level != grade_level:
                    continue
                
                # Check if topic matches
                if (topic_lower in standard.topic.lower() or
                    any(topic_lower in keyword.lower() for keyword in standard.keywords)):
                    results.append(standard)
        
        return results


class CurriculumValidator:
    """
    Validates educational content against curriculum standards.
    
    Features:
    - NCERT/CBSE/State board alignment checking
    - Learning objective coverage analysis
    - Keyword matching and relevance scoring
    - Factual accuracy verification
    - Recommendations for improvement
    """
    
    # Minimum scores for approval
    APPROVAL_THRESHOLDS = {
        'alignment_score': 80.0,
        'keyword_coverage': 70.0,
        'objective_coverage': 75.0
    }
    
    # Factual accuracy patterns (common errors to detect)
    FACTUAL_ERROR_PATTERNS = {
        'Mathematics': [
            (r'2\s*\+\s*2\s*=\s*5', 'Incorrect basic arithmetic'),
            (r'π\s*=\s*3\.14159', 'Approximation should be noted as ≈'),
        ],
        'Science': [
            (r'plants?\s+breathe', 'Plants respire, not breathe'),
            (r'sun\s+revolves?\s+around\s+earth', 'Earth revolves around sun'),
            (r'human\s+have\s+5\s+sense', 'Humans have more than 5 senses'),
        ],
        'History': [
            (r'columbus\s+discovered\s+america', 'Indigenous peoples already inhabited America'),
        ]
    }
    
    def __init__(self, curriculum_db: Optional[CurriculumDatabase] = None):
        """
        Initialize the curriculum validator.
        
        Args:
            curriculum_db: Optional curriculum database instance
        """
        self.curriculum_db = curriculum_db or CurriculumDatabase()
        logger.info("CurriculumValidator initialized")
    
    def validate_content(
        self,
        content: str,
        subject: str,
        grade_level: int,
        topic: Optional[str] = None,
        board: Board = Board.NCERT
    ) -> ValidationResult:
        """
        Validate content against curriculum standards.
        
        Args:
            content: Educational content to validate
            subject: Subject area
            grade_level: Grade level (1-12)
            topic: Optional specific topic
            board: Educational board (NCERT, CBSE, etc.)
        
        Returns:
            ValidationResult with alignment scores and recommendations
        
        Raises:
            ValueError: If inputs are invalid
        """
        # Validate inputs
        if not content or len(content.strip()) == 0:
            raise ValueError("Content cannot be empty")
        
        if not (1 <= grade_level <= 12):
            raise ValueError("Grade level must be between 1 and 12")
        
        logger.info(
            f"Validating {subject} content for grade {grade_level}, board: {board.value}"
        )
        
        # Get relevant curriculum standards
        standards = self.curriculum_db.get_standards(subject, grade_level, board)
        
        if not standards:
            logger.warning(f"No standards found for {subject}, grade {grade_level}")
            return ValidationResult(
                status=ValidationStatus.NEEDS_REVIEW,
                alignment_score=0.0,
                matched_objectives=[],
                missing_objectives=[],
                keyword_coverage=0.0,
                matched_keywords=[],
                recommendations=["No curriculum standards available for validation"],
                factual_issues=[],
                metadata={'standards_available': False}
            )
        
        # Filter by topic if provided
        if topic:
            standards = [s for s in standards if topic.lower() in s.topic.lower()]
        
        # Perform validation
        alignment_score, matched_objectives, missing_objectives = \
            self._check_learning_objectives(content, standards)
        
        keyword_coverage, matched_keywords = \
            self._check_keyword_coverage(content, standards)
        
        factual_issues = self._check_factual_accuracy(content, subject)
        
        recommendations = self._generate_recommendations(
            alignment_score,
            keyword_coverage,
            missing_objectives,
            factual_issues
        )
        
        # Determine validation status
        status = self._determine_status(
            alignment_score,
            keyword_coverage,
            len(factual_issues)
        )
        
        logger.info(
            f"Validation complete: status={status.value}, "
            f"alignment={alignment_score:.1f}%, keywords={keyword_coverage:.1f}%"
        )
        
        return ValidationResult(
            status=status,
            alignment_score=alignment_score,
            matched_objectives=matched_objectives,
            missing_objectives=missing_objectives,
            keyword_coverage=keyword_coverage,
            matched_keywords=matched_keywords,
            recommendations=recommendations,
            factual_issues=factual_issues,
            metadata={
                'subject': subject,
                'grade_level': grade_level,
                'board': board.value,
                'standards_checked': len(standards),
                'content_length': len(content)
            }
        )
    
    def _check_learning_objectives(
        self,
        content: str,
        standards: List[CurriculumStandard]
    ) -> Tuple[float, List[str], List[str]]:
        """
        Check if content covers learning objectives.
        
        Args:
            content: Content to check
            standards: List of curriculum standards
        
        Returns:
            Tuple of (alignment_score, matched_objectives, missing_objectives)
        """
        content_lower = content.lower()
        
        all_objectives = []
        matched_objectives = []
        
        for standard in standards:
            for objective in standard.learning_objectives:
                all_objectives.append(objective)
                
                # Check if objective is addressed in content
                # Extract key terms from objective
                key_terms = self._extract_key_terms(objective)
                
                # Check if majority of key terms are in content
                matches = sum(1 for term in key_terms if term.lower() in content_lower)
                coverage = matches / len(key_terms) if key_terms else 0
                
                if coverage >= 0.6:  # 60% of key terms must be present
                    matched_objectives.append(objective)
        
        # Calculate alignment score
        if all_objectives:
            alignment_score = (len(matched_objectives) / len(all_objectives)) * 100
        else:
            alignment_score = 0.0
        
        missing_objectives = [obj for obj in all_objectives if obj not in matched_objectives]
        
        return alignment_score, matched_objectives, missing_objectives
    
    def _check_keyword_coverage(
        self,
        content: str,
        standards: List[CurriculumStandard]
    ) -> Tuple[float, List[str]]:
        """
        Check keyword coverage in content.
        
        Args:
            content: Content to check
            standards: List of curriculum standards
        
        Returns:
            Tuple of (coverage_percentage, matched_keywords)
        """
        content_lower = content.lower()
        
        # Collect all keywords from standards
        all_keywords: Set[str] = set()
        for standard in standards:
            all_keywords.update(keyword.lower() for keyword in standard.keywords)
        
        # Check which keywords are present
        matched_keywords = [
            keyword for keyword in all_keywords
            if keyword in content_lower
        ]
        
        # Calculate coverage
        if all_keywords:
            coverage = (len(matched_keywords) / len(all_keywords)) * 100
        else:
            coverage = 0.0
        
        return coverage, matched_keywords
    
    def _check_factual_accuracy(
        self,
        content: str,
        subject: str
    ) -> List[str]:
        """
        Check for common factual errors.
        
        Args:
            content: Content to check
            subject: Subject area
        
        Returns:
            List of detected factual issues
        """
        issues = []
        
        # Get patterns for this subject
        patterns = self.FACTUAL_ERROR_PATTERNS.get(subject, [])
        
        for pattern, issue_description in patterns:
            if re.search(pattern, content, re.IGNORECASE):
                issues.append(issue_description)
        
        # Check for common spelling errors in technical terms
        technical_terms_errors = {
            'photosinthesis': 'photosynthesis',
            'mitocondria': 'mitochondria',
            'chromosone': 'chromosome',
            'newtons': 'Newton\'s',
            'pythagorean': 'Pythagorean',
        }
        
        content_lower = content.lower()
        for error, correct in technical_terms_errors.items():
            if error in content_lower:
                issues.append(f"Spelling error: '{error}' should be '{correct}'")
        
        return issues
    
    def _generate_recommendations(
        self,
        alignment_score: float,
        keyword_coverage: float,
        missing_objectives: List[str],
        factual_issues: List[str]
    ) -> List[str]:
        """
        Generate recommendations for content improvement.
        
        Args:
            alignment_score: Learning objective alignment score
            keyword_coverage: Keyword coverage percentage
            missing_objectives: List of objectives not covered
            factual_issues: List of factual errors
        
        Returns:
            List of recommendation strings
        """
        recommendations = []
        
        # Alignment recommendations
        if alignment_score < self.APPROVAL_THRESHOLDS['alignment_score']:
            recommendations.append(
                f"Alignment score ({alignment_score:.1f}%) is below threshold "
                f"({self.APPROVAL_THRESHOLDS['alignment_score']:.1f}%). "
                f"Consider addressing more learning objectives."
            )
            
            if missing_objectives:
                recommendations.append(
                    f"Missing {len(missing_objectives)} learning objective(s). "
                    f"Consider covering: {', '.join(missing_objectives[:3])}"
                )
        
        # Keyword recommendations
        if keyword_coverage < self.APPROVAL_THRESHOLDS['keyword_coverage']:
            recommendations.append(
                f"Keyword coverage ({keyword_coverage:.1f}%) is low. "
                f"Include more curriculum-relevant terminology."
            )
        
        # Factual accuracy recommendations
        if factual_issues:
            recommendations.append(
                f"Found {len(factual_issues)} potential factual issue(s). "
                f"Please review and correct."
            )
            for issue in factual_issues:
                recommendations.append(f"  - {issue}")
        
        # Positive feedback
        if not recommendations:
            recommendations.append(
                "Content meets all curriculum standards. Excellent alignment!"
            )
        
        return recommendations
    
    def _determine_status(
        self,
        alignment_score: float,
        keyword_coverage: float,
        factual_issues_count: int
    ) -> ValidationStatus:
        """
        Determine overall validation status.
        
        Args:
            alignment_score: Learning objective alignment score
            keyword_coverage: Keyword coverage percentage
            factual_issues_count: Number of factual issues found
        
        Returns:
            ValidationStatus enum value
        """
        # Reject if factual errors found
        if factual_issues_count > 0:
            return ValidationStatus.NEEDS_REVIEW
        
        # Check if meets approval thresholds
        meets_alignment = alignment_score >= self.APPROVAL_THRESHOLDS['alignment_score']
        meets_keywords = keyword_coverage >= self.APPROVAL_THRESHOLDS['keyword_coverage']
        
        if meets_alignment and meets_keywords:
            return ValidationStatus.APPROVED
        elif alignment_score >= 50 or keyword_coverage >= 50:
            return ValidationStatus.PARTIALLY_ALIGNED
        else:
            return ValidationStatus.NEEDS_REVIEW
    
    def _extract_key_terms(self, text: str) -> List[str]:
        """
        Extract key terms from text for matching.
        
        Args:
            text: Text to extract terms from
        
        Returns:
            List of key terms
        """
        # Remove common words and extract meaningful terms
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'as', 'is', 'are', 'was', 'were', 'be',
            'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'can',
            'could', 'will', 'would', 'should', 'may', 'might', 'must', 'shall'
        }
        
        # Extract words
        words = re.findall(r'\b\w+\b', text.lower())
        
        # Filter out stop words and short words
        key_terms = [
            word for word in words
            if word not in stop_words and len(word) > 3
        ]
        
        return key_terms
    
    def batch_validate(
        self,
        contents: List[Dict[str, Any]],
        subject: str,
        grade_level: int,
        board: Board = Board.NCERT
    ) -> List[ValidationResult]:
        """
        Validate multiple content items in batch.
        
        Args:
            contents: List of content dictionaries with 'text' and optional 'topic'
            subject: Subject area
            grade_level: Grade level
            board: Educational board
        
        Returns:
            List of ValidationResult objects
        """
        results = []
        
        for i, content_item in enumerate(contents):
            try:
                result = self.validate_content(
                    content=content_item['text'],
                    subject=subject,
                    grade_level=grade_level,
                    topic=content_item.get('topic'),
                    board=board
                )
                results.append(result)
                
                logger.info(f"Batch validation {i+1}/{len(contents)}: {result.status.value}")
                
            except Exception as e:
                logger.error(f"Batch validation failed for item {i+1}: {e}")
                # Add failed result
                results.append(ValidationResult(
                    status=ValidationStatus.REJECTED,
                    alignment_score=0.0,
                    matched_objectives=[],
                    missing_objectives=[],
                    keyword_coverage=0.0,
                    matched_keywords=[],
                    recommendations=[f"Validation error: {str(e)}"],
                    factual_issues=[],
                    metadata={'error': str(e)}
                ))
        
        return results


# Convenience functions
def validate_ncert_content(
    content: str,
    subject: str,
    grade_level: int,
    topic: Optional[str] = None
) -> ValidationResult:
    """
    Quick validation against NCERT standards.
    
    Args:
        content: Content to validate
        subject: Subject area
        grade_level: Grade level (1-12)
        topic: Optional topic filter
    
    Returns:
        ValidationResult
    """
    validator = CurriculumValidator()
    return validator.validate_content(
        content=content,
        subject=subject,
        grade_level=grade_level,
        topic=topic,
        board=Board.NCERT
    )


if __name__ == "__main__":
    # Example usage
    sample_content = """
    Photosynthesis is the process by which plants make their food using sunlight, 
    water, and carbon dioxide. The green pigment chlorophyll in leaves captures 
    sunlight energy. This energy is used to convert water and carbon dioxide into 
    glucose (sugar) and oxygen. The equation is: 6CO₂ + 6H₂O + light → C₆H₁₂O₆ + 6O₂.
    
    This process takes place in chloroplasts, which are organelles found in plant cells.
    Photosynthesis is essential for life on Earth as it produces oxygen and food.
    """
    
    validator = CurriculumValidator()
    
    result = validator.validate_content(
        content=sample_content,
        subject="Science",
        grade_level=6,
        topic="Plants",
        board=Board.NCERT
    )
    
    print(f"Validation Status: {result.status.value}")
    print(f"Alignment Score: {result.alignment_score:.1f}%")
    print(f"Keyword Coverage: {result.keyword_coverage:.1f}%")
    print(f"\nMatched Objectives ({len(result.matched_objectives)}):")
    for obj in result.matched_objectives:
        print(f"  ✓ {obj}")
    
    if result.missing_objectives:
        print(f"\nMissing Objectives ({len(result.missing_objectives)}):")
        for obj in result.missing_objectives:
            print(f"  ✗ {obj}")
    
    print("\nRecommendations:")
    for rec in result.recommendations:
        print(f"  - {rec}")
