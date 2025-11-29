"""
NCERT Curriculum Knowledge Graph - Pedagogical Validation Engine

A graph-based curriculum model that validates content against:
- Concept prerequisite chains
- Bloom's taxonomy cognitive levels per grade
- Vocabulary constraints by grade
- Regional board alignment (CBSE, State boards)

Architecture:
    ┌────────────────────────────────────────────────────────────┐
    │                    Knowledge Graph                         │
    │                                                            │
    │  ┌──────────┐    requires    ┌──────────┐                 │
    │  │ Concept  │───────────────▶│ Concept  │                 │
    │  │ Grade 7  │                │ Grade 6  │                 │
    │  └────┬─────┘                └──────────┘                 │
    │       │ has_level                                          │
    │       ▼                                                    │
    │  ┌──────────┐                ┌──────────┐                 │
    │  │ Bloom's  │                │ Subject  │                 │
    │  │ Level    │◀───────────────│ Domain   │                 │
    │  └──────────┘   constrains   └──────────┘                 │
    │                                                            │
    └────────────────────────────────────────────────────────────┘

Usage:
    graph = NCERTKnowledgeGraph()
    result = graph.validate_content(
        text="Photosynthesis converts light energy...",
        grade=8,
        subject="Science"
    )
    print(result.alignment_score)  # 0.85
    print(result.missing_prerequisites)  # []
    print(result.bloom_level_appropriate)  # True
"""
import logging
import re
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Dict, List, Optional, Set, Tuple, Any
from collections import defaultdict

logger = logging.getLogger(__name__)


class BloomLevel(IntEnum):
    """Bloom's Taxonomy cognitive levels."""
    REMEMBER = 1      # Recall facts, basic concepts
    UNDERSTAND = 2    # Explain ideas, concepts
    APPLY = 3         # Use info in new situations
    ANALYZE = 4       # Draw connections, organize
    EVALUATE = 5      # Justify, critique
    CREATE = 6        # Produce new or original work


class SubjectDomain(str, Enum):
    """Subject domains with different cognitive requirements."""
    MATHEMATICS = "mathematics"
    SCIENCE = "science"
    PHYSICS = "physics"
    CHEMISTRY = "chemistry"
    BIOLOGY = "biology"
    SOCIAL_STUDIES = "social_studies"
    HISTORY = "history"
    GEOGRAPHY = "geography"
    CIVICS = "civics"
    ENGLISH = "english"
    HINDI = "hindi"


class Board(str, Enum):
    """Education board standards."""
    CBSE = "cbse"
    ICSE = "icse"
    STATE_UP = "state_up"
    STATE_MAHARASHTRA = "state_maharashtra"
    STATE_TAMIL_NADU = "state_tamil_nadu"
    STATE_KARNATAKA = "state_karnataka"


@dataclass
class Concept:
    """A curriculum concept/topic."""
    id: str
    name: str
    subject: SubjectDomain
    grade: int
    chapter: Optional[str] = None
    description: Optional[str] = None
    keywords: List[str] = field(default_factory=list)
    bloom_level: BloomLevel = BloomLevel.UNDERSTAND
    prerequisite_ids: List[str] = field(default_factory=list)
    difficulty_score: float = 0.5  # 0.0-1.0


@dataclass
class GradeProfile:
    """Cognitive and vocabulary profile for a grade level."""
    grade: int
    max_sentence_length: int
    max_word_syllables: float  # Average max syllables
    max_bloom_level: BloomLevel
    vocabulary_level: int  # Word frequency rank threshold
    reading_level_flesch: Tuple[float, float]  # Min-max Flesch-Kincaid
    

@dataclass
class ValidationResult:
    """Result of curriculum validation."""
    alignment_score: float  # 0.0-1.0
    is_valid: bool
    bloom_level_detected: BloomLevel
    bloom_level_appropriate: bool
    grade_appropriate: bool
    missing_prerequisites: List[str]
    forward_references: List[str]  # References to higher-grade concepts
    vocabulary_issues: List[str]
    sentence_complexity_ok: bool
    recommendations: List[str]
    matched_concepts: List[str]
    board_alignment: Dict[Board, float]


# Grade profiles based on NCERT guidelines
GRADE_PROFILES: Dict[int, GradeProfile] = {
    5: GradeProfile(
        grade=5,
        max_sentence_length=15,
        max_word_syllables=2.5,
        max_bloom_level=BloomLevel.UNDERSTAND,
        vocabulary_level=3000,
        reading_level_flesch=(70, 90),
    ),
    6: GradeProfile(
        grade=6,
        max_sentence_length=18,
        max_word_syllables=2.8,
        max_bloom_level=BloomLevel.UNDERSTAND,
        vocabulary_level=4000,
        reading_level_flesch=(65, 85),
    ),
    7: GradeProfile(
        grade=7,
        max_sentence_length=20,
        max_word_syllables=3.0,
        max_bloom_level=BloomLevel.APPLY,
        vocabulary_level=5000,
        reading_level_flesch=(60, 80),
    ),
    8: GradeProfile(
        grade=8,
        max_sentence_length=22,
        max_word_syllables=3.2,
        max_bloom_level=BloomLevel.APPLY,
        vocabulary_level=6000,
        reading_level_flesch=(55, 75),
    ),
    9: GradeProfile(
        grade=9,
        max_sentence_length=25,
        max_word_syllables=3.5,
        max_bloom_level=BloomLevel.ANALYZE,
        vocabulary_level=8000,
        reading_level_flesch=(50, 70),
    ),
    10: GradeProfile(
        grade=10,
        max_sentence_length=28,
        max_word_syllables=3.8,
        max_bloom_level=BloomLevel.ANALYZE,
        vocabulary_level=10000,
        reading_level_flesch=(45, 65),
    ),
    11: GradeProfile(
        grade=11,
        max_sentence_length=30,
        max_word_syllables=4.0,
        max_bloom_level=BloomLevel.EVALUATE,
        vocabulary_level=12000,
        reading_level_flesch=(40, 60),
    ),
    12: GradeProfile(
        grade=12,
        max_sentence_length=35,
        max_word_syllables=4.2,
        max_bloom_level=BloomLevel.CREATE,
        vocabulary_level=15000,
        reading_level_flesch=(35, 55),
    ),
}


# Bloom's action verbs for level detection
BLOOM_VERBS: Dict[BloomLevel, List[str]] = {
    BloomLevel.REMEMBER: [
        "define", "list", "name", "recall", "identify", "recognize",
        "state", "describe", "match", "select", "label"
    ],
    BloomLevel.UNDERSTAND: [
        "explain", "summarize", "interpret", "classify", "compare",
        "discuss", "distinguish", "predict", "translate", "paraphrase"
    ],
    BloomLevel.APPLY: [
        "apply", "demonstrate", "solve", "use", "calculate",
        "implement", "execute", "illustrate", "operate", "practice"
    ],
    BloomLevel.ANALYZE: [
        "analyze", "differentiate", "organize", "attribute", "deconstruct",
        "examine", "contrast", "investigate", "categorize", "experiment"
    ],
    BloomLevel.EVALUATE: [
        "evaluate", "judge", "critique", "justify", "argue",
        "assess", "defend", "prioritize", "recommend", "support"
    ],
    BloomLevel.CREATE: [
        "create", "design", "construct", "develop", "formulate",
        "compose", "produce", "invent", "plan", "synthesize"
    ],
}


class NCERTKnowledgeGraph:
    """
    Knowledge graph for NCERT curriculum validation.
    
    Provides:
    - Concept prerequisite chain validation
    - Bloom's taxonomy level checking
    - Grade-appropriate vocabulary validation
    - Sentence complexity analysis
    - Multi-board alignment scoring
    """
    
    def __init__(self):
        self._concepts: Dict[str, Concept] = {}
        self._concept_by_subject: Dict[SubjectDomain, List[str]] = defaultdict(list)
        self._concept_by_grade: Dict[int, List[str]] = defaultdict(list)
        self._keyword_index: Dict[str, Set[str]] = defaultdict(set)  # keyword -> concept IDs
        
        # Initialize with core NCERT concepts
        self._load_core_concepts()
        
        logger.info(
            f"NCERTKnowledgeGraph initialized with {len(self._concepts)} concepts"
        )
    
    def _load_core_concepts(self) -> None:
        """Load core NCERT curriculum concepts."""
        # Science concepts (Grade 6-10)
        science_concepts = [
            # Grade 6 Science
            Concept(
                id="sci_6_food",
                name="Food: Where Does It Come From",
                subject=SubjectDomain.SCIENCE,
                grade=6,
                chapter="Chapter 1",
                keywords=["food", "plants", "animals", "herbivore", "carnivore", "omnivore"],
                bloom_level=BloomLevel.UNDERSTAND
            ),
            Concept(
                id="sci_6_plants",
                name="Getting to Know Plants",
                subject=SubjectDomain.SCIENCE,
                grade=6,
                chapter="Chapter 7",
                keywords=["herbs", "shrubs", "trees", "stem", "leaf", "root"],
                bloom_level=BloomLevel.UNDERSTAND,
                prerequisite_ids=["sci_6_food"]
            ),
            # Grade 7 Science
            Concept(
                id="sci_7_nutrition",
                name="Nutrition in Plants",
                subject=SubjectDomain.SCIENCE,
                grade=7,
                chapter="Chapter 1",
                keywords=["photosynthesis", "chlorophyll", "stomata", "nutrients", "autotrophs"],
                bloom_level=BloomLevel.UNDERSTAND,
                prerequisite_ids=["sci_6_plants"]
            ),
            Concept(
                id="sci_7_nutrition_animals",
                name="Nutrition in Animals",
                subject=SubjectDomain.SCIENCE,
                grade=7,
                chapter="Chapter 2",
                keywords=["digestion", "stomach", "intestine", "enzymes", "absorption"],
                bloom_level=BloomLevel.UNDERSTAND,
                prerequisite_ids=["sci_7_nutrition"]
            ),
            # Grade 8 Science
            Concept(
                id="sci_8_cell",
                name="Cell - Structure and Functions",
                subject=SubjectDomain.SCIENCE,
                grade=8,
                chapter="Chapter 8",
                keywords=["cell", "nucleus", "cytoplasm", "membrane", "organelles", "prokaryote", "eukaryote"],
                bloom_level=BloomLevel.APPLY,
                prerequisite_ids=["sci_7_nutrition", "sci_7_nutrition_animals"]
            ),
            Concept(
                id="sci_8_photosynthesis",
                name="Conservation of Plants and Animals",
                subject=SubjectDomain.SCIENCE,
                grade=8,
                chapter="Chapter 7",
                keywords=["biodiversity", "conservation", "deforestation", "ecosystem", "endangered"],
                bloom_level=BloomLevel.ANALYZE,
                prerequisite_ids=["sci_7_nutrition"]
            ),
            # Grade 9 Science
            Concept(
                id="sci_9_tissue",
                name="Tissues",
                subject=SubjectDomain.SCIENCE,
                grade=9,
                chapter="Chapter 6",
                keywords=["tissue", "meristematic", "permanent", "epithelial", "connective"],
                bloom_level=BloomLevel.ANALYZE,
                prerequisite_ids=["sci_8_cell"]
            ),
            # Grade 10 Science
            Concept(
                id="sci_10_life_processes",
                name="Life Processes",
                subject=SubjectDomain.SCIENCE,
                grade=10,
                chapter="Chapter 6",
                keywords=["respiration", "transportation", "excretion", "metabolism"],
                bloom_level=BloomLevel.ANALYZE,
                prerequisite_ids=["sci_9_tissue", "sci_8_cell"]
            ),
        ]
        
        # Mathematics concepts (Grade 6-10)
        math_concepts = [
            Concept(
                id="math_6_numbers",
                name="Knowing Our Numbers",
                subject=SubjectDomain.MATHEMATICS,
                grade=6,
                chapter="Chapter 1",
                keywords=["numbers", "place value", "comparison", "estimation"],
                bloom_level=BloomLevel.UNDERSTAND
            ),
            Concept(
                id="math_6_fractions",
                name="Fractions",
                subject=SubjectDomain.MATHEMATICS,
                grade=6,
                chapter="Chapter 7",
                keywords=["fraction", "numerator", "denominator", "equivalent"],
                bloom_level=BloomLevel.APPLY,
                prerequisite_ids=["math_6_numbers"]
            ),
            Concept(
                id="math_7_integers",
                name="Integers",
                subject=SubjectDomain.MATHEMATICS,
                grade=7,
                chapter="Chapter 1",
                keywords=["integers", "negative", "positive", "number line", "absolute"],
                bloom_level=BloomLevel.APPLY,
                prerequisite_ids=["math_6_numbers"]
            ),
            Concept(
                id="math_7_rational",
                name="Rational Numbers",
                subject=SubjectDomain.MATHEMATICS,
                grade=7,
                chapter="Chapter 9",
                keywords=["rational", "fraction", "decimal", "terminating", "recurring"],
                bloom_level=BloomLevel.APPLY,
                prerequisite_ids=["math_6_fractions", "math_7_integers"]
            ),
            Concept(
                id="math_8_algebraic",
                name="Algebraic Expressions and Identities",
                subject=SubjectDomain.MATHEMATICS,
                grade=8,
                chapter="Chapter 9",
                keywords=["algebra", "expression", "identity", "polynomial", "binomial"],
                bloom_level=BloomLevel.APPLY,
                prerequisite_ids=["math_7_rational"]
            ),
            Concept(
                id="math_9_polynomial",
                name="Polynomials",
                subject=SubjectDomain.MATHEMATICS,
                grade=9,
                chapter="Chapter 2",
                keywords=["polynomial", "degree", "coefficient", "zero", "factor"],
                bloom_level=BloomLevel.ANALYZE,
                prerequisite_ids=["math_8_algebraic"]
            ),
            Concept(
                id="math_10_quadratic",
                name="Quadratic Equations",
                subject=SubjectDomain.MATHEMATICS,
                grade=10,
                chapter="Chapter 4",
                keywords=["quadratic", "roots", "discriminant", "formula", "factorization"],
                bloom_level=BloomLevel.ANALYZE,
                prerequisite_ids=["math_9_polynomial"]
            ),
        ]
        
        # Add all concepts
        for concept in science_concepts + math_concepts:
            self.add_concept(concept)
    
    def add_concept(self, concept: Concept) -> None:
        """Add a concept to the knowledge graph."""
        self._concepts[concept.id] = concept
        self._concept_by_subject[concept.subject].append(concept.id)
        self._concept_by_grade[concept.grade].append(concept.id)
        
        # Index keywords
        for keyword in concept.keywords:
            self._keyword_index[keyword.lower()].add(concept.id)
    
    def get_concept(self, concept_id: str) -> Optional[Concept]:
        """Get concept by ID."""
        return self._concepts.get(concept_id)
    
    def get_prerequisites(self, concept_id: str, recursive: bool = True) -> List[str]:
        """
        Get prerequisite concepts for a given concept.
        
        Args:
            concept_id: Concept to get prerequisites for
            recursive: Whether to get transitive prerequisites
            
        Returns:
            List of prerequisite concept IDs
        """
        concept = self._concepts.get(concept_id)
        if not concept:
            return []
        
        if not recursive:
            return concept.prerequisite_ids
        
        # BFS for all prerequisites
        all_prereqs: List[str] = []
        visited: Set[str] = set()
        queue = list(concept.prerequisite_ids)
        
        while queue:
            prereq_id = queue.pop(0)
            if prereq_id in visited:
                continue
            visited.add(prereq_id)
            all_prereqs.append(prereq_id)
            
            prereq = self._concepts.get(prereq_id)
            if prereq:
                queue.extend(prereq.prerequisite_ids)
        
        return all_prereqs
    
    def find_concepts_by_keywords(self, text: str) -> List[Tuple[str, float]]:
        """
        Find concepts matching keywords in text.
        
        Returns list of (concept_id, match_score) tuples.
        """
        text_lower = text.lower()
        words = set(re.findall(r'\b\w+\b', text_lower))
        
        concept_scores: Dict[str, float] = defaultdict(float)
        
        for word in words:
            if word in self._keyword_index:
                for concept_id in self._keyword_index[word]:
                    concept = self._concepts[concept_id]
                    # Score based on keyword importance
                    concept_scores[concept_id] += 1.0 / len(concept.keywords)
        
        # Normalize scores
        if concept_scores:
            max_score = max(concept_scores.values())
            concept_scores = {
                k: v / max_score for k, v in concept_scores.items()
            }
        
        return sorted(concept_scores.items(), key=lambda x: x[1], reverse=True)
    
    def detect_bloom_level(self, text: str) -> BloomLevel:
        """Detect Bloom's taxonomy level from text content."""
        text_lower = text.lower()
        
        level_scores: Dict[BloomLevel, int] = {level: 0 for level in BloomLevel}
        
        for level, verbs in BLOOM_VERBS.items():
            for verb in verbs:
                # Count occurrences
                pattern = rf'\b{verb}(?:s|ed|ing|e)?\b'
                matches = len(re.findall(pattern, text_lower))
                level_scores[level] += matches
        
        # Return highest scoring level
        if max(level_scores.values()) == 0:
            return BloomLevel.UNDERSTAND  # Default
        
        return max(level_scores, key=level_scores.get)
    
    def calculate_readability(self, text: str) -> Dict[str, float]:
        """Calculate readability metrics."""
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        words = re.findall(r'\b\w+\b', text)
        
        if not sentences or not words:
            return {"flesch_kincaid": 0.0, "avg_sentence_length": 0.0}
        
        # Simple syllable count (approximation)
        def count_syllables(word: str) -> int:
            word = word.lower()
            vowels = "aeiou"
            count = 0
            prev_vowel = False
            for char in word:
                is_vowel = char in vowels
                if is_vowel and not prev_vowel:
                    count += 1
                prev_vowel = is_vowel
            return max(1, count)
        
        total_syllables = sum(count_syllables(w) for w in words)
        avg_syllables = total_syllables / len(words)
        avg_sentence_length = len(words) / len(sentences)
        
        # Flesch-Kincaid Reading Ease
        flesch = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables)
        flesch = max(0, min(100, flesch))  # Clamp to 0-100
        
        return {
            "flesch_kincaid": flesch,
            "avg_sentence_length": avg_sentence_length,
            "avg_syllables_per_word": avg_syllables,
            "total_sentences": len(sentences),
            "total_words": len(words)
        }
    
    def validate_content(
        self,
        text: str,
        grade: int,
        subject: str,
        board: Board = Board.CBSE
    ) -> ValidationResult:
        """
        Validate content against curriculum requirements.
        
        Args:
            text: Content text to validate
            grade: Target grade level (5-12)
            subject: Subject area
            board: Education board
            
        Returns:
            ValidationResult with detailed analysis
        """
        # Get grade profile
        profile = GRADE_PROFILES.get(grade, GRADE_PROFILES[8])
        
        # Map subject string to domain
        subject_map = {
            "science": SubjectDomain.SCIENCE,
            "mathematics": SubjectDomain.MATHEMATICS,
            "math": SubjectDomain.MATHEMATICS,
            "physics": SubjectDomain.PHYSICS,
            "chemistry": SubjectDomain.CHEMISTRY,
            "biology": SubjectDomain.BIOLOGY,
            "social studies": SubjectDomain.SOCIAL_STUDIES,
            "history": SubjectDomain.HISTORY,
            "geography": SubjectDomain.GEOGRAPHY,
            "civics": SubjectDomain.CIVICS,
            "english": SubjectDomain.ENGLISH,
            "hindi": SubjectDomain.HINDI,
        }
        domain = subject_map.get(subject.lower(), SubjectDomain.SCIENCE)
        
        # Find matching concepts
        concept_matches = self.find_concepts_by_keywords(text)
        matched_concepts = [c[0] for c in concept_matches[:5]]  # Top 5 matches
        
        # Detect Bloom's level
        detected_bloom = self.detect_bloom_level(text)
        bloom_appropriate = detected_bloom.value <= profile.max_bloom_level.value
        
        # Check readability
        readability = self.calculate_readability(text)
        flesch = readability["flesch_kincaid"]
        grade_appropriate = (
            profile.reading_level_flesch[0] <= flesch <= profile.reading_level_flesch[1]
        )
        sentence_ok = readability["avg_sentence_length"] <= profile.max_sentence_length
        
        # Check prerequisites
        missing_prereqs: List[str] = []
        forward_refs: List[str] = []
        
        for concept_id, score in concept_matches:
            if score < 0.3:  # Low confidence match
                continue
                
            concept = self._concepts.get(concept_id)
            if not concept:
                continue
            
            # Check if concept is above grade level
            if concept.grade > grade:
                forward_refs.append(f"{concept.name} (Grade {concept.grade})")
            
            # Check if prerequisites are covered
            prereqs = self.get_prerequisites(concept_id)
            for prereq_id in prereqs:
                prereq = self._concepts.get(prereq_id)
                if prereq and prereq.grade < grade:
                    # Check if prereq concepts are referenced in text
                    prereq_mentioned = any(
                        kw.lower() in text.lower() 
                        for kw in prereq.keywords[:3]  # Check top keywords
                    )
                    if not prereq_mentioned:
                        # Check if there are other texts that could cover this
                        missing_prereqs.append(prereq.name)
        
        # Deduplicate
        missing_prereqs = list(set(missing_prereqs))[:5]
        forward_refs = list(set(forward_refs))[:3]
        
        # Vocabulary issues (simplified check)
        vocab_issues: List[str] = []
        complex_words = [
            w for w in re.findall(r'\b\w+\b', text)
            if len(w) > 12 and readability["avg_syllables_per_word"] > profile.max_word_syllables
        ]
        if complex_words:
            vocab_issues.append(f"Complex vocabulary: {', '.join(complex_words[:3])}")
        
        # Generate recommendations
        recommendations: List[str] = []
        
        if not bloom_appropriate:
            recommendations.append(
                f"Simplify cognitive level: content uses {detected_bloom.name} level, "
                f"but grade {grade} max is {profile.max_bloom_level.name}"
            )
        
        if not grade_appropriate:
            if flesch < profile.reading_level_flesch[0]:
                recommendations.append("Content is too complex - use simpler sentences")
            else:
                recommendations.append("Content is too simple for this grade level")
        
        if not sentence_ok:
            recommendations.append(
                f"Reduce sentence length: avg {readability['avg_sentence_length']:.1f} words, "
                f"max recommended {profile.max_sentence_length}"
            )
        
        if missing_prereqs:
            recommendations.append(
                f"Consider introducing prerequisite concepts: {', '.join(missing_prereqs[:2])}"
            )
        
        if forward_refs:
            recommendations.append(
                f"Remove advanced concepts: {', '.join(forward_refs[:2])}"
            )
        
        # Calculate alignment score
        score_components = [
            0.25 * (1.0 if bloom_appropriate else 0.5),
            0.20 * (1.0 if grade_appropriate else 0.5),
            0.20 * (1.0 if sentence_ok else 0.7),
            0.20 * (1.0 - min(len(missing_prereqs) / 5, 1.0)),
            0.15 * (1.0 - min(len(forward_refs) / 3, 1.0)),
        ]
        alignment_score = sum(score_components)
        
        # Board alignment (simplified - CBSE baseline)
        board_alignment = {
            Board.CBSE: alignment_score,
            Board.ICSE: alignment_score * 0.95,  # Slight variation
        }
        
        return ValidationResult(
            alignment_score=round(alignment_score, 3),
            is_valid=alignment_score >= 0.7,
            bloom_level_detected=detected_bloom,
            bloom_level_appropriate=bloom_appropriate,
            grade_appropriate=grade_appropriate,
            missing_prerequisites=missing_prereqs,
            forward_references=forward_refs,
            vocabulary_issues=vocab_issues,
            sentence_complexity_ok=sentence_ok,
            recommendations=recommendations,
            matched_concepts=matched_concepts,
            board_alignment=board_alignment
        )
    
    def get_concept_path(
        self,
        from_grade: int,
        to_grade: int,
        subject: SubjectDomain
    ) -> List[List[str]]:
        """
        Get the learning path (concept progression) for a subject.
        
        Returns list of concept IDs per grade.
        """
        path = []
        for grade in range(from_grade, to_grade + 1):
            grade_concepts = [
                cid for cid in self._concept_by_grade.get(grade, [])
                if self._concepts[cid].subject == subject
            ]
            if grade_concepts:
                path.append(grade_concepts)
        return path


# Global instance
_knowledge_graph: Optional[NCERTKnowledgeGraph] = None


def get_knowledge_graph() -> NCERTKnowledgeGraph:
    """Get global knowledge graph instance."""
    global _knowledge_graph
    if _knowledge_graph is None:
        _knowledge_graph = NCERTKnowledgeGraph()
    return _knowledge_graph


def validate_curriculum_alignment(
    text: str,
    grade: int,
    subject: str,
    board: str = "cbse"
) -> ValidationResult:
    """
    Convenience function for curriculum validation.
    
    Args:
        text: Content text
        grade: Target grade (5-12)
        subject: Subject name
        board: Education board
        
    Returns:
        ValidationResult
    """
    graph = get_knowledge_graph()
    board_enum = Board(board.lower()) if board.lower() in [b.value for b in Board] else Board.CBSE
    return graph.validate_content(text, grade, subject, board_enum)
