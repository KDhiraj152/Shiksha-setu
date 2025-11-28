"""
Curriculum validator using IndicBERT fine-tuned on NCERT corpus.
For grade-level adaptation and curriculum alignment.
"""
import torch
import logging
from typing import Dict, Any, List, Optional
from pathlib import Path

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from ..core.config import settings

logger = logging.getLogger(__name__)


class CurriculumValidator:
    """Validate content against NCERT curriculum standards."""
    
    def __init__(self):
        self.model_id = settings.VALIDATOR_MODEL_ID
        self.fine_tune_path = settings.VALIDATOR_FINE_TUNE_PATH
        self.model = None
        self.tokenizer = None
        
        # Grade level mappings
        self.grade_ranges = {
            "primary": (1, 5),
            "middle": (6, 8),
            "secondary": (9, 10),
            "senior_secondary": (11, 12)
        }
        
        # Subject categories
        self.subjects = [
            "mathematics", "science", "social_science",
            "english", "hindi", "languages", "arts", "physical_education"
        ]
    
    def _load_model(self):
        """Load IndicBERT model for validation."""
        if self.model is None:
            logger.info(f"Loading curriculum validator: {self.model_id}")
            
            # Try to load fine-tuned model first
            fine_tune_path = Path(self.fine_tune_path)
            if fine_tune_path.exists():
                try:
                    logger.info(f"Loading fine-tuned model from: {self.fine_tune_path}")
                    self.tokenizer = AutoTokenizer.from_pretrained(self.fine_tune_path)
                    self.model = AutoModelForSequenceClassification.from_pretrained(
                        self.fine_tune_path,
                        torch_dtype=torch.float16,
                        device_map="auto"
                    )
                    logger.info("Fine-tuned curriculum validator loaded successfully")
                    return
                except Exception as e:
                    logger.warning(f"Failed to load fine-tuned model: {e}, falling back to base model")
            
            # Fallback to base model
            try:
                logger.info(f"Loading base model: {self.model_id}")
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_id,
                    cache_dir=str(settings.MODEL_CACHE_DIR)
                )
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    self.model_id,
                    num_labels=12,  # 12 grades
                    torch_dtype=torch.float16,
                    device_map="auto",
                    cache_dir=str(settings.MODEL_CACHE_DIR)
                )
                logger.info("Base curriculum validator loaded successfully")
            except Exception as e:
                # If gated model fails, try ungated alternative (MuRIL)
                if 'gated repo' in str(e).lower() or '401' in str(e):
                    logger.warning(f"Model {self.model_id} is gated or requires authentication")
                    logger.info("Trying ungated alternative: google/muril-base-cased")
                    try:
                        fallback_model = 'google/muril-base-cased'
                        self.tokenizer = AutoTokenizer.from_pretrained(
                            fallback_model,
                            cache_dir=str(settings.MODEL_CACHE_DIR)
                        )
                        self.model = AutoModelForSequenceClassification.from_pretrained(
                            fallback_model,
                            num_labels=12,
                            torch_dtype=torch.float16,
                            device_map="auto",
                            cache_dir=str(settings.MODEL_CACHE_DIR)
                        )
                        self.model_id = fallback_model
                        logger.info(f"Fallback model loaded: {fallback_model}")
                        logger.info("Note: For better accuracy with Indian languages, authenticate with HuggingFace:")
                        logger.info("  1. Get token: https://huggingface.co/settings/tokens")
                        logger.info("  2. Run: huggingface-cli login")
                        logger.info("  3. Or set: HUGGINGFACE_API_KEY in .env")
                    except Exception as fallback_error:
                        logger.error(f"Failed to load fallback model: {fallback_error}")
                        raise
                else:
                    logger.error(f"Failed to load curriculum validator: {e}")
                    raise
    
    def validate_grade_level(
        self,
        text: str,
        target_grade: int,
        subject: str = "general",
        confidence_threshold: float = 0.6
    ) -> Dict[str, Any]:
        """
        Validate if text is appropriate for target grade.
        
        Args:
            text: Content to validate
            target_grade: Target grade level (1-12)
            subject: Subject area
            confidence_threshold: Minimum confidence for classification
            
        Returns:
            Dict with validation results
        """
        if not 1 <= target_grade <= 12:
            raise ValueError("target_grade must be between 1 and 12")
        
        self._load_model()
        
        # Encode text
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding=True
        ).to(self.model.device)
        
        # Get predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1)
            predicted_grade = torch.argmax(probabilities, dim=-1).item() + 1  # 1-12
            confidence = probabilities[0][predicted_grade - 1].item()
        
        # Determine if appropriate (within 1 grade level)
        grade_difference = abs(predicted_grade - target_grade)
        is_appropriate = grade_difference <= 1
        
        # Get grade range category
        grade_category = self._get_grade_category(target_grade)
        
        return {
            "is_appropriate": is_appropriate,
            "predicted_grade": predicted_grade,
            "target_grade": target_grade,
            "confidence": confidence,
            "grade_difference": grade_difference,
            "grade_category": grade_category,
            "subject": subject,
            "recommendation": self._get_recommendation(
                predicted_grade,
                target_grade,
                confidence,
                confidence_threshold
            )
        }
    
    def _get_grade_category(self, grade: int) -> str:
        """Get grade range category."""
        for category, (min_grade, max_grade) in self.grade_ranges.items():
            if min_grade <= grade <= max_grade:
                return category
        return "unknown"
    
    def _get_recommendation(
        self,
        predicted: int,
        target: int,
        confidence: float,
        threshold: float
    ) -> str:
        """Generate recommendation based on validation."""
        if confidence < threshold:
            return f"Low confidence ({confidence:.2%}). Manual review recommended."
        
        diff = predicted - target
        if diff == 0:
            return f"Content is perfectly aligned with Grade {target} (confidence: {confidence:.2%})"
        elif diff == 1:
            return f"Content is slightly advanced. Consider simplifying 10-20% for Grade {target}"
        elif diff == -1:
            return f"Content is slightly simple. Consider adding 10-20% more depth for Grade {target}"
        elif diff > 1:
            return f"Content is too advanced (Grade {predicted}). Significant simplification needed for Grade {target}"
        else:
            return f"Content is too simple (Grade {predicted}). Significant enrichment needed for Grade {target}"
    
    def check_terminology_alignment(
        self,
        text: str,
        subject: str,
        grade: int
    ) -> Dict[str, Any]:
        """
        Check if terminology matches NCERT standards.
        
        Args:
            text: Content to check
            subject: Subject area
            grade: Grade level
            
        Returns:
            Dict with terminology analysis
        """
        # Load NCERT terminology database
        ncert_terms = self._load_ncert_terminology(subject, grade)
        
        # Extract terms from text
        extracted_terms = self._extract_technical_terms(text)
        
        if not extracted_terms:
            return {
                "aligned_terms": [],
                "misaligned_terms": [],
                "alignment_score": 1.0,
                "ncert_coverage": 0.0,
                "recommendation": "No technical terms found"
            }
        
        # Check alignment
        aligned_terms = [t for t in extracted_terms if t.lower() in ncert_terms]
        misaligned_terms = [t for t in extracted_terms if t.lower() not in ncert_terms]
        
        alignment_score = len(aligned_terms) / len(extracted_terms) if extracted_terms else 1.0
        ncert_coverage = len(aligned_terms) / len(ncert_terms) if ncert_terms else 0.0
        
        return {
            "aligned_terms": aligned_terms,
            "misaligned_terms": misaligned_terms,
            "alignment_score": alignment_score,
            "ncert_coverage": ncert_coverage,
            "recommendation": self._get_terminology_recommendation(alignment_score, misaligned_terms)
        }
    
    def _get_terminology_recommendation(
        self,
        alignment_score: float,
        misaligned_terms: List[str]
    ) -> str:
        """Generate terminology recommendation."""
        if alignment_score >= 0.9:
            return "Excellent terminology alignment with NCERT standards"
        elif alignment_score >= 0.7:
            return f"Good alignment. Consider reviewing: {', '.join(misaligned_terms[:3])}"
        elif alignment_score >= 0.5:
            return f"Moderate alignment. Review these terms: {', '.join(misaligned_terms[:5])}"
        else:
            return f"Poor alignment. Significant terminology revision needed for {len(misaligned_terms)} terms"
    
    def _load_ncert_terminology(self, subject: str, grade: int) -> set:
        """Load NCERT terminology from curriculum database."""
        # Try to load from curriculum data
        curriculum_dir = Path("data/curriculum")
        term_file = curriculum_dir / f"ncert_terminology_{subject}_grade{grade}.json"
        
        if term_file.exists():
            try:
                import json
                with open(term_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    return set(term.lower() for term in data.get('terms', []))
            except Exception as e:
                logger.warning(f"Failed to load terminology file {term_file}: {e}")
        
        # Fallback: return empty set (will result in 0 coverage)
        logger.warning(f"No terminology data found for {subject} Grade {grade}")
        return set()
    
    def _extract_technical_terms(self, text: str) -> List[str]:
        """Extract technical/domain terms from text."""
        import re
        
        # Extract capitalized words and technical patterns
        # This is a simple heuristic; could be improved with NER
        words = []
        
        # Pattern 1: Capitalized words (likely proper nouns/technical terms)
        capitalized = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        words.extend(capitalized)
        
        # Pattern 2: Words with special characters (chemical formulas, equations)
        technical = re.findall(r'\b[A-Za-z]+[0-9]+\b|\b[A-Z]{2,}\b', text)
        words.extend(technical)
        
        # Remove duplicates and common words
        common_words = {'The', 'This', 'That', 'These', 'Those', 'For', 'With', 'From'}
        unique_terms = list(set(w for w in words if w not in common_words))
        
        return unique_terms
    
    def batch_validate(
        self,
        texts: List[str],
        target_grades: List[int],
        subjects: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Batch validate multiple texts.
        
        Args:
            texts: List of texts to validate
            target_grades: List of target grades
            subjects: Optional list of subjects
            
        Returns:
            List of validation results
        """
        if subjects is None:
            subjects = ["general"] * len(texts)
        
        if len(texts) != len(target_grades) or len(texts) != len(subjects):
            raise ValueError("texts, target_grades, and subjects must have same length")
        
        results = []
        for text, grade, subject in zip(texts, target_grades, subjects):
            try:
                result = self.validate_grade_level(text, grade, subject)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to validate text: {e}")
                results.append({
                    "error": str(e),
                    "target_grade": grade,
                    "subject": subject
                })
        
        return results
    
    def get_readability_metrics(self, text: str) -> Dict[str, Any]:
        """
        Calculate readability metrics for text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dict with readability scores
        """
        import re
        
        # Basic metrics
        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
        words = text.split()
        
        if not sentences or not words:
            return {
                "avg_sentence_length": 0,
                "avg_word_length": 0,
                "total_sentences": 0,
                "total_words": 0
            }
        
        avg_sentence_length = len(words) / len(sentences)
        avg_word_length = sum(len(w) for w in words) / len(words)
        
        # Estimate grade level (simple Flesch-Kincaid approximation)
        syllable_count = sum(self._count_syllables(w) for w in words)
        if len(words) > 0 and len(sentences) > 0:
            flesch_reading_ease = 206.835 - 1.015 * (len(words) / len(sentences)) - 84.6 * (syllable_count / len(words))
            estimated_grade = max(1, min(12, int((100 - flesch_reading_ease) / 10)))
        else:
            flesch_reading_ease = 0
            estimated_grade = 6
        
        return {
            "avg_sentence_length": round(avg_sentence_length, 2),
            "avg_word_length": round(avg_word_length, 2),
            "total_sentences": len(sentences),
            "total_words": len(words),
            "flesch_reading_ease": round(flesch_reading_ease, 2),
            "estimated_grade": estimated_grade
        }
    
    def _count_syllables(self, word: str) -> int:
        """Estimate syllable count for a word."""
        word = word.lower()
        vowels = "aeiou"
        syllable_count = 0
        previous_was_vowel = False
        
        for char in word:
            is_vowel = char in vowels
            if is_vowel and not previous_was_vowel:
                syllable_count += 1
            previous_was_vowel = is_vowel
        
        # Adjust for silent e
        if word.endswith('e'):
            syllable_count -= 1
        
        # Ensure at least 1 syllable
        return max(1, syllable_count)


# Global instance
_validator_instance = None


def get_curriculum_validator() -> CurriculumValidator:
    """Get or create global curriculum validator instance."""
    global _validator_instance
    if _validator_instance is None:
        _validator_instance = CurriculumValidator()
    return _validator_instance
