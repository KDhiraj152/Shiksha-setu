"""
Code-Mixing Generator for Hindi-English Mixed Language Content.

This module generates natural code-mixed content (Hinglish) for educational purposes,
mimicking how many Indian students naturally communicate.
"""
import re
import random
import logging
from typing import List, Dict, Any, Tuple, Optional, Set
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class Language(Enum):
    """Language types."""
    HINDI = "hindi"
    ENGLISH = "english"
    HINGLISH = "hinglish"


class MixingStrategy(Enum):
    """Code-mixing strategies."""
    NATURAL = "natural"  # Natural switching at phrase boundaries
    TECHNICAL = "technical"  # Keep technical terms in English
    BALANCED = "balanced"  # Equal mix of both languages
    MATRIX = "matrix"  # Matrix language with embedded words


@dataclass
class CodeMixedText:
    """Represents code-mixed text."""
    text: str
    language: Language
    mixing_ratio: float  # 0.0 (all Hindi) to 1.0 (all English)
    switch_points: List[int]
    strategy: MixingStrategy
    metadata: Dict[str, Any]


class HindiEnglishVocabulary:
    """
    Vocabulary mappings between Hindi and English for code-mixing.
    """
    
    # Common words that are often code-mixed
    COMMON_HINDI_ENGLISH = {
        # Verbs
        'है': 'is',
        'हैं': 'are',
        'था': 'was',
        'थे': 'were',
        'करना': 'to do',
        'होना': 'to be',
        'जाना': 'to go',
        'आना': 'to come',
        'देखना': 'to see',
        'समझना': 'to understand',
        
        # Nouns
        'किताब': 'book',
        'पानी': 'water',
        'खाना': 'food',
        'घर': 'house',
        'स्कूल': 'school',
        'टीचर': 'teacher',
        'क्लास': 'class',
        'परीक्षा': 'exam',
        'सवाल': 'question',
        'जवाब': 'answer',
        
        # Adjectives
        'अच्छा': 'good',
        'बुरा': 'bad',
        'बड़ा': 'big',
        'छोटा': 'small',
        'आसान': 'easy',
        'मुश्किल': 'difficult',
        
        # Question words
        'क्या': 'what',
        'कब': 'when',
        'कहाँ': 'where',
        'कैसे': 'how',
        'क्यों': 'why',
        'कौन': 'who',
        
        # Conjunctions
        'और': 'and',
        'लेकिन': 'but',
        'या': 'or',
        'क्योंकि': 'because',
        'इसलिए': 'therefore',
    }
    
    # English words commonly used in Hinglish (loan words)
    ENGLISH_LOAN_WORDS = [
        # Education
        'class', 'teacher', 'student', 'homework', 'exam', 'test',
        'school', 'college', 'university', 'book', 'copy', 'pen',
        'subject', 'chapter', 'lesson', 'question', 'answer',
        
        # Technology
        'computer', 'mobile', 'phone', 'internet', 'online', 'app',
        'website', 'email', 'video', 'photo', 'file', 'folder',
        
        # Modern life
        'time', 'date', 'day', 'week', 'month', 'year',
        'bus', 'train', 'car', 'bike', 'road', 'station',
        'market', 'shop', 'mall', 'restaurant', 'hotel',
        
        # Common adjectives
        'good', 'bad', 'nice', 'easy', 'difficult', 'simple',
        'important', 'special', 'normal', 'perfect', 'total',
    ]
    
    # Technical terms that should remain in English
    TECHNICAL_TERMS = {
        'Mathematics': [
            'equation', 'formula', 'variable', 'constant', 'coefficient',
            'addition', 'subtraction', 'multiplication', 'division',
            'fraction', 'decimal', 'percentage', 'ratio', 'proportion',
            'algebra', 'geometry', 'trigonometry', 'calculus', 'statistics',
            'theorem', 'proof', 'solution', 'answer', 'problem'
        ],
        'Science': [
            'photosynthesis', 'respiration', 'evaporation', 'condensation',
            'molecule', 'atom', 'element', 'compound', 'mixture',
            'force', 'energy', 'power', 'velocity', 'acceleration',
            'cell', 'tissue', 'organ', 'organism', 'species',
            'experiment', 'observation', 'hypothesis', 'theory', 'law'
        ],
        'General': [
            'computer', 'internet', 'technology', 'science', 'mathematics',
            'history', 'geography', 'economics', 'politics', 'democracy'
        ]
    }
    
    # Common Hinglish phrases (natural code-mixing patterns)
    HINGLISH_PHRASES = [
        # Greetings and responses
        ('hello', 'नमस्ते'),
        ('thank you', 'धन्यवाद'),
        ('okay', 'ठीक है'),
        ('yes', 'हाँ'),
        ('no', 'नहीं'),
        
        # Common expressions
        ('I mean', 'मेरा मतलब है'),
        ('you know', 'तुम जानते हो'),
        ('I think', 'मुझे लगता है'),
        ('actually', 'असल में'),
        ('basically', 'मूल रूप से'),
    ]
    
    @classmethod
    def is_technical_term(cls, word: str, subject: Optional[str] = None) -> bool:
        """Check if a word is a technical term that should stay in English."""
        word_lower = word.lower()
        
        if subject:
            technical_set = {term.lower() for term in cls.TECHNICAL_TERMS.get(subject, [])}
            if word_lower in technical_set:
                return True
        
        # Check all technical terms
        for term_list in cls.TECHNICAL_TERMS.values():
            if word_lower in [term.lower() for term in term_list]:
                return True
        
        return False
    
    @classmethod
    def is_english_loan_word(cls, word: str) -> bool:
        """Check if word is commonly used English loan word in Hindi."""
        return word.lower() in [w.lower() for w in cls.ENGLISH_LOAN_WORDS]


class LanguageDetector:
    """
    Detects language of text segments.
    """
    
    # Hindi Unicode range
    HINDI_RANGE = (0x0900, 0x097F)
    
    # English Unicode range
    ENGLISH_RANGE = (0x0041, 0x007A)
    
    @classmethod
    def detect_language(cls, text: str) -> Language:
        """
        Detect primary language of text.
        
        Args:
            text: Text to analyze
        
        Returns:
            Language enum
        """
        hindi_chars = 0
        english_chars = 0
        
        for char in text:
            char_code = ord(char)
            
            if cls.HINDI_RANGE[0] <= char_code <= cls.HINDI_RANGE[1]:
                hindi_chars += 1
            elif ((cls.ENGLISH_RANGE[0] <= char_code <= 0x005A) or  # A-Z
                  (0x0061 <= char_code <= cls.ENGLISH_RANGE[1])):  # a-z
                english_chars += 1
        
        total = hindi_chars + english_chars
        
        if total == 0:
            return Language.ENGLISH
        
        hindi_ratio = hindi_chars / total
        
        if hindi_ratio > 0.6:
            return Language.HINDI
        elif hindi_ratio < 0.4:
            return Language.ENGLISH
        else:
            return Language.HINGLISH
    
    @classmethod
    def split_by_language(cls, text: str) -> List[Tuple[str, Language]]:
        """
        Split text into segments by language.
        
        Args:
            text: Text to split
        
        Returns:
            List of (segment, language) tuples
        """
        segments = []
        current_segment = ""
        current_lang = None
        
        for char in text:
            char_lang = cls.detect_language(char)
            
            if current_lang is None:
                current_lang = char_lang
                current_segment = char
            elif current_lang == char_lang:
                current_segment += char
            else:
                if current_segment.strip():
                    segments.append((current_segment, current_lang))
                current_segment = char
                current_lang = char_lang
        
        # Add final segment
        if current_segment.strip():
            segments.append((current_segment, current_lang))
        
        return segments


class CodeMixer:
    """
    Generates code-mixed (Hinglish) content for educational purposes.
    
    Features:
    - Natural language switching at phrase boundaries
    - Preserve technical terms in English
    - Configurable mixing ratios
    - Multiple mixing strategies
    - Context-aware code-mixing
    """
    
    def __init__(
        self,
        default_strategy: MixingStrategy = MixingStrategy.NATURAL,
        preserve_technical: bool = True
    ):
        """
        Initialize the code mixer.
        
        Args:
            default_strategy: Default mixing strategy
            preserve_technical: Whether to keep technical terms in English
        """
        self.default_strategy = default_strategy
        self.preserve_technical = preserve_technical
        self.vocab = HindiEnglishVocabulary()
        self.detector = LanguageDetector()
        logger.info(f"CodeMixer initialized with strategy: {default_strategy.value}")
    
    def generate_code_mixed_text(
        self,
        text: str,
        target_language: Language = Language.HINGLISH,
        mixing_ratio: float = 0.5,
        strategy: Optional[MixingStrategy] = None,
        subject: Optional[str] = None
    ) -> CodeMixedText:
        """
        Generate code-mixed text from input.
        
        Args:
            text: Input text (English or Hindi)
            target_language: Target language
            mixing_ratio: Ratio of English (0.0=all Hindi, 1.0=all English)
            strategy: Code-mixing strategy
            subject: Subject area for technical term preservation
        
        Returns:
            CodeMixedText object
        """
        if not text or len(text.strip()) == 0:
            raise ValueError("Text cannot be empty")
        
        # Validate mixing ratio
        mixing_ratio = max(0.0, min(1.0, mixing_ratio))
        
        strategy = strategy or self.default_strategy
        
        logger.info(
            f"Generating code-mixed text: ratio={mixing_ratio:.2f}, "
            f"strategy={strategy.value}"
        )
        
        # Detect input language
        input_language = self.detector.detect_language(text)
        logger.debug(f"Detected input language: {input_language.value}")
        
        # Generate mixed text based on strategy
        if strategy == MixingStrategy.NATURAL:
            mixed_text, switch_points = self._natural_mixing(
                text, mixing_ratio, subject
            )
        elif strategy == MixingStrategy.TECHNICAL:
            mixed_text, switch_points = self._technical_mixing(
                text, mixing_ratio, subject
            )
        elif strategy == MixingStrategy.BALANCED:
            mixed_text, switch_points = self._balanced_mixing(
                text, mixing_ratio
            )
        else:  # MATRIX
            mixed_text, switch_points = self._matrix_mixing(
                text, mixing_ratio
            )
        
        logger.info(f"Generated code-mixed text with {len(switch_points)} switch points")
        
        return CodeMixedText(
            text=mixed_text,
            language=target_language,
            mixing_ratio=mixing_ratio,
            switch_points=switch_points,
            strategy=strategy,
            metadata={
                'input_language': input_language.value,
                'subject': subject,
                'preserve_technical': self.preserve_technical
            }
        )
    
    def _natural_mixing(
        self,
        text: str,
        mixing_ratio: float,
        subject: Optional[str] = None
    ) -> Tuple[str, List[int]]:
        """
        Natural code-mixing at phrase/sentence boundaries.
        
        This mimics natural speech patterns where people switch languages
        at natural breaking points.
        """
        # Split into sentences
        sentences = self._split_sentences(text)
        
        mixed_sentences = []
        switch_points = []
        
        for sentence in sentences:
            # Decide if this sentence should be mixed
            if random.random() < mixing_ratio:
                # Mix this sentence
                mixed_sentence = self._mix_sentence(sentence, subject)
                mixed_sentences.append(mixed_sentence)
                switch_points.append(len(' '.join(mixed_sentences)))
            else:
                # Keep original
                mixed_sentences.append(sentence)
        
        return ' '.join(mixed_sentences), switch_points
    
    def _technical_mixing(
        self,
        text: str,
        mixing_ratio: float,
        subject: Optional[str] = None
    ) -> Tuple[str, List[int]]:
        """
        Keep all technical terms in English, mix other words.
        """
        words = text.split()
        mixed_words = []
        switch_points = []
        
        for word in words:
            # Check if technical term
            if self.vocab.is_technical_term(word, subject):
                # Keep in English
                mixed_words.append(word)
            elif random.random() < (1 - mixing_ratio):
                # Convert to Hindi (if possible)
                hindi_word = self._get_hindi_equivalent(word)
                mixed_words.append(hindi_word if hindi_word else word)
                switch_points.append(len(' '.join(mixed_words)))
            else:
                # Keep in English
                mixed_words.append(word)
        
        return ' '.join(mixed_words), switch_points
    
    def _balanced_mixing(
        self,
        text: str,
        mixing_ratio: float
    ) -> Tuple[str, List[int]]:
        """
        Balanced mixing with roughly equal Hindi and English.
        """
        words = text.split()
        mixed_words = []
        switch_points = []
        
        english_count = 0
        hindi_count = 0
        
        for word in words:
            # Calculate current ratio
            total = english_count + hindi_count
            current_ratio = english_count / total if total > 0 else 0.5
            
            # Decide language for this word
            if current_ratio < mixing_ratio:
                # Use English
                mixed_words.append(word)
                english_count += 1
            else:
                # Use Hindi
                hindi_word = self._get_hindi_equivalent(word)
                mixed_words.append(hindi_word if hindi_word else word)
                hindi_count += 1
                switch_points.append(len(' '.join(mixed_words)))
        
        return ' '.join(mixed_words), switch_points
    
    def _matrix_mixing(
        self,
        text: str,
        mixing_ratio: float
    ) -> Tuple[str, List[int]]:
        """
        Matrix language approach: One base language with embedded words from other.
        """
        # Use Hindi as matrix language, embed English words
        words = text.split()
        mixed_words = []
        switch_points = []
        
        for word in words:
            # Embed English words at specified ratio
            if random.random() < mixing_ratio or self.vocab.is_english_loan_word(word):
                # Keep English
                mixed_words.append(word)
            else:
                # Use Hindi
                hindi_word = self._get_hindi_equivalent(word)
                mixed_words.append(hindi_word if hindi_word else word)
                if hindi_word:
                    switch_points.append(len(' '.join(mixed_words)))
        
        return ' '.join(mixed_words), switch_points
    
    def _mix_sentence(self, sentence: str, subject: Optional[str]) -> str:
        """Mix a single sentence naturally."""
        # For simplicity, add some English loan words
        words = sentence.split()
        
        # Replace some words with Hinglish equivalents
        for i, word in enumerate(words):
            if (not self.vocab.is_technical_term(word, subject) and
                random.random() < 0.3):  # 30% chance to keep English
                # Keep as is (English loan word in Hindi context)
                continue
        
        return ' '.join(words)
    
    def _get_hindi_equivalent(self, english_word: str) -> Optional[str]:
        """Get Hindi equivalent of English word if available."""
        # Reverse lookup in vocabulary
        word_lower = english_word.lower()
        
        for hindi, english in self.vocab.COMMON_HINDI_ENGLISH.items():
            if english == word_lower:
                return hindi
        
        return None
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def detect_code_mixing_quality(self, text: str) -> Dict[str, Any]:
        """
        Analyze the quality and naturalness of code-mixing.
        
        Args:
            text: Code-mixed text to analyze
        
        Returns:
            Dictionary with quality metrics
        """
        language = self.detector.detect_language(text)
        segments = self.detector.split_by_language(text)
        
        # Calculate metrics
        total_segments = len(segments)
        switch_count = total_segments - 1  # Number of language switches
        
        # Count Hindi and English segments
        hindi_segments = sum(1 for _, lang in segments if lang == Language.HINDI)
        english_segments = sum(1 for _, lang in segments if lang == Language.ENGLISH)
        
        # Calculate naturalness (fewer switches = more natural)
        switches_per_sentence = switch_count / max(1, len(self._split_sentences(text)))
        
        naturalness_score = max(0, 1.0 - (switches_per_sentence / 5.0))  # 0-1 scale
        
        return {
            'detected_language': language.value,
            'total_segments': total_segments,
            'switch_count': switch_count,
            'hindi_segments': hindi_segments,
            'english_segments': english_segments,
            'mixing_ratio': english_segments / max(1, total_segments),
            'naturalness_score': naturalness_score,
            'switches_per_sentence': switches_per_sentence
        }


# Convenience function
def create_hinglish_text(
    text: str,
    mixing_ratio: float = 0.5,
    subject: Optional[str] = None
) -> str:
    """
    Quick utility to create Hinglish text.
    
    Args:
        text: Input English text
        mixing_ratio: Ratio of English (0.5 = balanced)
        subject: Optional subject area
    
    Returns:
        Code-mixed Hinglish text
    """
    mixer = CodeMixer()
    result = mixer.generate_code_mixed_text(
        text=text,
        mixing_ratio=mixing_ratio,
        subject=subject
    )
    return result.text


if __name__ == "__main__":
    # Example usage
    sample_text = """
    Photosynthesis is the process by which plants make their food.
    They use sunlight, water, and carbon dioxide to produce glucose and oxygen.
    This is a very important process for all living things on Earth.
    """
    
    mixer = CodeMixer()
    
    # Try different strategies
    strategies = [
        MixingStrategy.NATURAL,
        MixingStrategy.TECHNICAL,
        MixingStrategy.BALANCED
    ]
    
    for strategy in strategies:
        print(f"\n{'='*60}")
        print(f"Strategy: {strategy.value}")
        print('='*60)
        
        result = mixer.generate_code_mixed_text(
            text=sample_text,
            mixing_ratio=0.6,  # 60% English, 40% Hindi
            strategy=strategy,
            subject="Science"
        )
        
        print(f"\nMixed Text:\n{result.text}")
        print(f"\nMixing Ratio: {result.mixing_ratio:.2f}")
        print(f"Switch Points: {len(result.switch_points)}")
        
        # Analyze quality
        quality = mixer.detect_code_mixing_quality(result.text)
        print("\nQuality Metrics:")
        print(f"  Naturalness score: {quality['naturalness_score']:.2f}")
        print(f"  Switches per Sentence: {quality['switches_per_sentence']:.2f}")
