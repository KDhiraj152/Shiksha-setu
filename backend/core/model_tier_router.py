"""
Model Tier Router for resource-aware inference.

Routes tasks to appropriate model sizes based on complexity and available resources.
Implements local-first strategy with graceful degradation to API fallback.

Key Features:
- Language-aware complexity scoring with Indic script multipliers
- Predictive memory budgeting based on workload patterns
- Adaptive tier selection based on device capabilities
- Circuit breaker pattern for API fallback
"""
import logging
import re
from enum import Enum
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)


class ModelTier(str, Enum):
    """Model size tiers for resource-aware inference."""
    SMALL = "small"      # 1-3B params, <2GB RAM, fast inference
    MEDIUM = "medium"    # 7B params, 3-8GB RAM, balanced
    LARGE = "large"      # 13B+ params or API, 8GB+ RAM, best quality
    API = "api"          # External API fallback (Bhashini, OpenAI)


class ScriptType(str, Enum):
    """Unicode script classification for Indic languages."""
    LATIN = "latin"           # English, romanized text
    DEVANAGARI = "devanagari" # Hindi, Marathi, Sanskrit
    TAMIL = "tamil"           # Tamil script
    TELUGU = "telugu"         # Telugu script  
    BENGALI = "bengali"       # Bengali/Bangla script
    GUJARATI = "gujarati"     # Gujarati script
    KANNADA = "kannada"       # Kannada script
    MALAYALAM = "malayalam"   # Malayalam script
    ODIA = "odia"             # Odia/Oriya script
    GURMUKHI = "gurmukhi"     # Punjabi script
    UNKNOWN = "unknown"


@dataclass
class ScriptMetrics:
    """Script-specific metrics for complexity calculation."""
    script_type: ScriptType
    # Token expansion multiplier (Indic scripts expand ~1.5-2x vs Latin)
    token_multiplier: float
    # Average character per token (affects memory/compute)
    chars_per_token: float
    # Morphological complexity (agglutinative languages need more)
    morphological_factor: float


# Script multipliers based on NLLB-200 tokenizer analysis
SCRIPT_METRICS: Dict[ScriptType, ScriptMetrics] = {
    ScriptType.LATIN: ScriptMetrics(
        script_type=ScriptType.LATIN,
        token_multiplier=1.0,
        chars_per_token=4.0,
        morphological_factor=1.0
    ),
    ScriptType.DEVANAGARI: ScriptMetrics(
        script_type=ScriptType.DEVANAGARI,
        token_multiplier=1.4,
        chars_per_token=2.8,
        morphological_factor=1.2
    ),
    ScriptType.TAMIL: ScriptMetrics(
        script_type=ScriptType.TAMIL,
        token_multiplier=1.6,  # Tamil has complex agglutination
        chars_per_token=2.5,
        morphological_factor=1.5
    ),
    ScriptType.TELUGU: ScriptMetrics(
        script_type=ScriptType.TELUGU,
        token_multiplier=1.5,
        chars_per_token=2.6,
        morphological_factor=1.3
    ),
    ScriptType.BENGALI: ScriptMetrics(
        script_type=ScriptType.BENGALI,
        token_multiplier=1.4,
        chars_per_token=2.7,
        morphological_factor=1.2
    ),
    ScriptType.GUJARATI: ScriptMetrics(
        script_type=ScriptType.GUJARATI,
        token_multiplier=1.35,
        chars_per_token=2.9,
        morphological_factor=1.15
    ),
    ScriptType.KANNADA: ScriptMetrics(
        script_type=ScriptType.KANNADA,
        token_multiplier=1.45,
        chars_per_token=2.6,
        morphological_factor=1.25
    ),
    ScriptType.MALAYALAM: ScriptMetrics(
        script_type=ScriptType.MALAYALAM,
        token_multiplier=1.55,  # Malayalam is highly agglutinative
        chars_per_token=2.4,
        morphological_factor=1.4
    ),
    ScriptType.ODIA: ScriptMetrics(
        script_type=ScriptType.ODIA,
        token_multiplier=1.4,
        chars_per_token=2.7,
        morphological_factor=1.2
    ),
    ScriptType.GURMUKHI: ScriptMetrics(
        script_type=ScriptType.GURMUKHI,
        token_multiplier=1.35,
        chars_per_token=2.9,
        morphological_factor=1.15
    ),
    ScriptType.UNKNOWN: ScriptMetrics(
        script_type=ScriptType.UNKNOWN,
        token_multiplier=1.3,  # Conservative default
        chars_per_token=3.0,
        morphological_factor=1.1
    ),
}

# Language name to script mapping
LANGUAGE_TO_SCRIPT: Dict[str, ScriptType] = {
    # English
    "english": ScriptType.LATIN,
    "en": ScriptType.LATIN,
    # Hindi
    "hindi": ScriptType.DEVANAGARI,
    "hi": ScriptType.DEVANAGARI,
    "hin": ScriptType.DEVANAGARI,
    # Tamil
    "tamil": ScriptType.TAMIL,
    "ta": ScriptType.TAMIL,
    "tam": ScriptType.TAMIL,
    # Telugu
    "telugu": ScriptType.TELUGU,
    "te": ScriptType.TELUGU,
    "tel": ScriptType.TELUGU,
    # Bengali
    "bengali": ScriptType.BENGALI,
    "bangla": ScriptType.BENGALI,
    "bn": ScriptType.BENGALI,
    "ben": ScriptType.BENGALI,
    # Marathi
    "marathi": ScriptType.DEVANAGARI,
    "mr": ScriptType.DEVANAGARI,
    "mar": ScriptType.DEVANAGARI,
    # Gujarati
    "gujarati": ScriptType.GUJARATI,
    "gu": ScriptType.GUJARATI,
    "guj": ScriptType.GUJARATI,
    # Kannada
    "kannada": ScriptType.KANNADA,
    "kn": ScriptType.KANNADA,
    "kan": ScriptType.KANNADA,
    # Malayalam
    "malayalam": ScriptType.MALAYALAM,
    "ml": ScriptType.MALAYALAM,
    "mal": ScriptType.MALAYALAM,
    # Punjabi
    "punjabi": ScriptType.GURMUKHI,
    "pa": ScriptType.GURMUKHI,
    "pan": ScriptType.GURMUKHI,
    # Odia
    "odia": ScriptType.ODIA,
    "oriya": ScriptType.ODIA,
    "or": ScriptType.ODIA,
    "ori": ScriptType.ODIA,
    # Sanskrit (Devanagari script)
    "sanskrit": ScriptType.DEVANAGARI,
    "sa": ScriptType.DEVANAGARI,
    "san": ScriptType.DEVANAGARI,
}


@dataclass
class TaskComplexity:
    """Task complexity metrics for routing decisions."""
    token_count: int
    adjusted_token_count: int  # After script multiplier
    grade_level: int
    subject_technical: bool  # Science/Math vs Social/English
    translation_pairs: int   # Number of languages
    requires_cultural_context: bool
    complexity_score: float
    # New fields for language awareness
    source_script: ScriptType = ScriptType.LATIN
    target_scripts: list = field(default_factory=list)
    script_complexity_factor: float = 1.0
    morphological_complexity: float = 1.0


class ModelTierRouter:
    """
    Routes inference tasks to appropriate model tiers.
    
    Strategy:
    1. Calculate task complexity score
    2. Check available resources (memory, device)
    3. Select optimal tier (SMALL/MEDIUM/LARGE/API)
    4. Return model configuration
    
    Thresholds are tuned for Apple Silicon M4 with 16GB unified memory.
    """
    
    # Token thresholds for complexity scoring
    TOKEN_SMALL = 512       # <512 tokens = SMALL model sufficient
    TOKEN_MEDIUM = 2048     # <2048 tokens = MEDIUM model
    TOKEN_LARGE = 4096      # >4096 tokens = LARGE model or API
    
    # Grade level thresholds
    GRADE_SIMPLE = 8        # Grade 5-8: simpler language
    GRADE_COMPLEX = 10      # Grade 9-10: more complex
    
    # Technical subjects need better models
    TECHNICAL_SUBJECTS = {'Mathematics', 'Science', 'Physics', 'Chemistry', 'Biology'}
    
    # Memory budgets per tier (in GB)
    MEMORY_BUDGET = {
        ModelTier.SMALL: 2.0,    # 1.5B model in 4-bit
        ModelTier.MEDIUM: 6.0,   # 7B model in 4-bit
        ModelTier.LARGE: 12.0,   # 13B model in 4-bit or API
        ModelTier.API: 0.1       # Minimal local memory
    }
    
    # Model configurations per tier
    MODEL_CONFIGS = {
        ModelTier.SMALL: {
            "model_id": "Qwen/Qwen2.5-1.5B-Instruct",
            "quantization": "4bit",
            "max_tokens": 512,
            "batch_size": 8,
            "device_preference": ["mps", "cuda", "cpu"],
        },
        ModelTier.MEDIUM: {
            "model_id": "Qwen/Qwen2.5-7B-Instruct",
            "quantization": "4bit",
            "max_tokens": 2048,
            "batch_size": 4,
            "device_preference": ["mps", "cuda", "cpu"],
        },
        ModelTier.LARGE: {
            "model_id": "Qwen/Qwen2.5-14B-Instruct",  # Or vLLM endpoint
            "quantization": "4bit",
            "max_tokens": 4096,
            "batch_size": 1,
            "device_preference": ["cuda", "api"],  # Prefer GPU or API
        },
        ModelTier.API: {
            "api_provider": "bhashini",
            "fallback": True,
            "max_tokens": 4096,
        }
    }
    
    def __init__(self, max_memory_gb: float = 8.0, device_type: str = "mps"):
        """
        Initialize model tier router.
        
        Args:
            max_memory_gb: Maximum memory budget in GB (default 8GB for M4)
            device_type: Device type (mps, cuda, cpu)
        """
        self.max_memory_gb = max_memory_gb
        self.device_type = device_type
        
        # Track current memory usage (simplified - actual tracking in ModelLoader)
        self.current_memory_gb = 0.0
        
        logger.info(
            f"ModelTierRouter initialized: max_memory={max_memory_gb}GB, device={device_type}"
        )
    
    def _detect_script_from_text(self, text: str) -> ScriptType:
        """
        Detect script type from text using Unicode ranges.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Detected ScriptType
        """
        # Unicode ranges for Indic scripts
        script_patterns = {
            ScriptType.DEVANAGARI: re.compile(r'[\u0900-\u097F]'),
            ScriptType.BENGALI: re.compile(r'[\u0980-\u09FF]'),
            ScriptType.GURMUKHI: re.compile(r'[\u0A00-\u0A7F]'),
            ScriptType.GUJARATI: re.compile(r'[\u0A80-\u0AFF]'),
            ScriptType.ODIA: re.compile(r'[\u0B00-\u0B7F]'),
            ScriptType.TAMIL: re.compile(r'[\u0B80-\u0BFF]'),
            ScriptType.TELUGU: re.compile(r'[\u0C00-\u0C7F]'),
            ScriptType.KANNADA: re.compile(r'[\u0C80-\u0CFF]'),
            ScriptType.MALAYALAM: re.compile(r'[\u0D00-\u0D7F]'),
        }
        
        # Count matches for each script
        script_counts = {}
        for script, pattern in script_patterns.items():
            matches = pattern.findall(text)
            if matches:
                script_counts[script] = len(matches)
        
        if script_counts:
            # Return script with most matches
            return max(script_counts, key=script_counts.get)
        
        # Default to Latin for ASCII text
        return ScriptType.LATIN
    
    def _get_script_for_language(self, language: str) -> ScriptType:
        """
        Get script type for a language name.
        
        Args:
            language: Language name or code
            
        Returns:
            Associated ScriptType
        """
        normalized = language.lower().strip()
        return LANGUAGE_TO_SCRIPT.get(normalized, ScriptType.UNKNOWN)
    
    def calculate_task_complexity(
        self,
        text: str,
        grade_level: int,
        subject: str,
        language_count: int = 1,
        requires_cultural_context: bool = False,
        source_language: str = "English",
        target_languages: list = None
    ) -> TaskComplexity:
        """
        Calculate task complexity score with language-aware adjustments.
        
        This method now accounts for:
        - Script-specific token expansion (Indic scripts tokenize differently)
        - Morphological complexity (agglutinative languages like Tamil/Malayalam)
        - Cross-script translation difficulty
        
        Args:
            text: Input text
            grade_level: Target grade level (5-12)
            subject: Subject area
            language_count: Number of language pairs (for translation)
            requires_cultural_context: Whether cultural adaptation needed
            source_language: Source language for translation
            target_languages: List of target languages
            
        Returns:
            TaskComplexity object with language-aware metrics
        """
        target_languages = target_languages or []
        
        # Detect source script from text content
        source_script = self._detect_script_from_text(text)
        if source_script == ScriptType.LATIN and source_language:
            # If text is romanized, use language hint
            source_script = self._get_script_for_language(source_language)
        
        # Get target scripts
        target_scripts = [
            self._get_script_for_language(lang) 
            for lang in target_languages
        ]
        
        # Calculate script complexity factor
        source_metrics = SCRIPT_METRICS.get(source_script, SCRIPT_METRICS[ScriptType.UNKNOWN])
        
        # Get max target complexity (worst case for multi-language)
        target_multipliers = [
            SCRIPT_METRICS.get(script, SCRIPT_METRICS[ScriptType.UNKNOWN]).token_multiplier
            for script in target_scripts
        ] if target_scripts else [1.0]
        
        max_target_multiplier = max(target_multipliers)
        
        # Combined script complexity factor
        script_complexity_factor = (source_metrics.token_multiplier + max_target_multiplier) / 2
        
        # Calculate morphological complexity (affects model requirements)
        target_morph_factors = [
            SCRIPT_METRICS.get(script, SCRIPT_METRICS[ScriptType.UNKNOWN]).morphological_factor
            for script in target_scripts
        ] if target_scripts else [1.0]
        
        morphological_complexity = max(
            source_metrics.morphological_factor,
            max(target_morph_factors)
        )
        
        # Base token count estimation (Latin baseline)
        base_token_count = len(text) // 4
        
        # Adjusted token count with script multiplier
        adjusted_token_count = int(base_token_count * script_complexity_factor)
        
        # Technical subject increases complexity
        is_technical = subject in self.TECHNICAL_SUBJECTS
        
        # Calculate complexity score (0.0 - 1.0)
        score = 0.0
        
        # Token count factor (35% weight) - using adjusted tokens
        if adjusted_token_count < self.TOKEN_SMALL:
            score += 0.1
        elif adjusted_token_count < self.TOKEN_MEDIUM:
            score += 0.22
        elif adjusted_token_count < self.TOKEN_LARGE:
            score += 0.30
        else:
            score += 0.35
        
        # Grade level factor (15% weight)
        if grade_level <= self.GRADE_SIMPLE:
            score += 0.05
        elif grade_level <= self.GRADE_COMPLEX:
            score += 0.10
        else:
            score += 0.15
        
        # Subject factor (15% weight)
        if is_technical:
            score += 0.15
        else:
            score += 0.08
        
        # Language/script factor (20% weight) - enhanced
        script_score = 0.0
        if language_count > 1:
            script_score += 0.05
        if script_complexity_factor > 1.3:  # Significant script complexity
            script_score += 0.08
        if morphological_complexity > 1.3:  # Complex morphology
            script_score += 0.07
        score += min(script_score, 0.20)
        
        # Cultural context (10% weight)
        if requires_cultural_context:
            score += 0.10
        
        # Cross-script penalty (5% weight)
        if source_script != ScriptType.LATIN and target_scripts:
            cross_script_pairs = sum(1 for t in target_scripts if t != source_script)
            if cross_script_pairs > 0:
                score += min(0.05, cross_script_pairs * 0.02)
        
        complexity = TaskComplexity(
            token_count=base_token_count,
            adjusted_token_count=adjusted_token_count,
            grade_level=grade_level,
            subject_technical=is_technical,
            translation_pairs=language_count,
            requires_cultural_context=requires_cultural_context,
            complexity_score=min(score, 1.0),
            source_script=source_script,
            target_scripts=target_scripts,
            script_complexity_factor=script_complexity_factor,
            morphological_complexity=morphological_complexity
        )
        
        logger.debug(
            f"Task complexity: {complexity.complexity_score:.2f} "
            f"(tokens={base_token_count}→{adjusted_token_count}, "
            f"script_factor={script_complexity_factor:.2f}, "
            f"morph={morphological_complexity:.2f})"
        )
        
        return complexity
    
    def select_tier(
        self,
        complexity: TaskComplexity,
        force_tier: Optional[ModelTier] = None,
        available_memory_gb: Optional[float] = None
    ) -> ModelTier:
        """
        Select optimal model tier based on complexity and resources.
        
        Args:
            complexity: TaskComplexity object
            force_tier: Force specific tier (for testing)
            available_memory_gb: Available memory (if known)
            
        Returns:
            Selected ModelTier
        """
        if force_tier:
            logger.info(f"Forced tier: {force_tier}")
            return force_tier
        
        # Check available memory
        available = available_memory_gb or (self.max_memory_gb - self.current_memory_gb)
        
        # Decision logic based on complexity score
        if complexity.complexity_score < 0.3:
            # Simple task - use SMALL model
            tier = ModelTier.SMALL
        elif complexity.complexity_score < 0.6:
            # Medium complexity - prefer MEDIUM model
            if available >= self.MEMORY_BUDGET[ModelTier.MEDIUM]:
                tier = ModelTier.MEDIUM
            else:
                logger.warning(
                    f"Insufficient memory for MEDIUM ({available:.1f}GB < "
                    f"{self.MEMORY_BUDGET[ModelTier.MEDIUM]}GB), using SMALL"
                )
                tier = ModelTier.SMALL
        else:
            # High complexity - prefer LARGE or API
            if self.device_type == "cuda" and available >= self.MEMORY_BUDGET[ModelTier.LARGE]:
                tier = ModelTier.LARGE
            else:
                # For MPS/CPU with high complexity, use API
                logger.info(
                    f"High complexity task on {self.device_type}, routing to API"
                )
                tier = ModelTier.API
        
        logger.info(
            f"Selected tier: {tier.value} (complexity={complexity.complexity_score:.2f}, "
            f"available_memory={available:.1f}GB)"
        )
        
        return tier
    
    def get_model_config(self, tier: ModelTier) -> Dict[str, Any]:
        """
        Get model configuration for the selected tier.
        
        Args:
            tier: Selected ModelTier
            
        Returns:
            Model configuration dictionary
        """
        config = self.MODEL_CONFIGS.get(tier, self.MODEL_CONFIGS[ModelTier.API])
        
        # Add runtime info
        config["tier"] = tier.value
        config["device_type"] = self.device_type
        config["memory_budget_gb"] = self.MEMORY_BUDGET.get(tier, 0.1)
        
        return config
    
    def route_task(
        self,
        text: str,
        grade_level: int,
        subject: str,
        language_count: int = 1,
        requires_cultural_context: bool = False,
        force_tier: Optional[ModelTier] = None
    ) -> Tuple[ModelTier, Dict[str, Any], TaskComplexity]:
        """
        Complete routing: calculate complexity, select tier, return config.
        
        Args:
            text: Input text
            grade_level: Target grade level
            subject: Subject area
            language_count: Number of languages
            requires_cultural_context: Cultural adaptation needed
            force_tier: Force specific tier
            
        Returns:
            Tuple of (selected_tier, model_config, task_complexity)
        """
        # Calculate complexity
        complexity = self.calculate_task_complexity(
            text=text,
            grade_level=grade_level,
            subject=subject,
            language_count=language_count,
            requires_cultural_context=requires_cultural_context
        )
        
        # Select tier
        tier = self.select_tier(complexity, force_tier)
        
        # Get configuration
        config = self.get_model_config(tier)
        
        logger.info(
            f"Route decision: tier={tier.value}, model={config.get('model_id', 'api')}, "
            f"complexity={complexity.complexity_score:.2f}"
        )
        
        return tier, config, complexity
    
    def update_memory_usage(self, tier: ModelTier, loaded: bool):
        """
        Update current memory usage tracking.
        
        Args:
            tier: Model tier loaded/unloaded
            loaded: True if loaded, False if unloaded
        """
        memory = self.MEMORY_BUDGET.get(tier, 0.0)
        if loaded:
            self.current_memory_gb += memory
        else:
            self.current_memory_gb = max(0, self.current_memory_gb - memory)
        
        logger.debug(
            f"Memory usage: {self.current_memory_gb:.1f}GB / {self.max_memory_gb}GB "
            f"({self.current_memory_gb / self.max_memory_gb * 100:.1f}%)"
        )


# Global router instance (initialized in main.py)
_router: Optional[ModelTierRouter] = None


def get_router() -> ModelTierRouter:
    """Get global router instance (lazy init)."""
    global _router
    if _router is None:
        from ..core.config import settings
        from ..utils.device_manager import get_device_manager
        
        device_manager = get_device_manager()
        
        # Default to 8GB for M4, can be overridden via settings
        max_memory = getattr(settings, 'MAX_MODEL_MEMORY_GB', 8.0)
        
        _router = ModelTierRouter(
            max_memory_gb=max_memory,
            device_type=device_manager.device
        )
    
    return _router


def init_router(max_memory_gb: float = 8.0, device_type: str = "mps") -> ModelTierRouter:
    """Initialize global router (called at startup)."""
    global _router
    _router = ModelTierRouter(max_memory_gb=max_memory_gb, device_type=device_type)
    return _router
