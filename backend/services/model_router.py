"""A/B Testing framework for multi-model translation comparison."""
import random
import hashlib
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone
from enum import Enum

from sqlalchemy.orm import Session
from ..core.config import settings
from ..utils.logging import get_logger
from ..database import get_db_session
from ..models import ProcessedContent

logger = get_logger(__name__)


class ModelVariant(str, Enum):
    """Available model variants for A/B testing."""
    CONTROL = "control"  # Default model
    VARIANT_A = "variant_a"  # Alternative model 1
    VARIANT_B = "variant_b"  # Alternative model 2


class ModelRouter:
    """
    Routes translation requests to different model variants for A/B testing.
    
    Features:
    - Sticky user assignment (consistent variant per user)
    - Configurable traffic split percentages
    - Automatic quality tracking (BLEU scores)
    - Auto-rollback on quality degradation
    """
    
    def __init__(self):
        self.variants = {
            ModelVariant.CONTROL: {
                "model": "facebook/m2m100_418M",
                "percentage": 90,  # 90% traffic
                "bleu_threshold": 0.0,  # Baseline
            },
            ModelVariant.VARIANT_A: {
                "model": "facebook/nllb-200-distilled-600M",
                "percentage": 5,  # 5% traffic
                "bleu_threshold": -0.1,  # Max 10% quality drop
            },
            ModelVariant.VARIANT_B: {
                "model": "Helsinki-NLP/opus-mt-en-hi",
                "percentage": 5,  # 5% traffic
                "bleu_threshold": -0.1,
            }
        }
        
        # Quality tracking
        self.quality_scores: Dict[str, List[float]] = {
            variant.value: [] for variant in ModelVariant
        }
    
    def assign_variant(self, user_id: Optional[str] = None, text: str = "") -> ModelVariant:
        """
        Assign a model variant based on traffic split.
        
        Args:
            user_id: User identifier for sticky assignment
            text: Text being translated (used for hashing if no user_id)
            
        Returns:
            ModelVariant to use for this request
        """
        # Use user_id for consistent assignment, fallback to text hash
        hash_input = user_id or text
        hash_value = int(hashlib.md5(hash_input.encode()).hexdigest(), 16)
        percentage = hash_value % 100
        
        # Determine variant based on traffic split
        cumulative = 0
        for variant, config in self.variants.items():
            cumulative += config["percentage"]
            if percentage < cumulative:
                logger.info(
                    f"Assigned variant: {variant.value}",
                    extra={
                        "variant": variant.value,
                        "user_id": user_id,
                        "percentage": percentage
                    }
                )
                return variant
        
        # Fallback to control
        return ModelVariant.CONTROL
    
    def get_model_config(self, variant: ModelVariant) -> Dict[str, Any]:
        """Get model configuration for a variant."""
        return self.variants[variant]
    
    def track_quality(self, variant: ModelVariant, bleu_score: float):
        """
        Track translation quality for a variant.
        
        Args:
            variant: Model variant
            bleu_score: BLEU score for this translation
        """
        self.quality_scores[variant.value].append(bleu_score)
        
        # Keep only last 100 scores
        if len(self.quality_scores[variant.value]) > 100:
            self.quality_scores[variant.value] = self.quality_scores[variant.value][-100:]
        
        # Check for quality degradation
        if variant != ModelVariant.CONTROL:
            self._check_quality_threshold(variant)
    
    def _check_quality_threshold(self, variant: ModelVariant):
        """Check if variant quality has degraded below threshold."""
        variant_scores = self.quality_scores[variant.value]
        control_scores = self.quality_scores[ModelVariant.CONTROL.value]
        
        if len(variant_scores) < 10 or len(control_scores) < 10:
            return  # Not enough data
        
        # Calculate average scores
        variant_avg = sum(variant_scores[-10:]) / 10
        control_avg = sum(control_scores[-10:]) / 10
        
        quality_diff = variant_avg - control_avg
        threshold = self.variants[variant]["bleu_threshold"]
        
        if quality_diff < threshold:
            logger.warning(
                f"Variant {variant.value} quality degraded: "
                f"{quality_diff:.3f} below threshold {threshold}",
                extra={
                    "variant": variant.value,
                    "variant_avg": variant_avg,
                    "control_avg": control_avg,
                    "quality_diff": quality_diff
                }
            )
            
            # Auto-rollback: reduce traffic to 1%
            self.variants[variant]["percentage"] = 1
            self.variants[ModelVariant.CONTROL]["percentage"] = 94  # Redistribute
            
            logger.info(
                f"Auto-rollback triggered for {variant.value}. "
                f"Traffic reduced to 1%"
            )
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get A/B testing statistics."""
        stats = {}
        
        for variant in ModelVariant:
            scores = self.quality_scores[variant.value]
            config = self.variants[variant]
            
            stats[variant.value] = {
                "model": config["model"],
                "traffic_percentage": config["percentage"],
                "sample_count": len(scores),
                "avg_bleu": sum(scores) / len(scores) if scores else 0.0,
                "min_bleu": min(scores) if scores else 0.0,
                "max_bleu": max(scores) if scores else 0.0,
            }
        
        return stats
    
    def reset_variant(self, variant: ModelVariant):
        """Reset a variant to default configuration."""
        if variant == ModelVariant.CONTROL:
            self.variants[variant]["percentage"] = 90
        else:
            self.variants[variant]["percentage"] = 5
        
        self.quality_scores[variant.value] = []
        logger.info(f"Reset variant: {variant.value}")


# Global router instance
model_router = ModelRouter()


def calculate_bleu_score(reference: str, hypothesis: str) -> float:
    """
    Calculate BLEU score between reference and hypothesis.
    
    Args:
        reference: Ground truth translation
        hypothesis: Model-generated translation
        
    Returns:
        BLEU score (0.0 to 1.0)
    """
    try:
        from sacrebleu import sentence_bleu
        score = sentence_bleu(hypothesis, [reference])
        return score.score / 100.0  # Normalize to 0-1
    except Exception as e:
        logger.error(f"BLEU calculation failed: {e}")
        return 0.0


async def translate_with_ab_testing(
    text: str,
    source_lang: str,
    target_lang: str,
    user_id: Optional[str] = None,
    reference: Optional[str] = None
) -> Dict[str, Any]:
    """
    Translate text using A/B testing framework.
    
    Args:
        text: Text to translate
        source_lang: Source language code
        target_lang: Target language code
        user_id: Optional user identifier for sticky assignment
        reference: Optional reference translation for quality tracking
        
    Returns:
        Dictionary with translation result and metadata
    """
    # Assign variant
    variant = model_router.assign_variant(user_id, text)
    config = model_router.get_model_config(variant)
    
    # Perform translation (placeholder - integrate with actual translation service)
    # In production, this would call the appropriate model
    from ..translate.engine import TranslationEngine
    engine = TranslationEngine()
    
    try:
        translation = engine.translate(
            text=text,
            source_lang=source_lang,
            target_lang=target_lang,
            model_name=config["model"]
        )
        
        # Track quality if reference available
        if reference:
            bleu_score = calculate_bleu_score(reference, translation)
            model_router.track_quality(variant, bleu_score)
        
        return {
            "translation": translation,
            "variant": variant.value,
            "model": config["model"],
            "bleu_score": bleu_score if reference else None,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    
    except Exception as e:
        logger.error(f"Translation failed for variant {variant.value}: {e}")
        # Fallback to control variant
        if variant != ModelVariant.CONTROL:
            logger.info("Falling back to control variant")
            return await translate_with_ab_testing(
                text, source_lang, target_lang, user_id, reference
            )
        raise
